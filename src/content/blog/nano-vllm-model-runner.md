---
title: '一个 prompt 的一生：nano-vllm 从起服务到吐出 token'
description: '顺着 LLM(...) 和 engine.step() 两条线，把 nano-vllm 起服务时的进程拓扑、权重加载、显存规划、CUDA graph，以及一次推理里调度、张量准备、前向、采样、状态回写的全流程串成一张图。'
pubDate: '2026-07-30'
tags: ['llm', 'inference', 'vllm', 'nano-vllm', 'model-runner']
---

前面两篇分别拆过 [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) 的两个局部：

- [《同一个 Qwen3，两种写法：从 transformers 到 nano-vllm 的推理改造》](/blog/qwen3-inference-nano-vllm-vs-transformers/) 这篇介绍了 Qwen3 的模型结构和前向的计算。
- [《对照 Nano-vLLM 实现：读懂 vLLM 调度》](/blog/vllm-schedule-from-nano-vllm/) 这篇介绍了 `Scheduler` 和 `BlockManager` 怎么决定每轮跑谁。

但读完这两篇，仍然回答不了一个问题：**从 `LLM(path)` 这行代码开始，到 `outputs[0]["text"]` 拿到结果，中间到底发生了什么。** 这篇就主要回答这个问题，把整个流程串起来。

---

## 1 先给一个心智模型

整个引擎只有四个角色，职责边界非常干净：

```
  LLMEngine          总循环。tokenize、驱动 step、收集输出
      │
      ├── Scheduler  ──┐  决定「这一轮算哪些序列的哪些 token」
      │   BlockManager ┘  决定「这些 token 的 KV 放进哪个物理块」
      │
      └── ModelRunner    把上面两个决定翻译成张量,喂给模型,拿回 token
              │
              └── Qwen3ForCausalLM   一次纯函数式前向,对上面这些一无所知
```

一句话概括全流程：

> **调度器决定算什么，块管理器决定 KV 放哪，ModelRunner 把这两个决定编码成一组扁平张量和一个全局 Context，模型层在完全不知道 batching 和 paging 存在的前提下做一次前向，采样出 token，再把状态写回序列对象。**

时间上分两幕。第一幕是 `LLM(...)` 构造函数里一次性完成的准备工作，第二幕是 `generate()` 里反复执行的稳态循环：

```
第一幕:起服务(只发生一次)
  Config 解析 → 拉起 TP 进程 → 建模型 → 加载权重
  → warmup 探测显存 → 划分 KV cache → 捕获 CUDA graph

第二幕:一次 step(循环到所有序列结束)
  schedule → prepare 张量 → forward → sample → postprocess
```

---

## 2 第一幕：起服务

```python
llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)
```

`LLM` 本身直接继承 `LLMEngine`，什么也没加，我们看一下 `LLMEngine` 的初始化都做了什么：

```python
class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)
```

### 2.1 LLMEngine 的初始化

首先加载 Config，获取 `kvcache_block_size`、`tensor_parallel_size` 等配置。

如果 `config.tensor_parallel_size > 1`，即张量并行的情况下：

- 主进程用 `spawn` 拉起 `tp_size - 1` 个子进程，子进程的 `target` 直接就是 `ModelRunner` 这个类，没有包一层 worker 函数——**子进程的一生就是执行一遍这个构造函数**。而构造函数末尾 rank > 0 的分支会进入 `loop()` 死循环等命令，永远不返回，进程也就不会退出（见 2.2.5）。
- 主进程里直接用 `self.model_runner = ModelRunner(config, 0, self.events)` 构造 rank 0 的 `ModelRunner`。所以主进程身兼二职：**既是 engine，又是 0 号 worker**，省掉了 engine 和 rank 0 之间的一次进程间通信。
- 分工上，每个 rank 都持有一份完整的 `ModelRunner`——各自一份分片后的权重、各自一块 KV cache、各自一套 CUDA graph；而 `Tokenizer` 和 `Scheduler` 只存在于主进程里。也就是说**调度是全局唯一的，worker 只负责算**，子进程既不 tokenize 也不做任何调度决策。
- 主进程是唯一的发令者。它每次调 `model_runner.call("run", seqs, is_prefill)`，都会先把「调用哪个方法、参数是什么」pickle 进那块 1MB 的 `SharedMemory`，再 `set` 所有 `Event` 把子进程唤醒，然后自己也执行一遍同名方法。于是所有 rank 在同一时刻执行同一个方法、拿到同一批输入。具体实现见 3.3.1。

```
  tp_size = 2 时:

  主进程                                 子进程(rank 1)
  ┌─────────────────────────┐            ┌─────────────────────┐
  │ LLMEngine               │            │                     │
  │   ├─ ModelRunner(rank0) │◄─共享内存─►│ ModelRunner(rank 1) │
  │   ├─ Tokenizer          │            │  └─ loop() 等命令   │
  │   └─ Scheduler          │            │                     │
  └─────────────────────────┘            └─────────────────────┘
```

### 2.2 ModelRunner 的初始化

```python
class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()
```

#### 2.2.1 加载模型

**初始化通信组。** 即使 `tensor_parallel_size=1` 也会走 `init_process_group`。因为模型里的 `ColumnParallelLinear`、`VocabParallelEmbedding` 等模块在 `__init__` 里直接调 `dist.get_rank()` / `dist.get_world_size()`，没有进程组就报错。单卡时 `world_size=1`，所有分片逻辑自然退化成恒等操作。

**默认设备设成 cuda 再加载模型。** 模型构造时 `torch.empty(...)` 出来的空壳权重直接落在显存里，`load_model` 再把 safetensors 里的张量逐个 copy 进去。省掉了「先在 CPU 建一份完整模型再整体搬到 GPU」的那次全量拷贝，也省掉了对应的 CPU 内存峰值。

**权重加载与分片解耦。** `load_model` 本身只有十几行，做两件事：查 `packed_modules_mapping`，把 checkpoint 里分开的权重映射到融合后的参数上；然后调用每个 Parameter 上挂着的 `weight_loader`。

```
checkpoint 里的名字                模型里的参数
  ...self_attn.q_proj.weight  ─┐
  ...self_attn.k_proj.weight  ─┼─► ...self_attn.qkv_proj.weight
  ...self_attn.v_proj.weight  ─┘    (weight_loader 按 shard_id 写进不同偏移)

  ...mlp.gate_proj.weight     ─┐
  ...mlp.up_proj.weight       ─┴─► ...mlp.gate_up_proj.weight
```

分片逻辑（这个权重按行切还是按列切、本 rank 该拿哪一段）全部封装在各个 Linear 子类自己的 `weight_loader` 里。加载器不需要知道张量并行的存在，模型定义也不需要——每个 layer 自己对自己的权重负责。权重融合和 TP 切分的细节见[模型结构那篇](/blog/qwen3-inference-nano-vllm-vs-transformers/)。

#### 2.2.2 warmup_model：为了量显存而跑的一次前向

`warmup_model` 构造了一批**假的** `Sequence`（token 全是 0），在最大负载下跑一次完整的 prefill：

```python
seq_len  = min(max_num_batched_tokens, max_model_len)
num_seqs = min(max_num_batched_tokens // seq_len, max_num_seqs)
```

目的不是预热 kernel（虽然顺带也预热了），而是**测出模型在最大负载下激活值要占多少显存**。跑之前 `reset_peak_memory_stats()`，跑完就能从 `torch.cuda.memory_stats()` 里读到峰值，这个值会在 `allocate_kv_cache()` 计算可用于 KV cache 的显存时用到。

#### 2.2.3 KV cache：先算能开多少块，再挂到模型上

`allocate_kv_cache` 是第一幕的核心。它先算显存预算：

```python
free, total = torch.cuda.mem_get_info()
used = total - free
peak    = memory_stats()["allocated_bytes.all.peak"]
current = memory_stats()["allocated_bytes.all.current"]
num_kvcache_blocks = int(total * gpu_memory_utilization - used - peak + current) // block_bytes
```

拆开看 `used - peak + current`：`used` 是整张卡当前的占用（模型权重 + 其他进程 + 碎片），`peak - current` 是刚才 warmup 测出来的**激活值峰值余量**。所以这个式子的含义是：

```
可用于 KV cache = 总显存 × 利用率 － 当前稳态占用 － 激活峰值余量
```

而公式中的 `block_bytes` 是通过以下公式算出的，其中 `block_size` 代表一个 block 中能装下多少 token 的 KV cache，是块内 token 数：

```
block_bytes = 2 × num_layers × block_size × num_kv_heads × head_dim × dtype_size
              ↑                             ↑
            K 和 V                      注意这里已经除以了 tp_size
```

算出块数后，一次性 `torch.empty` 出整个 cache 池：

```
self.kv_cache = torch.empty(2, num_layers, num_kvcache_blocks, block_size, num_kv_heads, head_dim)
                            ↑  ↑          ↑                   ↑
                            │  │          │                   └ 块内 token 数
                            │  │          └ 全局块池
                            │  └ 每层
                            └ K/V
```

然后是**把 cache 注入进模型**：

```python
for module in self.model.modules():
    if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
        module.k_cache = self.kv_cache[0, layer_id]
        module.v_cache = self.kv_cache[1, layer_id]
        layer_id += 1
```

遍历所有子模块，凡是长得像 `Attention` 的（有 `k_cache` / `v_cache` 属性），就把全局池里属于它那一层的**视图**赋过去。不拷贝，只是 view。

#### 2.2.4 CUDA graph：只给 decode 捕获

如果没开 `enforce_eager`，最后一步是 `capture_cudagraph`。

decode 阶段每步只算 batch_size 个 token，GPU 上真正的计算量极小，反而是 Python 侧逐层 launch kernel 的开销成了瓶颈。CUDA graph 把整串 kernel 调用录制成一个可回放的图，一次 `replay()` 全部下发。

**为什么 launch kernel 会成为瓶颈？**

单个 decoder layer 走一遍要下发十来个算子：

```
  input_layernorm → qkv_proj → q_norm → k_norm → rope → store_kvcache
  → flash_attn → o_proj → post_attention_layernorm → gate_up_proj
  → silu_and_mul → down_proj                                  ≈ 12 个
```

Qwen3-8B 有 36 层，乘起来就是**四五百个 kernel**，再加上 embedding 和最后的 norm。eager 模式下每一个都要走一遍「Python 函数调用 → PyTorch dispatch → CUDA launch」，单个算子在 CPU 侧的开销大致是 10 μs 量级，几百个累起来就到了**毫秒量级**。而 kernel 下发是异步的，所以一步的实际耗时约等于 `max(CPU 下发时间, GPU 执行时间)`。当 batch_size 不大时，decode 一步的 GPU 时间和 CPU 下发时间是同一个量级——GPU 干完一个 kernel 就得停下来等 CPU 递下一条：

```
eager decode 一步:
  CPU  │█ launch ×~430 个,一个接一个 ────────────────────│  毫秒量级
  GPU  │ ░░▓░░▓░░▓░░▓░░▓░░▓ ...                        │  ▓算 ░等
        └── 每个 kernel 算完就闲着,等 CPU 下一条指令 ──┘

CUDA graph decode 一步:
  CPU  │█ 拷 5 个张量进 graph_vars + 一次 replay()       │  微秒量级
  GPU  │ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓                   │  连续跑,没有气泡
```

**CUDA graph 可以让 GPU 不必等 CPU。** 整个图在捕获时就已经把几百次 launch 固化成了一份可回放的指令序列，运行时 CPU 只需要把新数据拷进那几个静态张量，然后 `graph.replay()` 一次性交给 GPU。但 batch 越大，GPU 执行时间越长，下发开销的占比越小，CUDA graph 的收益也就越薄，不值得再多备一档图去占显存。

而 CUDA graph 要求**输入张量的地址和形状固定**，因此捕获时要做两件事：

**一是预分配一组静态张量。** `graph_vars` 里的 `input_ids`、`positions`、`slot_mapping`、`context_lens`、`block_tables`、`outputs` 都按最大规格预分配好。运行时不是「传新张量进去」，而是「把新数据写进这些固定张量的前 bs 行」，然后 replay。

**二是按 batch size 分档捕获。** 形状固定意味着一个图只能服务一个 batch size，于是捕获一组阶梯，捕获顺序是**从大到小**，第一个图创建的内存池被后续所有图共享（`graph_pool`），先捕获最大的能让池子一次到位；捕获的范围只到 `self.model(...)`，也就是**只到 hidden_states 为止，不含 lm_head**——因为 lm_head 的输出是 `[bs, vocab_size]`，词表十几万，做成静态张量太浪费，而且 prefill 和 decode 走 lm_head 的逻辑还不一样。

```python
graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
```

#### 2.2.5 收尾：还原全局默认值，建立通信通道

```python
    torch.set_default_device("cpu")
    torch.set_default_dtype(default_dtype)

    if self.world_size > 1:
        if rank == 0:
            self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
            dist.barrier()
        else:
            dist.barrier()
            self.shm = SharedMemory(name="nanovllm")
            self.loop()
```

**还原全局默认值。** 前面为了让模型权重直接建在显存里，把 `default_device` 改成了 `cuda`、`default_dtype` 改成了模型的 dtype。这两个都是**进程级的全局开关**，建完模型就得改回去，否则后面所有不写 `device=` 的张量都会默认落到 GPU 上，并且变成 bfloat16。

单卡的话到这里 `__init__` 就结束了，正常返回。

多卡的话，还需要建立控制面通道：rank 0 建一块共享内存，其他 rank 连上去。之后主进程正常返回、继续后面的工作，而子进程走到 `self.loop()` 就再也不返回了，进入「等 Event → 从共享内存读命令 → 执行 → 清 Event → 再等」这个死循环（3.3.1 会展开）。

至此第一幕结束，`LLM(path)` 返回。GPU 上现在有：一份完整的模型权重、一个划好的 KV cache 块池、一组录好的 CUDA graph。

---

## 3 第二幕：一次 step 里发生了什么

```python
class LLMEngine:

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens
```

`generate` 的骨架很朴素：把所有 prompt tokenize 成 `Sequence` 塞进调度器，然后不停 `step()` 直到队列清空。每次 `step()` 是四步：

```
  ① scheduler.schedule()        → (seqs, is_prefill)   决定算什么
  ② model_runner.call("run", …) → token_ids            真正的计算
  ③ scheduler.postprocess(…)                           状态回写
  ④ 收集 is_finished 的序列
```

### 3.1 `Sequence`：全系统唯一的状态载体

展开之前先明确一件事：**引擎里所有可变状态都挂在 `Sequence` 对象上**。调度器、块管理器、ModelRunner 都不持有序列状态，它们只是读写这个对象。

一条序列身上有用的字段就几个：

```
  token_ids             已有的全部 token(prompt + 已生成)
  num_cached_tokens     已经算完并存进 KV cache 的前缀长度
  num_scheduled_tokens  本轮要算多少个 token   ← 调度器每轮写
  block_table           [块号, 块号, ...]      ← 块管理器写
  status                WAITING / RUNNING / FINISHED
```

`num_cached_tokens` 和 `num_scheduled_tokens` 这一对是理解 chunked prefill 的钥匙：一条长 prompt 可能要好几轮才算完，每轮从 `num_cached_tokens` 处接着往后算 `num_scheduled_tokens` 个。

### 3.2 第一步：调度

`schedule()` 返回 `(seqs, is_prefill)`——**一轮要么全是 prefill 要么全是 decode，不混批**。调度器给每条序列写好 `num_scheduled_tokens`，块管理器给每条序列的 `block_table` 填好物理块号。这两个决定就是交给 ModelRunner 的全部输入。

调度策略本身（prefill 优先、token 预算、chunked prefill 的限制、抢占、prefix cache 命中）在[调度那篇](/blog/vllm-schedule-from-nano-vllm/)里已经拆过，这里只需要记住它的产出：

```
  输入: waiting / running 两个队列
  输出: 一批序列,每条都带着
          num_cached_tokens     —— 从第几个 token 开始算
          num_scheduled_tokens  —— 这轮算几个 token
          block_table           —— KV 往哪些物理块里写
        以及一个 is_prefill 标志
```

### 3.3 第二步：`model_runner.call("run", …)`

#### 3.3.1 多卡下 worker 间的通信

`model_runner` 调用 `run` 方法，我们先看下多卡时 worker 之间是如何通信的：

```python
def call(self, method_name, *args):
    if self.world_size > 1 and self.rank == 0:
        self.write_shm(method_name, *args)     # 序列化进共享内存,set 所有 event
    return getattr(self, method_name)(*args)   # rank 0 自己也执行一遍

def loop(self):                                # rank > 0 的一生
    while True:
        method_name, args = self.read_shm()    # event.wait() → 反序列化
        self.call(method_name, *args)
        if method_name == "exit": break
```

一块 1MB 的 `SharedMemory` 加一组 `multiprocessing.Event`，就实现了「主进程调用一个方法，所有 rank 同步执行」。没有 RPC 框架，没有队列，pickle 之后前 4 字节写长度，剩下写数据。

```
  rank 0 (主进程)                        rank 1 (子进程)
  ──────────────────────────────────────────────────────────
  call("run", seqs, True)
      │
      ├─ write_shm ──► [len][pickle(["run", seqs, True])]
      │                        │
      │                   event.set() ──────► event.wait() 返回
      │                                            │
      ├─ self.run(seqs, True)                      ├─ self.run(seqs, True)
      │       │                                    │       │
      │       └── 前向中的 all_reduce ◄──── NCCL ───┴───────┘
      │
      └─ sampler → token_ids                     返回 None(不采样)
```

两个细节值得看：

**只有 rank 0 采样。** `run()` 里 `prepare_sample` 和 `sampler` 都包在 `if self.rank == 0` 里。因为 `ParallelLMHead` 算完各自那片词表的 logits 后是用 `dist.gather` 收到 rank 0 的（不是 all_gather），其他 rank 手上根本没有完整 logits。这样省掉了一次全量 logits 的广播。

**`Sequence` 定制了 pickle 行为。** 广播序列对象时走的是 `__getstate__`：

```python
def __getstate__(self):
    last_state = self.last_token if not self.is_prefill else self.token_ids
    return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens,
            self.num_scheduled_tokens, self.block_table, last_state)
```

只传 6 个字段——`temperature`、`max_tokens` 这些采样参数不传（子进程不采样），`status` 不传（子进程不调度）。更狠的是 **decode 阶段只传 `last_token` 这一个整数，而不是整个 `token_ids` 列表**。decode 时每条序列的输入本来就只有最后一个 token，几百条序列每条省下一个上千元素的 list，而这是每步都要走一遍的路径。

这是很典型的工程取舍：为了一条每秒执行几十次的热路径，专门定制了序列化协议。

#### 3.3.2 执行计算：ModelRunner.run 方法

```python
class ModelRunner:

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids
```

`prepare_prefill` / `prepare_decode` 干的事是**把一批 Python 对象翻译成 GPU 能直接消费的扁平张量**。

**prefill 的产出**，以两条序列为例（假设 block_size=4，实际默认 256）：

```
  seq A: 这轮算 3 个 token, block_table=[7]
  seq B: 这轮算 2 个 token, block_table=[3]
                    ↓
  input_ids    [a0 a1 a2 | b0 b1]      全部拼成一维,没有 padding
  positions    [ 0  1  2 |  0  1]      每个 token 在自己序列里的位置
  cu_seqlens_q [0, 3, 5]               前缀和,标记每条序列的边界
  slot_mapping [28 29 30 | 12 13]      每个 token 的 KV 该写进哪个物理槽位
```

**`slot_mapping`** 是分页的「地址翻译表」：`slot_mapping[i]` 存的是被拍平的张量里第 i 个 token 的 KV 的地址。

然后是 `run_model` 方法：

```
  is_prefill or enforce_eager or bs > 512  →  普通 eager forward，CUDA graph 最多只捕获到 batch size 512
  否则                                      →  填 graph_vars,graph.replay()
```

关于 eager forward（RMSNorm 融合残差、qkv 融合、RoPE 查表、GQA），可以参见[《同一个 Qwen3，两种写法：从 transformers 到 nano-vllm 的推理改造》](/blog/qwen3-inference-nano-vllm-vs-transformers/)，里面有详细介绍。

最后是通过采样得到 `token_ids`。

### 3.4 第三步：状态回写，闭环

`postprocess` 把这一轮的结果写回 `Sequence`，同时决定序列的去留：

```
  for 每条序列:
      block_manager.hash_blocks(seq)          ← 把这轮填满的块登记进 hash 表
      seq.num_cached_tokens += num_scheduled_tokens
      seq.num_scheduled_tokens = 0

      if 还是 prefill 且没算完:  continue      ← chunk 中途不产 token
      seq.append_token(token_id)
      if 撞到 eos 或 达到 max_tokens:
          status = FINISHED
          block_manager.deallocate(seq)        ← 归还块
          running.remove(seq)
```

`hash_blocks` 是 prefix cache 的写入端：把这轮算完的完整块按内容算出 hash 登记进全局表，后续新请求如果前缀相同，`can_allocate` 就能查到并直接复用，跳过重算。

那句 `continue` 是 chunked prefill 的关键：一条长 prompt 分几轮算，只有算到最后一个 chunk 时才真正产生一个新 token，中间几轮的 logits 是被丢弃的。

到这里状态就闭环了：序列的 `num_cached_tokens` 前进了，KV cache 里多了一段内容，`token_ids` 可能多了一个 token，下一轮 `schedule()` 读到的就是新状态。

### 3.5 Context：元数据是怎么送到模型深处的

回头补一个 3.3.2 里跳过的问题：`prepare_prefill` / `prepare_decode` 只 return 了 `input_ids` 和 `positions`，那 `slot_mapping`、`cu_seqlens`、`block_tables` 这些元数据是怎么到达几十层深处的 `Attention.forward` 的？

答案是一个进程内全局唯一的 Context 对象——它就是 `nanovllm/utils/context.py` 模块顶层的一个变量，靠 Python「模块只加载一次」的性质天然成为单例。

```python
@dataclass(slots=True)
class Context:
    is_prefill: bool = False
    cu_seqlens_q / cu_seqlens_k: Tensor | None = None
    max_seqlen_q / max_seqlen_k: int = 0
    slot_mapping:  Tensor | None = None
    context_lens:  Tensor | None = None
    block_tables:  Tensor | None = None

_CONTEXT = Context()
def set_context(...):  global _CONTEXT; _CONTEXT = Context(...)
def reset_context():   global _CONTEXT; _CONTEXT = Context()
```

`set_context` 是**整体替换**而不是原地改字段，所以上一轮的残留值不会漏到下一轮。

---

## 4 把两幕接起来

最后用一张图把全流程串一遍：

```
═══ 第一幕:LLM(path) ═══════════════════════════════════════════════

  Config 解析 ─► spawn TP 进程 ─► init_process_group
       │
       ▼
  在 cuda 上建空壳模型 ─► load_model(safetensors → weight_loader 分片)
       │
       ▼
  warmup 跑一次最大 batch ──► 测得激活峰值
       │                          │
       ▼                          ▼
  allocate_kv_cache: 总显存×利用率 − 稳态占用 − 激活峰值 = 能开几个块
       │
       ├─► 一次性 empty 出块池,view 注入每层 Attention 的 k_cache/v_cache
       │
       ▼
  capture_cudagraph(按 bs 阶梯,只捕获 decode,不含 lm_head)
       │
       ▼
  num_kvcache_blocks 写回 config ──► Scheduler 据此建 BlockManager


═══ 第二幕:每次 step ═══════════════════════════════════════════════

  schedule()
    ├─ Scheduler:    这轮跑哪几条,每条算几个 token
    └─ BlockManager: KV 写进哪些物理块 → seq.block_table
       │
       ▼  (seqs, is_prefill)
  prepare_prefill / prepare_decode
    ├─ 序列压平 → input_ids / positions / cu_seqlens
    ├─ 逻辑块号翻译成物理槽位 → slot_mapping
    └─ set_context(...)  把这些元数据挂到全局
       │
       ▼
  model.forward(input_ids, positions)      ← 签名里只有这两个
    └─ Attention: get_context() → 写 KV cache → 调对应的 flash-attn 后端
       │
       ▼  hidden_states
  lm_head(只取每条序列末位) → logits → Sampler(Gumbel-max) → token_ids
       │
       ▼
  postprocess()
    ├─ hash_blocks: 填满的块登记进 prefix cache
    ├─ num_cached_tokens 推进
    └─ 撞 eos / max_tokens → FINISHED,归还块
       │
       └──────────► 回到 schedule(),直到队列清空
```

回头看，nano-vllm 能用一千多行把这套东西说清楚，靠的是几处很干净的边界：

- **模型层完全无状态**。签名只有 `(input_ids, positions)`，KV cache 靠外部注入，调度元数据靠全局 Context 递进去。paging 和 batching 的复杂度全被挡在模型之外，模型代码读起来和一个普通的 Qwen3 实现差不多。
- **`Sequence` 是唯一的状态容器**。三个组件都只是它的读写方，不存在第二份真相。
- **`Config` 当共享黑板**，让「显存探测」这种必须运行时才能确定的量，能从 ModelRunner 流回 Scheduler。
- **`slot_mapping` 把分页压成一个整数数组**，写 cache 的 kernel 因此可以完全不懂分页。
