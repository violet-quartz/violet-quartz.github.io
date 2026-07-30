---
title: '同一个 Qwen3，两种写法：从 transformers 到 nano-vllm 的推理改造'
description: '以 Qwen3-8B 为例，先把模型结构和每一处维度画清楚，再逐项对照 transformers 与 nano-vllm 的实现差异：权重融合、扁平批次、分页 KV cache、注意力后端、RoPE、残差融合与张量并行。'
pubDate: '2026-07-29'
tags: ['llm', 'inference', 'qwen3', 'transformers', 'nano-vllm']
---


最近在读 [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) 代码，它是以 Qwen3 模型为例进行推理的，看了一下模型的实现，又去看了一下经典的[transformers](https://github.com/huggingface/transformers) 中 Qwen3 模型的实现，两者有些不同，想对比一下；同时也借此梳理一下 Qwen3 的模型结构，熟悉了 LLM 的基本架构才能对推理的过程有个全面的认识。

首先我们以 Qwen3-8B 为例探究一下其模型结构和运算维度，再看看 [transformers](https://github.com/huggingface/transformers) 和 [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) 在对 Qwen3 模型的实现上有何不同。

---

## 1 大语言模型的基本架构

[The Transformer](https://arxiv.org/abs/1706.03762) 最初是一个 encoder-decoder 结构（机器翻译用），encoder 负责理解输入，decoder 负责生成输出，两者之间靠 cross-attention 连接。今天的大语言模型几乎全部只保留了 decoder 那一半，去掉 cross-attention，这就是 **decoder-only 架构**：一串同构的模块首尾相接堆叠起来，输入和输出共享同一个序列。

decoder-only 的核心机制是**自回归**：模型每次只预测下一个 token，把预测结果拼回输入，再预测下一个，如此循环。这决定了它必须用**因果注意力**（causal attention）—— 第 i 个位置只能看到 1..i，看不到未来，否则「预测下一个词」就变成了作弊。

### 1.1 decoder-only block

一个 decoder-only block 的通用结构：

```
  hidden_states
      │
      ├──────────── residual ──┐
      ▼  Norm                  │
      ▼  Self-Attention (causal)
      ▼  ◄──────────── + ──────┘
      │
      ├──────────── residual ──┐
      ▼  Norm                  │
      ▼  Feed-Forward Network
      ▼  ◄──────────── + ──────┘
      │
  hidden_states'
```

### 1.2 Self-Attention

上面图里的 Self-Attention 具体在算什么：每个 token 的向量先被三个矩阵分别投影成 **Q（query，我想找什么）**、**K（key，我能提供什么）**、**V（value，我实际携带的内容）**。拿 Q 和所有位置的 K 做点积，得到「这个位置该多关注哪些位置」的分数，除以 $\sqrt{d_k}$ 缩放、softmax 归一化成权重，再用这组权重对 V 加权求和，就是这个位置的输出：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

因果掩码就加在 softmax 之前：把第 i 行里 j > i 的位置全部设成 $-\infty$，softmax 之后这些位置的权重变成 0，未来的 token 对当前位置的输出没有任何贡献。

**多头**（multi-head）是把这件事平行做 $h$ 次：每个头各自拥有一份小尺寸的 Q/K/V 投影，独立算完注意力后把 $h$ 份输出拼接起来，再过一个输出投影混合。用意是让不同的头去捕捉不同种类的关系（比如有的头专注局部语法，有的头专注长距离指代），单头做不到这种分工。

Q、K、V 各自的头数并不要求相等 —— Qwen3 用的 GQA 正是让 K、V 的头数比 Q 少，多个 Q 头共享同一组 K、V，下一节会给出具体的头数和维度。

### 1.3 decoder-only 架构

decoder-only block 堆叠 N 层，前面接一个 embedding 把 token id 转成向量，后面接一个 norm 和一个线性层（lm_head）把向量转回词表上的分数（logits）。整个前向过程就是：

```
token id → embedding → [Norm → Attention → +] × N → Norm → lm_head → 下一个 token 的分数
                        └──────────────────┘
                              一层 decoder block
```


具体到每一代模型，这几个组件都在被持续替换和改进，但骨架三十年没变：

| 组件 | 早期 Transformer / GPT-2 | Qwen3 等现代 LLM |
|---|---|---|
| 归一化 | LayerNorm，后置（post-norm） | RMSNorm，前置（pre-norm），更省算力也更稳定 |
| 位置编码 | 可学习的绝对位置向量 / 正弦编码 | RoPE（旋转位置编码），作用在 Q/K 上而非输入embedding |
| 注意力头 | Multi-Head Attention（MHA），Q/K/V 头数相同 | Grouped-Query Attention（GQA），K/V 头数少于 Q，省 KV cache |
| FFN | 两层线性 + ReLU | SwiGLU：门控 + SiLU，两个升维矩阵并一个降维矩阵 |

下一节就以 Qwen3-8B 为例，把这张表里每一项换成具体的模块和数字。

## 2 Qwen3-8B 的模型结构

先把配置摊开（来自 `Qwen/Qwen3-8B` 的 `config.json`）：

```
hidden_size            4096          num_hidden_layers        36
num_attention_heads      32          num_key_value_heads       8
head_dim                128          intermediate_size     12288
vocab_size           151936          max_position_embeddings 40960
rope_theta              1e6          rms_norm_eps          1e-06
hidden_act             silu          tie_word_embeddings   false
torch_dtype        bfloat16
```

两个数值决定了这个模型的性格：`32 / 8 = 4`，即 **GQA 4:1**，每 4 个 Q head 共享一组 KV head；以及 `tie_word_embeddings: false`，意味着 `lm_head` 是一份**独立**的 `[151936, 4096]` 矩阵，不与 embedding 共享权重。

### 2.1 模型结构

下图中 `T` 是本次前向的 token 总数。右侧一列标的是**权重矩阵形状**（PyTorch 的 `[out_features, in_features]` 约定）。

```
                                                        张量形状        权重形状
  input_ids                                           [T]
      │
      ▼  embed_tokens                                              [151936, 4096]
  hidden                                              [T, 4096]
      │
  ┌───────────────────── Qwen3DecoderLayer ────────────────────────────────────┐
      │
      ├───────────────────────────── residual ───┐
      ▼  input_layernorm    RMSNorm              │                 [4096]
      │                                          │
      │  ┌ q_proj ──► q     [T, 32, 128]         │                 [4096, 4096]
      ├──┼ k_proj ──► k     [T,  8, 128]         │                 [1024, 4096]
      │  └ v_proj ──► v     [T,  8, 128]         │                 [1024, 4096]
      │      └── GQA 4:1，每 4 个 Q head 共用一组 KV
      │                                          │
      ▼  q_norm / k_norm    RMSNorm              │                 [128]
      │      └── Qwen3 特有：只归一化 head_dim 这一维，不是整个 hidden
      │                                          │
      ▼  RoPE (θ = 1e6)     仅作用于 q 和 k，形状不变
      │                                          │
      ▼  attention          [T, 32, 128] ──flatten──► [T, 4096]
      ▼  o_proj                                  │                 [4096, 4096]
      │                                          │
      ▼  ◄───────────────────── + ───────────────┘
      │
      ├───────────────────────────── residual ───┐
      ▼  post_attention_layernorm  RMSNorm       │                 [4096]
      │                                          │
      │  ┌ gate_proj ──► [T, 12288] ─┐           │                 [12288, 4096]
      ├──┤                           │           │
      │  └ up_proj   ──► [T, 12288] ─┤           │                 [12288, 4096]
      │                              ▼           │
      │                    SiLU(gate) × up       │
      │                       [T, 12288]         │
      ▼  down_proj  ◄──────────┘                 │                 [4096, 12288]
      │      [T, 4096]                           │
      ▼  ◄───────────────────── + ───────────────┘
      │
  └────────────────────────── × 36 层 ───────────────────────────────────────────┘
      │
      ▼  norm               RMSNorm                                [4096]
      ▼  lm_head                                                [151936, 4096]
  logits                                              [T, 151936]
```

nano-vllm 把其中两组权重**在模型结构层面就合并了**，checkpoint 里仍是分开存的，加载时才拼进去：

```
q_proj    [4096, 4096] ┐
k_proj    [1024, 4096] ├──►  qkv_proj      [6144, 4096]
v_proj    [1024, 4096] ┘

gate_proj [12288, 4096] ┐
up_proj   [12288, 4096] ┴──►  gate_up_proj  [24576, 4096]
```

### 2.2 存储计算

**参数量**

| 部分 | 计算 | 参数量 |
|---|---|---|
| embed_tokens | 151936 × 4096 | 0.62 B |
| 每层 attention | (4096 + 1024 + 1024 + 4096) × 4096 | 41.9 M |
| 每层 MLP | 12288 × 4096 × 3 | 151.0 M |
| 36 层合计 | 36 × 192.9 M | 6.95 B |
| lm_head | 151936 × 4096（未与 embedding 绑定） | 0.62 B |
| **总计** | | **≈ 8.19 B** |

bf16 下权重约 **16.4 GB**。

**KV cache**

这是推理引擎真正要操心的账：

```
每个 token 的 KV = 2 × 36 层 × 8 个 KV head × 128 × 2 字节
                = 147,456 字节
                = 144 KiB
```

于是：

| | 大小 |
|---|---|
| 1 个 token | 144 KiB |
| 一条 40960 的满上下文 | **6.0 GB** |
| nano-vllm 的一个块（256 token） | 36 MiB |

**一条序列跑满上下文，KV 就要吃掉 6 GB，而整个模型权重才 16.4 GB。** 几条长序列并发就能把显存压垮。

GQA 在这里已经帮了大忙：如果是 MHA（32 个 KV head），上面所有数字要乘以 4。

---

## 3 transformers 和 nano-vllm 对 Qwen3 模型的不同实现

[transformers](https://github.com/huggingface/transformers) 和 [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) 对 Qwen3 模型实现的入口都是 `class Qwen3ForCausalLM`，我们先来看一下这个类的定义：

```python
# nano-vllm
class Qwen3ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.model = Qwen3Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)
```

```python
# transformers
@auto_docstring
class Qwen3ForCausalLM(Qwen3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    _fsdp_plan = {"lm_head": "keep_full_weight"}

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        logits_to_keep=0,
        **kwargs,
    ):
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
            past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss, logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```


从 `Qwen3ForCausalLM` 的定义上可以看出一些差异：

- nano-vllm 放弃了训练功能，只支持推理，同时在返回值中没有 `output_attentions`、`output_hidden_states` 这类结构化返回，是只关注推理基本功能的版本。
- nano-vllm 的 `forward` 和 `compute_logits` 被拆成了两步，这样做是为了让 run_model 能在两者之间插入「decode 走重放图」这个动作。在录制图时，由于 `lm_head` 参数占用较大，图里只录入 `self.model(...)` 这一段（embed → 36 层 → norm），`lm_head` 留在图外单独调用；decode 的重放路径也是基于这一设计，以节省推理时间。
- transformers 用 `_tp_plan`、`_pp_plan`、`_fsdp_plan` 三个类属性声明 `lm_head` 在张量并行、流水线并行、FSDP 下各自该怎么切分——模型定义和并行策略是分离的。nano-vllm 没有这类声明，并行方式直接编码在用的是哪个层类（`ParallelLMHead` 本身就是并行的）里（第 4 节详细对比）。


我们再往里一层，看一下核心模块 `Qwen3Model`：


```python
# nano-vllm
class Qwen3Model(nn.Module):
    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states
```

```python
# transformers
@auto_docstring
class Qwen3Model(Qwen3PreTrainedModel):
    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        **kwargs,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)          # 懒初始化的 KV cache

        if position_ids is None:                                        # 位置也要现推
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):  # 现场构造 mask
            mask_kwargs = {"config": self.config, "inputs_embeds": inputs_embeds,
                           "attention_mask": attention_mask, "past_key_values": past_key_values,
                           "position_ids": position_ids}
            causal_mask_mapping = {"full_attention": create_causal_mask(**mask_kwargs)}
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for i, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )
```

对比下来，两者结构基本一致，`__init__` 里都是 embedding + 一串 `DecoderLayer` + 最后一个 norm，`forward` 结构都是先 embed tokens、再 forward `DecoderLayer`、最后再来 norm。

下面我们深入阅读代码后，对两者的不同进行总结，这里我们主要关注 model 实现本身的不同，涉及到和 model runner 相关的部分会在其他文章中进行阐述：

### 3.1 nano-vllm 对权重进行融合：5 次 GEMM 压成 2 次

transformers 里 q/k/v 是三个独立的 `nn.Linear`，MLP 的 gate/up 也是两个：

```python
# transformers
self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=attention_bias)
self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)

def forward(self, x):                       # Qwen3MLP
    return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

三者吃的是**同一个** `hidden_states`，输出维度也对齐，完全可以拼成一次矩阵乘法。nano-vllm 就是这么做的：

```python
# nano-vllm
self.qkv_proj = QKVParallelLinear(hidden_size, head_dim,
                                  total_num_heads, total_num_kv_heads, bias=qkv_bias)

qkv = self.qkv_proj(hidden_states)                              # 一次 GEMM
q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
```

```python
# nano-vllm Qwen3MLP
gate_up = self.gate_up_proj(x)      # 一次 GEMM 出 [T, 24576]
x = self.act_fn(gate_up)            # SiluAndMul：chunk 成两半 → silu(a) * b
x = self.down_proj(x)
```

收益有三重：kernel 启动次数少了，`hidden_states` 只从显存读一遍，矩阵更大也更容易打满 GPU。

代价是加载权重时要做名字改写和分段写入 —— checkpoint 里存的仍是 `q_proj` / `k_proj` / `v_proj`。nano-vllm 用一张映射表处理：

```python
packed_modules_mapping = {
    "q_proj": ("qkv_proj", "q"),
    "k_proj": ("qkv_proj", "k"),
    "v_proj": ("qkv_proj", "v"),
    "gate_proj": ("gate_up_proj", 0),
    "up_proj": ("gate_up_proj", 1),
}
```

加载器看到 `...q_proj.weight`，就把名字换成 `qkv_proj` 并附带一个 `shard_id`，由该层自定义的 `weight_loader` 决定写进融合矩阵的哪一段。

### 3.2 transformers vs nano-vllm 的批次布局：从 `[B, S, H]` 到扁平的 `[T, H]`

这是两份代码在**每一个张量形状**上都不同的根源。
transformers 用标准的三维批次。一批里序列长度不同，就 padding 到最长，再用 `attention_mask` 把补出来的位置屏蔽掉：

```python
# transformers：一切都是 [batch, seq, ...]
hidden_states                            # [B, S, 4096]
query_states = ....transpose(1, 2)       # [B, 32, S, 128]
causal_mask_mapping = {"full_attention": create_causal_mask(**mask_kwargs)}
```

nano-vllm 把所有序列的 token **首尾相接拍成一维**，完全不 padding，改用累积长度表标记边界：

```python
# nano-vllm：一切都是 [总 token 数, ...]
input_ids     # [T]                  三条序列 300/400/255 拼接 → T = 955
cu_seqlens_q  # [0, 300, 700, 955]   边界表
hidden        # [T, 4096]
q             # [T, 32, 128]
```

假设一批里混着 100 token 和 4000 token 的序列，padding 方案要按 4000 算，算力被浪费在无用的地方，而扁平方案就没有这方面的浪费。

但扁平方案的代价是 mask 没法再用一个矩阵表达。`[B, S, H]` 布局下每条序列各有一个独立的 `S×S` 因果矩阵，隔离在 batch 维里；拍平之后 batch 维消失，所有 token 挤在同一根轴上，mask 就得是一个**块对角 + 块内下三角**的 `[T, T]` 矩阵：

```
拿 A(3) B(2) C(2) 三条序列，T = 7

           A0 A1 A2 │ B0 B1 │ C0 C1
      A0 [  ✓  ·  ·  │  ✗  ✗  │  ✗  ✗ ]
      A1 [  ✓  ✓  ·  │  ✗  ✗  │  ✗  ✗ ]   ✓ 可见
      A2 [  ✓  ✓  ✓  │  ✗  ✗  │  ✗  ✗ ]   · 被因果性挡住
         ─────────────┼────────┼─────────  ✗ 被跨序列隔离挡住
      B0 [  ✗  ✗  ✗  │  ✓  ·  │  ✗  ✗ ]
      B1 [  ✗  ✗  ✗  │  ✓  ✓  │  ✗  ✗ ]
         ─────────────┼────────┼─────────
      C0 [  ✗  ✗  ✗  │  ✗  ✗  │  ✓  · ]
      C1 [  ✗  ✗  ✗  │  ✗  ✗  │  ✓  ✓ ]

有效的只有对角线上那三个小三角，其余全是浪费
```

这带来两个问题。**尺寸从「各自平方之和」变成「总和的平方」**：$n$ 条等长序列正好差 $n$ 倍，64 条 512 token 的序列，分开是 64 × 512² ≈ 33 MB，扁平后是 32768² ≈ 2 GB，根本物化不出来——而扁平布局的意义恰恰是多塞几条序列进来，两件事直接冲突。**而且 `is_causal=True` 这条捷径失效了**：它只会按全局下标做下三角，于是 `B0` 会看到 `A0/A1/A2`，跨序列串味。

nano-vllm 的解法是根本不造 mask，把边界信息直接交给 kernel：

```python
flash_attn_varlen_func(q, k, v,
                       cu_seqlens_q=..., cu_seqlens_k=...,   # 边界表 [0, 300, 700, 955]
                       causal=True)                          # 每个块内部各自因果
```

kernel 读 `cu_seqlens` 就知道哪几段属于同一条序列，只在段内做因果注意力，段之间压根不计算——那些 ✗ 的位置不是被加了 $-\infty$，而是从未进入计算。

代价是后端被锁死：eager 和 sdpa 都要求 `[B, H, S, D]` 的形状，且只认显式 mask 张量或 `is_causal` 标志，没有任何参数能接受边界表，扁平张量喂进去第一关就过不了。只有 FlashAttention 的 varlen 系列认 `cu_seqlens`。

有意思的是，transformers 能自由切后端，并不是因为它不用 varlen——它选 flash 后端时内部**也会**转成 varlen（`modeling_flash_attention_utils.py` 里的 `_unpad_input` 根据 `attention_mask` 算出 `cu_seqlens`、抠掉 padding、调 `flash_attn_varlen_func`，最后再 `pad_input` 填回去）。真正的区别是**哪种表示是「正典」**：

```
transformers:   [B, S, H] + mask  ←── 正典，模型代码全程用它
                     │
                     ├── eager   直接用
                     ├── sdpa    直接用
                     └── flash   在 attention 边界处 unpad → varlen → repad

nano-vllm:      [T, H] + cu_seqlens  ←── 正典，从 prepare_prefill 到 kernel 一路到底
                     │
                     └── flash varlen   只此一条
```

transformers 保留了那个「所有后端都懂的最小公分母」，代价是 padding 的算力浪费，以及 flash 路径上多一次 unpad/repad 的来回搬运。

### 3.3 transformers vs nano-vllm 的 KV cache：从 `torch.cat` 到分页块池

这是差距最大的一处。

transformers 的 `DynamicCache` 逐层持有两个张量，每步解码用 `torch.cat` 把新的 K/V 拼到尾巴上：

```python
# transformers DynamicLayer.update
self.keys = torch.cat([self.keys, key_states], dim=-2)
self.values = torch.cat([self.values, value_states], dim=-2)
return self.keys, self.values
```

语义清晰，但对在线服务有三个硬伤：

1. **每步都重新分配并拷贝整个 cache** —— 序列越长越慢，是 O(S) 的拷贝
2. **每条序列的 KV 必须连续** —— 显存碎片化严重，且总量无法预知
3. **前缀无法共享** —— 一百个请求带同样的 system prompt，就存一百份

nano-vllm 换成 PagedAttention 那一套：启动时一次性开出全部 KV cache，之后按固定大小的块分配。

```python
# nano-vllm：启动时一把开完，之后再不向 CUDA 申请显存
self.kv_cache = torch.empty(2, num_layers, num_kvcache_blocks,
                            block_size, num_kv_heads, head_dim)
```

每条序列持有一张 `block_table`（逻辑块号 → 物理块号的页表），KV 在显存里可以散落在任意位置。写入时不再 `cat`，而是由一个 triton kernel 按预先算好的槽位号直接落盘：

```python
# nano-vllm Attention.forward
if k_cache.numel() and v_cache.numel():
    store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
```

`slot_mapping` 是 CPU 侧算好的一张「token → cache 行号」表，行号 = `物理块号 × block_size + 块内偏移`。kernel 只管查表写入，分页寻址的复杂度全部挡在了 GPU 之外。

三个硬伤于是对应解决：写入是 O(1) 的定点写；块定长，显存零碎片；块带引用计数和前缀哈希，**相同前缀的序列可以指向同一个物理块**，KV 只存一份。

### 3.4 transformers vs nano-vllm 的注意力：`repeat_kv` 与两个不同的 FlashAttention 入口

GQA 下 K/V 只有 8 个 head，而 Q 有 32 个。transformers 的 eager 实现选择**把 K/V 物化成 32 份**再做标准注意力：

```python
# transformers
def repeat_kv(hidden_states, n_rep):
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
```

`expand` 本身是零拷贝的视图，但后面的 `reshape` 会真的分配 4 倍显存并复制。这条路径只在 eager 后端上走 —— transformers 通过 `ALL_ATTENTION_FUNCTIONS` 派发，选 sdpa 或 flash 后端时会绕开它。

nano-vllm 干脆不做这一步，FlashAttention 原生支持 GQA，head 的映射关系在 kernel 内部处理。而且它按 prefill / decode 分成两个不同的入口：

```python
# nano-vllm Attention.forward
if context.is_prefill:
    if context.block_tables is not None:      # 有前缀缓存或分块预填
        k, v = k_cache, v_cache               # K/V 改从分页 cache 取
    o = flash_attn_varlen_func(q, k, v,
                               cu_seqlens_q=..., cu_seqlens_k=...,
                               causal=True, block_table=context.block_tables)
else:                                         # decode
    o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                cache_seqlens=context.context_lens,
                                block_table=context.block_tables, causal=True)
```

- **prefill** 用变长接口，靠 `cu_seqlens` 切分扁平批次里的各条序列
- **decode** 用 `flash_attn_with_kvcache`，Q 只有一个 token，K/V 全部由 kernel 按页表从 cache 里取

值得留意中间那两行：prefill 时如果存在前缀缓存或分块预填，手上刚算出来的 `k` 只覆盖新 token，历史部分在分页 cache 里，所以要把 `k, v` **整个换成 cache**，让 FlashAttention 拿页表自己去取。而这能成立，是因为上一步的 `store_kvcache` 已经把新 token 写进 cache 了 —— 先写后读，缺一不可。

### 3.5 transformers vs nano-vllm 的RoPE：每步现算 vs 运算时查表

两边的数学完全等价，实现风格差得很远。

transformers 每次前向都从 `inv_freq` 现算 cos/sin，用的是 `rotate_half` 形式：

```python
# transformers
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
```

注意 `Qwen3RotaryEmbedding.forward` 里有一句 `emb = torch.cat((freqs, freqs), dim=-1)` —— cos/sin 被复制成两份拼到 128 维，好和 `rotate_half` 配合。

nano-vllm 在**初始化时就把整张表算完**存成 buffer，前向只做一次索引；旋转直接写成 2×2 矩阵的展开式：

```python
# nano-vllm，__init__ 中
freqs = torch.einsum("i,j -> ij", t, inv_freq)          # [max_pos, 64]
cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)     # [max_pos, 1, 128]
self.register_buffer("cos_sin_cache", cache, persistent=False)

# forward 中
cos_sin = self.cos_sin_cache[positions]                 # 查表，无三角运算
cos, sin = cos_sin.chunk(2, dim=-1)

def apply_rotary_emb(x, cos, sin):
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin                            # 旋转矩阵直接展开
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)
```

展开一下就能看出两者等价：transformers 的 `cos` 前后两半相同，所以 `q * cos + rotate_half(q) * sin` 的前半是 `x1·cos − x2·sin`、后半是 `x2·cos + x1·sin`，与 nano-vllm 逐字相同。

差别在开销：nano-vllm 的 cos/sin 只有 64 维（不复制成 128），也不需要每步做 `cat` 和三角函数运算。代价是启动时要为全部 40960 个位置预计算，并常驻一块20 MB 的表 —— 用显存换每步的延迟。

### 3.6 nano-vllm 的 RMSNorm：把残差加法融进归一化

transformers 的 RMSNorm 只做归一化，残差在外面加：

```python
# transformers Qwen3DecoderLayer.forward
residual = hidden_states
hidden_states = self.input_layernorm(hidden_states)
hidden_states, _ = self.self_attn(...)
hidden_states = residual + hidden_states
```

nano-vllm 让 RMSNorm 多接一个 `residual` 参数，把加法和归一化融进同一个融合 kernel，同时返回新的残差：

```python
# nano-vllm RMSNorm
@torch.compile
def add_rms_forward(self, x, residual):
    x = x.float().add_(residual.float())     # ① 补上一个块欠下的残差加法
    residual = x.to(orig_dtype)              # ② 这个和成为新的残差基准
    var = x.pow(2).mean(dim=-1, keepdim=True)
    x.mul_(torch.rsqrt(var + self.eps))      # ③ 归一化，供当前块使用
    return x.to(orig_dtype).mul_(self.weight), residual

# nano-vllm Qwen3DecoderLayer.forward
def forward(self, positions, hidden_states, residual):
    if residual is None:
        hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
    else:
        hidden_states, residual = self.input_layernorm(hidden_states, residual)
    hidden_states = self.self_attn(positions, hidden_states)
    hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
    hidden_states = self.mlp(hidden_states)
    return hidden_states, residual
```

理解这段代码的关键是：**在层与层之间流动时，真正的隐状态不是 `hidden_states`，而是 `hidden_states + residual`** —— 那个加法被故意推迟了，推到下一个 norm 里去做。


**省下来的是什么。** `[T, 4096]` 的 bf16 张量，每个 token 占 8 KiB。算一下「残差加 + 下一个 norm」这一对操作的显存流量：

```
写法 A：加法在 norm 外面（transformers 的结构）
  add:   读 h(8) + 读 r(8)  →  写 s(8)      = 24 KiB/token
  norm:  读 s(8)            →  写 out(8)    = 16 KiB/token
                                     合计    40 KiB/token

写法 B：加法在 norm 里面（nano-vllm 的 add_rms_forward）
  一个融合 kernel: 读 h(8) + 读 r(8) → 写 out(8) + 写 s(8)   = 32 KiB/token
```

省掉的是**对加法结果 `s` 的那一次读回**：`h + r` 算完就留在片上直接拿去求方差，不必落到 HBM 再读上来。注意写还是要写的——那个和正是下一个 norm 要用的 `residual`，必须落盘；省的只是往返里的「返」。36 层 × 每层 2 处，累积起来不算小。


代价是层与层之间要显式传递 `residual`，第一层还得特判 `residual is None`，而且「真正的隐状态是两个变量之和」这个隐含约定不看仔细很容易读错，用代码可读性换带宽。

---

### 3.7 张量并行：两边都有，路线不同

一个常见的误解是「transformers 不支持张量并行」。**它是支持的**，而且做得相当完整，只是表达方式和 nano-vllm 截然不同。

#### transformers：配置里的声明式 plan

Qwen3 的切分方案直接写在 config 类里：

```python
# configuration_qwen3.py
base_model_tp_plan = {
    "layers.*.self_attn.q_proj": "colwise",
    "layers.*.self_attn.k_proj": "colwise",
    "layers.*.self_attn.v_proj": "colwise",
    "layers.*.self_attn.q_norm": "replicated_with_grad_allreduce",
    "layers.*.self_attn.k_norm": "replicated_with_grad_allreduce",
    "layers.*.self_attn.o_proj": "rowwise",
    "layers.*.mlp.gate_proj": "colwise",
    "layers.*.mlp.up_proj": "colwise",
    "layers.*.mlp.down_proj": "rowwise",
}
```

```python
# modeling_qwen3.py
class Qwen3ForCausalLM(Qwen3PreTrainedModel, GenerationMixin):
    _tp_plan = {"lm_head": "colwise_gather_output"}
```

模型代码里全是普通的 `nn.Linear`，一行并行逻辑都没有。加载时打开对应开关（`tp_plan="auto"` 之类），框架读这张表，把匹配到的模块用相应策略包装成 DTensor 分片。

策略本身在 `integrations/tensor_parallel.py` 里注册，目前有二十来种：

```python
"colwise", "rowwise", "colwise_gather_output", "rowwise_split_input",
"packed_colwise", "packed_rowwise",                   # 给融合权重用的
"embedding_rowwise", "embedding_colwise",
"sequence_parallel", "replicated_with_grad_allreduce",
"ep_router", "moe_tp_experts", "grouped_gemm", ...    # MoE 相关
```

通信被封装成 autograd function，**前向反向都有**：

```python
class _AllReduceBackward(torch.autograd.Function):
    """Identity forward, all-reduce backward. Used before colwise layers (f in Megatron)."""

class _AllReduceForward(torch.autograd.Function):
    """All-reduce forward, identity backward. Used after rowwise layers (g in Megatron)."""
```

#### nano-vllm：并行写进层的类型里

nano-vllm 没有 plan，切分方式就是**你用哪个类**：

```python
self.qkv_proj     = QKVParallelLinear(...)           # 按输出维切
self.o_proj       = RowParallelLinear(...)           # 按输入维切
self.gate_up_proj = MergedColumnParallelLinear(...)
self.down_proj    = RowParallelLinear(...)
```

通信直接写死在 `forward` 里：

```python
class RowParallelLinear(LinearBase):
    def forward(self, x):
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y
```

（那个 `if self.tp_rank == 0` 是个容易漏掉的细节：`all_reduce` 会求和，如果每张卡都加 bias，加完就变成了 N 份，所以只让一张卡加。）

权重切分由每个类自己的 `weight_loader` 完成，加载时从完整 checkpoint 里 `narrow` / `chunk` 出属于本 rank 的那一片。


## 4 nano-vllm 放弃了什么

nano-vllm 确实在模型推理上做了不少优化改造，但是与 transformers 相比，它也放弃了不少功能：

**训练。** 没有反向传播，没有梯度，没有 `loss_function`，`@torch.inference_mode()` 贯穿始终。

**采样策略。** 只有 temperature 加一次 Gumbel-max 采样，一个 kernel 搞定。没有 top-p、top-k、repetition penalty、beam search，也没有 transformers 那套可插拔的 `LogitsProcessor` 流水线。

**注意力后端。** 写死 FlashAttention。transformers 通过 `ALL_ATTENTION_FUNCTIONS` 在 eager / sdpa / flash / flex 之间派发，能跑在没有 FA 的硬件上，也能在需要时吐出注意力权重。nano-vllm 两者都做不到。

**可观测性。** 没有 `output_attentions`、`output_hidden_states`，没有 `BaseModelOutputWithPast` 这类结构化返回。想看中间层结果就得改代码。

**位置编码变体。** `assert rotary_dim == head_size` 直接排除了 partial RoPE；`rope_scaling` 只认得 `rope_theta` 一个字段，YaRN 之类的长上下文外推方案没有实现。

**滑动窗口注意力。** Qwen3 的 config 里有 `sliding_window` / `layer_types` 字段，transformers 会据此构造两套 causal mask，nano-vllm 完全忽略。

**量化。** 没有 int8 / fp8 / GPTQ / AWQ，只跑 bf16。

**输入形式。** 只接受 `input_ids`，不支持 `inputs_embeds`，也没有 encoder-decoder 和多模态。

这说明：**transformers 是一个要覆盖所有情况的库，nano-vllm 是一份把单一场景做到极致的示范代码。** 