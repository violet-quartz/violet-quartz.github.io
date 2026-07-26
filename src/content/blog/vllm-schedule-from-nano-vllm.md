---
title: '对照 Nano-vLLM 实现：读懂 vLLM 调度'
description: '先用轮次快照把 nano-vllm 的调度看透，再对照 vLLM V1 的 schedule()，讲清混批、chunked prefill、token 预算与抢占这几处关键分歧。'
pubDate: '2026-07-26'
tags: ['llm', 'inference', 'vllm', 'paged-attention', 'schedule']
---

[Nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) 是一个轻量级的大语言模型（LLM）推理引擎实现，代码非常简洁，仅有 1200 行的 Python 和 Triton 代码，旨在从头开始重建著名推理框架 vLLM 的核心架构。它的主要定位并非替代生产级的 vLLM，而是作为一个教育和学习工具，帮助开发者深入理解 vLLM 推理引擎的内部工作原理。

## 1 Nano-vLLM 调度机制

Nano-vLLM 的调度实现在 nanovllm/engine/scheduler.py 和 nanovllm/engine/block_manager.py 中。

### 1.1 基本原理

调度由两个角色分工完成，读代码时把它们分开看会清楚很多：

- **`Scheduler`** 决定*谁这轮跑、跑多少 token*。它维护 `waiting` 和 `running` 两个队列，只做排队决策，自己不碰一寸显存。
- **`BlockManager`** 决定*显存装不装得下*。KV cache 被切成固定大小的块（默认 256 个 token 一块），它管着这些块的分配、释放、共享和回收。

调度器每想排一条序列，都得先问 BlockManager 一句「它的 KV 放得下吗」。放得下就排上，放不下就只能等，或者牺牲别人。

**`schedule()` 的主干。** 一次调度**要么全是 prefill，要么全是 decode**，两者不混批。关键在 `Scheduler.schedule()` 里 prefill 循环结束后那句提前 return——只要抓到了哪怕一条序列，就立刻返回，decode 分支根本不执行。

```
                        schedule()
                            │
                            ▼
            ┌───────────────────────────────┐
            │  prefill 循环                  │
            │  从 waiting 队头依次取序列       │
            │  受 max_num_seqs 和            │
            │     max_num_batched_tokens 限制 │
            └───────────────┬───────────────┘
                            │
                  ┌─────────┴─────────┐
        抓到了 ≥1 条              一条都没抓到
                  │                   │
                  ▼                   ▼
          return (seqs, True)  ┌──────────────────────┐
          ↑                    │  decode 循环          │
          │                    │  从 running 队头取     │
     prefill 有绝对优先权,      │  每条只算 1 个 token   │
     会打断正在进行的 decode     │  块不够 → 抢占队尾     │
                               └──────────┬───────────┘
                                          ▼
                                 return (seqs, False)
```

**两者之间的接口。** BlockManager 对外只有六个方法，调度的一轮就是按下面这个顺序把它们串起来：

```
  Scheduler                                       BlockManager
 ─────────────────────────────────────────────────────────────────────────────
  prefill 循环
    这条序列的 KV 装得下吗?     ──can_allocate(seq)──►  数要几个新块,
    顺带问:前缀有缓存吗?        ◄── -1 / 命中的块数 ───  同时查 hash_to_block_id
    装得下 → 分配               ──allocate(seq, n)───►  填 seq.block_table,
                                                        命中的块 ref_count += 1
    装不下 → break,继续等

  decode 循环
    再吐 1 个 token 还够吗?     ──can_append(seq)────►  只在跨块边界时才要
                               ◄──── True / False ───   新块,看 free_block_ids
    够 → 分配                   ──may_append(seq)────►  需要时补 1 个块
    不够 → 抢占队尾那条          ──deallocate(受害者)─►  归还它的全部块

  每轮算完 (postprocess)
    登记这轮填满的块            ──hash_blocks(seq)───►  写进 hash_to_block_id,
                                                        供后来者做 prefix cache
```

「装得下吗」在两个阶段问法不同，因为需求量差得远：prefill 要一次性安置整条 prompt，可能是几十个块；decode 每次只多算 1 个 token，最多只需要 1 个新块，而且只在正好跨过块边界时才需要。所以 `can_allocate` 返回的是 int（`-1` 表示装不下，否则是 prefix cache 命中的块数），`can_append` 返回的是 bool。

失败时的应对也不同。**prefill 装不下就 `break`**，序列继续在 `waiting` 里等下一轮——反正它还没开始算，不亏。**decode 装不下却不能等**，序列已经在跑了，所以调度器会从 `running` 队尾往前抢占别的序列，把它们的块全释放掉腾给当前这条。

**BlockManager 的四个字段。** 它一共只维护四份状态:

```
BlockManager
│
├── blocks: list[Block]           定长,创建后不增不删,下标就是 block_id
│      Block(i) {
│        ref_count   有几条序列在用这块 (0 = 空闲)
│        hash        这块对应的"前缀哈希",-1 表示未哈希
│        token_ids   这块装的 token,只用于防哈希碰撞
│      }
│
├── free_block_ids: deque[int]    空闲块号。popleft 取 / append 还 → FIFO
│
├── used_block_ids: set[int]      在用块号。只为 O(1) 回答"这块有人在用吗"
│
└── hash_to_block_id: dict        前缀哈希 → block_id,prefix cache 的索引
```

`free_block_ids` 和 `used_block_ids` 严格互补，一个块在 `used_block_ids` 里当且仅当它的 `ref_count > 0`。

后两个字段撑起了 prefix cache：`ref_count` 是块共享的基础——两条序列前缀相同就能指向同一个物理块，各自把 `ref_count` 加一，KV 在显存里只存一份；`hash` 则是找到这种共享机会的索引，它存的不是本块内容的哈希，而是「从头到本块」整段前缀的哈希（具体怎么串在 1.2.5 展开）。

这四份状态和调度的关系可以一句话概括：调度器看的是队列，BlockManager 看的是 `free_block_ids` 还剩几个。整篇文章后面的所有行为——为什么某轮只排得下三条、为什么一条刚跑起来的序列会被牺牲、为什么重算没想象中贵——归根到底都是这个数字在变。


### 1.2 演示例子

下面我们会通过演示一个例子说明进一步解释调度机制。

#### 1.2.1 演示示例配置

真实默认值是 `max_num_batched_tokens=16384`、`max_num_seqs=512`（见 `Config`），数字太大不好画。下面把它缩小成一组玩具参数，其余逻辑与代码完全一致：

```
max_num_batched_tokens = 1024      # 一个 batch 最多算多少 token
max_num_seqs           = 4         # 一个 batch 最多几条序列
kvcache_block_size     = 256       # 一个 KV block 装多少 token
num_kvcache_blocks     = 11        # 总共有多少物理块
```

六个请求，prompt 长度和它们各自需要的块数（`ceil(len / 256)`，见 `Sequence.num_blocks`）：

```
请求    prompt 长度    需要的块数
R1        300            2
R2        400            2
R3        255            1
R4       1200            5
R5        200            1      ← 第 5 轮才到达
```


#### 1.2.2 调度过程轮次示例

队列状态是**每一轮开始时**的快照。`空闲块` 一列是这一轮结束后 `free_block_ids` 的剩余量。

```
轮次      waiting 队列          running 队列            本轮 batch
────────────────────────────────────────────────────────────────────────────────────
Round 1   [R1 R2 R3 R4]         [ ]                     prefill R1+R2+R3 = 955 tok
                                                        └ 停:预算只剩 69 < R4 的 1200,
                                                          且本轮已有序列 → 不给非首条切块
                                                          空闲块 11 → 6

Round 2   [R4]                  [R1 R2 R3]              prefill R4[0:1024]  ← chunked
                                                        └ 停:预算正好用尽 (remaining==0)
                                                          R4 没算完,留在 waiting 队头
                                                          空闲块 6 → 1  (一次预留全部 5 块)

Round 3   [R4]                  [R1 R2 R3]              prefill R4[1024:1200] = 176 tok
                                                        └ 停:waiting 空
                                                          R4 算完 → 转入 running
                                                          空闲块 1

Round 4   [ ]                   [R1 R2 R3 R4]           decode ×4 (各生成 1 token)
                                                        └ 停:running 取空 / 已达 max_num_seqs
                                                          空闲块 1  (无人跨块边界)

Round 5   [R5]                  [R1 R2 R3 R4]           prefill R5 = 200 tok
                                                        └ 新请求到达,插队做 prefill,
                                                          本轮 decode 被整体跳过
                                                          空闲块 1 → 0

Round 6   [ ]                   [R1 R2 R3 R4 R5]        decode R1 R2 R3 R4
                                                        └ R3 跨块边界但空闲块=0
                                                          → 抢占队尾的 R5 (详见图 3)
                                                          停:已达 max_num_seqs
                                                          空闲块 0 → 1 → 0

Round 7   [R5]                  [R1 R2 R3 R4]           decode ×4
                                                        └ R5 想重新 prefill 但块不够
                                                          (can_allocate 返回 -1) → 继续等
                                                          prefill 一条没抓到 → 落到 decode
                                                          空闲块 0
```

几个容易看漏的点：

**Round 1 —— 为什么 R4 不切块？** prefill 循环里的条件是 `if remaining < num_tokens and scheduled_seqs: break`。后半个条件意味着 **chunked prefill 只对本轮第一条序列开放**。R4 前面已经排了三条，所以宁可空着 69 的预算也不切。

**Round 2 —— 块是按完整 prompt 一次性预留的。** R4 这轮只算了 1024 个 token，但 `allocate()` 直接按 `seq.num_blocks`（完整的 1200 token → 5 块）把 `block_table` 填满。所以空闲块一次就从 6 掉到 1，不是随 chunk 递增。

**Round 2→3 —— 没算完的序列留在 waiting 队头。** 只有 `num_cached_tokens + num_scheduled_tokens == num_tokens` 时才 `popleft()` 转入 running。同时 `postprocess()` 对没算完的序列会跳过 `append_token`，所以 chunked prefill 期间 `num_tokens` 不变，`block_table` 也就一直是完整的。

**Round 5 —— prefill 插队。** 这正是 prefill 优先带来的代价：一个新请求到达，所有正在 decode 的序列这一轮全部停摆。

**Round 7 —— prefill 失败不会卡住 decode。** `can_allocate` 返回 `-1` 时只是 `break`，`scheduled_seqs` 仍为空，于是自然落到 decode 分支。

---

#### 1.2.3 对 Round 6 抢占的详细说明

decode 循环碰到块不够时不是跳过这条序列，而是**牺牲队尾的序列给它腾块**。

```
进入 Round 6:  running = [R1 R2 R3 R4 R5]   空闲块 = 0

  popleft R1  len=302, 302%256=46 ≠ 1  → 不需要新块 → 调度 ✓
  popleft R2  len=402, 402%256=146 ≠ 1 → 不需要新块 → 调度 ✓
  popleft R3  len=257, 257%256=1  → 需要 1 个新块,但空闲块=0
              │
              │  while not can_append(R3):
              │      running 非空 → preempt(running.pop())
              │                              └─ 队尾是 R5,LIFO 牺牲最新的
              ▼
        preempt(R5):  status → WAITING
                      is_prefill → True
                      deallocate() 释放它的 1 个块   空闲块 0 → 1
                      塞回 waiting 队头
              │
              ▼
        再查 can_append(R3): 1 ≥ 1 ✓ → 调度 R3,may_append 吃掉那个块  空闲块 1 → 0
  popleft R4  len=1203, 1203%256=179 ≠ 1 → 调度 ✓   此时已有 4 条 = max_num_seqs,退出

退出 Round 6:  running = [R1 R2 R3 R4]   waiting = [R5]
```

R5 刚在上一轮 prefill 完就被牺牲，KV 全丢，下轮得重算——这就是 recompute 式抢占。vLLM V1 的调度器里也只有这一条路径，`_preempt_request()` 同样是释放全部块 + `num_computed_tokens = 0`；swap 式抢占是 V0 才有的东西。

不过重算通常没那么贵：`deallocate` 只把块还回 `free_block_ids`，**不清 `block.hash`**，`hash_to_block_id` 里的条目也留着（只在块被 `_allocate_block()` 真正复用时才删）。只要这些块还没被别人覆盖，下次 `can_allocate` 走 prefix cache 匹配就能把前缀整段捡回来。

---

#### 1.2.4 跟随上述例子走一遍块管理

下面每一格是一个物理块,`Rx.n` 表示"序列 x 的第 n 个逻辑块"。轮次与图 2 一一对应。

```
初始
块号     0     1     2     3     4     5     6     7     8     9    10
      ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
      │  ·  │  ·  │  ·  │  ·  │  ·  │  ·  │  ·  │  ·  │  ·  │  ·  │  ·  │
      └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
      free = [0,1,2,3,4,5,6,7,8,9,10]        hash_to_block_id = {}


Round 1 后    R1 拿 2 块、R2 拿 2 块、R3 拿 1 块 (各自 popleft 连号取走)
块号     0     1     2     3     4     5     6     7     8     9    10
      ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
      │R1.0 │R1.1 │R2.0 │R2.1 │R3.0 │  ·  │  ·  │  ·  │  ·  │  ·  │  ·  │
      └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
      free = [5,6,7,8,9,10]
      hash = { h(R1.0)→0, h(R2.0)→2 }
             R1.1/R2.1 是不满的尾块,R3.0 只有 255 token 也不满 → 都不哈希


Round 2 后    R4 一次性预留全部 5 块 (虽然这轮只算了 1024/1200 个 token)
块号     0     1     2     3     4     5     6     7     8     9    10
      ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
      │R1.0 │R1.1 │R2.0 │R2.1 │R3.0 │R4.0 │R4.1 │R4.2 │R4.3 │R4.4 │  ·  │
      └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
      free = [10]
      hash += { h(R4.0)→5, h(R4.1)→6, h(R4.2)→7, h(R4.3)→8 }
             这轮刚好算满前 4 块 → 立刻哈希;R4.4 还空着 → 不哈希


Round 3 后    块无变化。R4.4 填到 176/256,仍不满 → 仍不进哈希表
      free = [10]


Round 4 后    块无变化 (四条序列都没跨块边界)
      free = [10]
      hash += { h(R3.0)→4 }
             ← R3 这轮 decode 把 255 填成 256,尾块变满,补哈希


Round 5 后    R5 拿走最后 1 块
块号     0     1     2     3     4     5     6     7     8     9    10
      ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
      │R1.0 │R1.1 │R2.0 │R2.1 │R3.0 │R4.0 │R4.1 │R4.2 │R4.3 │R4.4 │R5.0 │
      └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
      free = []                    ← 满了
      hash 不变 (R5.0 只有 200 token,不满)


Round 6 中    R3 跨块边界,抢占 R5 腾出块 10,R3 立刻把它拿走
      ① preempt(R5) → deallocate:  块 10 ref_count 1→0,  free = [10]
      ② can_append(R3) 复查通过 → may_append → _allocate_block popleft 10
                                                 free = []
块号     0     1     2     3     4     5     6     7     8     9    10
      ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
      │R1.0 │R1.1 │R2.0 │R2.1 │R3.0 │R4.0 │R4.1 │R4.2 │R4.3 │R4.4 │R3.1 │
      └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
                                                                    ▲
                                              同一个块,一轮之内 R5 → R3 易主
      free = []
      hash 不变。块 10 的 hash 一直是 -1(R5.0 从没满过),
             所以 _allocate_block 里那句"删除过期哈希条目"没触发


Round 7 后    块无变化。R5 想重新 prefill 要 1 块,free 为空
             → can_allocate 返回 -1 → 继续在 waiting 等
      free = []
```

从这条轨迹能读出 BlockManager 的三条设计:

**只有填满的块才进哈希表。** `hash_blocks()` 用整除算出本轮填满了哪几块,`start == end` 就直接 return。不满的尾块内容还会变,拿它当 cache key 是错的。所以 R3.0 要等到 Round 4 被 decode 填满才补上哈希,而 R4.4 从头到尾都没进过哈希表。

**释放块不清哈希。** `deallocate()` 只把块号 append 回 `free_block_ids`,`block.hash` 和 `hash_to_block_id` 里的条目都留着。过期条目只在这块被 `_allocate_block()` 真正复用时才删。这就是被抢占的序列还能靠 prefix cache 把前缀捡回来的原因。

**空闲队列是 FIFO,这是有意的。** `popleft` 取、`append` 还,意味着刚释放的块排在队尾、最晚被复用——给了它的哈希最长的存活窗口,近似一个 LRU。Round 6 是个极端反例:`free` 恰好空了,刚还回去的块立刻被 `popleft` 拿走,缓存价值归零。

---

#### 1.2.5 prefix cache 命中时的块分配路径

先把 1.1 里欠下的那笔补上：块的 `hash` 为什么是「整段前缀」的哈希。`compute_hash()` 把上一块的哈希值当种子链进来:

```
  逻辑块 0              逻辑块 1                逻辑块 2
  [256 个 token]        [256 个 token]          [256 个 token]
        │                     │                       │
        ▼                     ▼                       ▼
   xxh64(tokens)      xxh64(h0 ‖ tokens)      xxh64(h1 ‖ tokens)
        │                     │                       │
        ▼                     ▼                       ▼
       h0 ────────────────►  h1 ──────────────────►  h2

   prefix.to_bytes(8, "little") 把上一块的 64 位哈希编成 8 字节喂进去
```

必须串前缀的理由是 KV 依赖全部前文。如果只哈希本块内容,两条前缀不同、但恰好第 3 块内容相同的序列就会错误地共享 KV。串上之后,`h2` 标识的是 `token[0:768]` 这整段,而不只是第三块那 256 个 token。

上面那条轨迹里六个请求互不相同,`can_allocate` 一次都没命中缓存。补一个命中的例子:假设 Round 1 之后来了个 R7,600 token(3 块),**前 256 个 token 和 R1 完全一样**。

```
can_allocate(R7)                                num_new_blocks = 3

  i=0  h = xxh64(R7.block(0))          = h(R1.0)
       hash_to_block_id[h]             → 块 0
       blocks[0].token_ids == R7.block(0) ?  ✓ 逐 token 复核,防哈希碰撞
       → num_cached_blocks = 1
       → 块 0 在 used_block_ids 里      → num_new_blocks -= 1  → 2

  i=1  h = xxh64(h0 ‖ R7.block(1))     ← 链式,和 R1.1 的哈希不同
       未命中 → break

  len(free) >= 2 ?  → 返回 num_cached_blocks = 1


allocate(R7, num_cached_blocks=1)

  逻辑块 0 → 复用块 0:  ref_count 1 → 2
  逻辑块 1 → _allocate_block()  新块
  逻辑块 2 → _allocate_block()  新块
  num_cached_tokens = 1 * 256 = 256


  块 0  ┌──────────────┐
        │  ref_count=2 │◄──── R1.block_table = [0, 1]
        │              │◄──── R7.block_table = [0, x, y]
        └──────────────┘      两条序列指向同一份 KV,物理上只存一份
```

回到调度器:prefill 循环据此算出 `num_tokens = 600 - 1*256 = 344`,R7 这轮只需要 prefill 344 个 token,前 256 个直接白拿。

`allocate()` 对命中的块分两种情况处理,区别在于块当前在哪:

```
命中的块在哪里?
│
├─ 在 used_block_ids —— 别人正在用
│    ref_count += 1
│    不消耗新块 → 所以 can_allocate 里 num_new_blocks -= 1
│
└─ 在 free_block_ids —— 已释放但还没被覆盖
     ref_count = 1
     free_block_ids.remove(block_id)     ← deque 的 O(n) 删除
     used_block_ids.add(block_id)
     这是被抢占的序列"捡回自己旧块"的路径。
     注意它仍占着一个 free 名额,所以 can_allocate 不减 num_new_blocks——
     两处口径一致,不会算错。

  两种情况都不能调 block.reset(),因为 reset 会清掉 hash 和 token_ids,
  而那正是复用的依据。reset() 只在 _allocate_block 里用,
  它里面的 ref_count = 1 就是"这块刚被分配出去"的意思。
```

---

## 2 vLLM 的 调度机制

看完了 Nano-vLLM 的调度，我们来看一下生产级应用 [vLLM](https://github.com/vllm-project/vllm) 的调度机制。

### 2.1 vLLM `schedule()` 伪代码

现在看真的 vLLM 的调度（`vllm/v1/core/sched/scheduler.py`）。它的 `schedule()` 有七百多行，但把多模态 encoder 预算、LoRA 上限、投机解码、KV connector（P/D 分离与 offload）、Mamba 对齐、DP 负载均衡这些正交的维度全剥掉之后，骨架其实不长：

```
def schedule():
    token_budget = max_num_batched_tokens
    scheduled    = {}            # req_id → 本轮要算的 token 数
    preempted    = []

    # ── 第一段：先服务 running 里的请求 ───────────────────────────
    i = 0
    while i < len(running) and token_budget > 0:
        req = running[i]

        # 这条请求还差多少 token 没算。注意这里不区分 prompt 和已生成:
        # 纯 decode 时它 = 1,prefill 没做完时它 = 剩余的整段
        num_new = req.num_tokens_with_spec - req.num_computed_tokens
        num_new = min(num_new, long_prefill_token_threshold)   # 单条限流
        num_new = min(num_new, token_budget)                   # 全局预算
        num_new = min(num_new, max_model_len - req.num_computed_tokens - 1)

        if num_new == 0:
            i += 1;  continue        # ← continue 而不是 break,
                                     #   故意放弃严格 FCFS,让后面的请求能插上来

        # 试着为这 num_new 个 token 分配 KV slot,不够就抢占别人再试
        while (blocks := kv_cache_manager.allocate_slots(req, num_new)) is None:
            if policy == PRIORITY:
                victim = max(running, key=lambda r: (r.priority, r.arrival_time))
                running.remove(victim)
            else:                          # FCFS
                victim = running.pop()     # 队尾,LIFO——和 nano-vllm 一样
            preempt(victim)
            preempted.append(victim)
            if victim is req:              # 没人可抢了,把自己也牺牲掉
                break
        if blocks is None:
            break

        scheduled[req.id] = num_new
        token_budget -= num_new
        i += 1

    # ── 第二段：预算还有剩,就从 waiting 里捞新请求 ─────────────────
    if not preempted:                      # ← 这一轮抢占过,就不再收新的
        while waiting and token_budget > 0:
            if len(running) >= max_num_seqs:
                break
            req = waiting.peek()

            if req.num_computed_tokens == 0:              # 第一次被调度
                cached_blocks, num_computed = \
                    kv_cache_manager.get_computed_blocks(req)   # prefix cache 查询
            else:                                         # 被抢占后恢复的
                num_computed = req.num_computed_tokens

            num_new = req.num_tokens - num_computed
            if not enable_chunked_prefill and num_new > token_budget:
                break
            num_new = min(num_new, long_prefill_token_threshold)
            num_new = min(num_new, token_budget)

            blocks = kv_cache_manager.allocate_slots(req, num_new, cached_blocks)
            if blocks is None:
                break                      # 块不够 → 直接停,这里不抢占

            waiting.pop()
            running.append(req)
            req.status              = RUNNING
            req.num_computed_tokens = num_computed
            scheduled[req.id]       = num_new
            token_budget           -= num_new

    return SchedulerOutput(scheduled, ...)


def preempt(req):
    kv_cache_manager.free(req)       # 释放全部块
    req.status              = PREEMPTED
    req.num_computed_tokens = 0      # 全部重算
    req.num_preemptions    += 1
    waiting.prepend(req)             # 回到 waiting 队头
```

源码里 `schedule()` 开头那段注释把设计意图说得很直白：

> There's no "decoding phase" nor "prefill phase" in the scheduler. Each request just has the `num_computed_tokens` and `num_tokens_with_spec`. At each step, the scheduler tries to assign tokens to the requests so that each request's `num_computed_tokens` can catch up its `num_tokens_with_spec`.

**调度器里根本没有 prefill 和 decode 的概念。** 每条请求只有两个数——"已经算了多少 token"和"总共该算多少 token"——调度器每一步就是发放 token 配额让前者去追后者。prompt 阶段差得多，一次追一大段；生成阶段每次只差 1，就追 1。chunked prefill、prefix caching、投机解码全都自然落在这一个抽象里，不需要为谁开分支。

这正是和 nano-vllm 最根本的分歧：nano-vllm 把 prefill / decode 写成了两个物理上分开的循环，中间那句提前 return 是硬隔离。

---

### 2.2 一个 batch 里装了什么

上面那点差异，落到实际发出去的 batch 上是这样：

```
nano-vllm —— 二选一,绝不混
                                                   Round 1
  ┌────────────────────────────────────────────┐
  │  R1[0:300]   R2[0:400]   R3[0:255]         │   955 个 token,全是 prompt
  └────────────────────────────────────────────┘
     prefill                                       (schedule 返回 is_prefill=True)

                                                   Round 4
  ┌────────────────────────────────────────────┐
  │  R1×1   R2×1   R3×1   R4×1                 │   4 个 token,全是新生成
  └────────────────────────────────────────────┘
     decode                                        (schedule 返回 is_prefill=False)

  新请求一到,decode 整轮停摆 → 所有在跑的请求都被卡一拍(图 2 的 Round 5)


vLLM —— 一个 batch 混着装
  ┌──────────────────────────────────────────────────────────────────┐
  │  R1×1    R2×1    R4[1024:1200]        R5[0:800]                  │
  │  decode  decode   prefill 尾块          prefill 首块(被预算切断)    │
  └──────────────────────────────────────────────────────────────────┘
    └──── 第一段:从 running 拿 ────┘  └── 第二段:预算有剩,再从 waiting 捞 ──┘

  新请求只吃掉 decode 之后剩下的预算 → 正在跑的请求不会被饿死
```

vLLM 的顺序是**先 running 后 waiting**：先保证已经在跑的请求继续推进（这里面既有纯 decode，也有上一轮没切完的 prefill 尾巴），预算还有剩才去 waiting 里捞新的。nano-vllm 反过来，waiting 优先，而且只要捞到一条就整轮不做 decode。

一句话概括：**nano-vllm 让新请求插队，vLLM 让在跑的请求优先。** 前者的 TTFT（首 token 延迟）更好看，代价是 ITL（token 间延迟）会被新请求打出毛刺；后者反过来，用一点 TTFT 换稳定的 ITL。

---

## Nano-vLLM 和 vLLM 差异对照

| | nano-vllm | vLLM V1 |
|---|---|---|
| **核心抽象** | 显式的 prefill / decode 两个阶段 | 只有 `num_computed_tokens` 追 `num_tokens_with_spec`，无阶段概念 |
| **两个循环的顺序** | 先 waiting（prefill），后 running（decode） | 先 running，后 waiting |
| **能否混批** | 不能。抓到一条 prefill 就 `return`，decode 整轮跳过 | 能。同一个 batch 里 decode 和 prefill chunk 并存 |
| **token 预算** | 只约束 prefill；decode 完全不看 | 一个 `token_budget` 贯穿两段，decode 也扣（每条扣 1 或 1+spec） |
| **谁能被切块** | 只有 batch 里第一条（`remaining < num_tokens and scheduled_seqs` 才 break） | 任何一条都能切；只有显式关掉 `enable_chunked_prefill` 才不切 |
| **单条限流** | 无。一条长 prompt 能吃满整个预算 | `long_prefill_token_threshold` 给单条封顶 |
| **块的分配粒度** | 首次调度就按完整 prompt 预留全部 `num_blocks` | `allocate_slots()` 只分配本 chunk 需要的量（除非开 `scheduler_reserve_full_isl`） |
| **"装得下吗"的接口** | `can_allocate` 返回 `-1`/缓存块数，`can_append` 返回 bool，查完再单独 `allocate` | 一个 `allocate_slots()`，成功返回块、失败返回 `None`，查与分配合一 |
| **某条排不上时** | 一律 `break`，不区分原因 | 分两类：「暂时轮不到这条」跳过继续（running 循环 `continue`，waiting 循环挪进 `skipped_waiting`），「资源真耗尽」才 `break`。源码注释明说前者是有意放弃严格 FCFS |
| **抢占挑谁** | 只有 `running.pop()`，队尾 LIFO | FCFS 下同样是队尾；PRIORITY 策略下挑 `(priority, arrival_time)` 最大的 |
| **抢占方式** | recompute（释放全部块，`is_prefill=True`，回 waiting 队头） | 一样是 recompute（`num_computed_tokens=0`，回 waiting 队头） |
| **抢占后本轮** | 无特殊处理 | `if not preempted_reqs` —— 这轮抢占过就不再收新请求，避免刚腾出的块立刻被新人吃掉 |
| **max_num_seqs 检查位置** | 两个循环都按 `len(scheduled_seqs)` 卡 | 只在 waiting 循环按 `len(running)` 卡 |
| **等待队列** | 一个 `deque` | `waiting` + `skipped_waiting` 两个队列，可插拔 FCFS / PRIORITY |

除此之外 vLLM 的 `schedule()` 里还塞着一堆 nano-vllm 完全没有的维度，它们各自都会再削一刀 `num_new_tokens` 或者让某条请求这轮直接出局：多模态 encoder 的独立算力预算与 encoder cache、LoRA 的 `max_loras` 上限、投机解码的 lookahead slot 与 draft token、KV connector（P/D 分离、KV offload，请求会进 `WAITING_FOR_REMOTE_KVS` 状态挂起）、Mamba 混合模型的块对齐切分、数据并行下的 prefill 节流。骨架是上面那六十行，剩下的七百行基本都是这些维度的交织。

两处值得单独拎出来的细节：

**抢占后本轮不再收新请求。** vLLM 第二段外面套着 `if not preempted_reqs`。理由很实际：既然刚刚已经因为块不够牺牲了别人，那这一轮就别再放新请求进来抢那些刚腾出来的块，否则会来回抖动。nano-vllm 因为 prefill 和 decode 天然不在同一轮，碰不到这个问题——但代价是图 2 Round 6 那种"R5 刚 prefill 完下一轮就被牺牲"的浪费它防不住。

**块的分配粒度。** nano-vllm 首次调度时 `allocate()` 就按完整 `seq.num_blocks` 把块全预留了（图 2 的 Round 2，空闲块一次从 6 掉到 1），好处是后续 chunk 不会中途缺块，坏处是一条长 prompt 一进来就锁住大量显存。vLLM 的 `allocate_slots()` 只按本 chunk 的 token 数分配，长 prompt 的显存是随 chunk 逐步吃进去的——更省，但也意味着一条正在 prefill 的请求可能算到一半才发现块不够，于是被抢占、前功尽弃。要 nano-vllm 那种行为得显式开 `scheduler_reserve_full_isl`。
