---
title: 'nano-vllm 的调度机制：一次 schedule() 到底做了什么'
description: '用一组轮次快照，逐轮拆解 nano-vllm 调度器的 prefill 优先、chunked prefill、token 预算与抢占机制。'
pubDate: '2026-07-26'
tags: ['llm', 'inference', 'vllm', 'paged-attention']
draft: true
---

## 演示用的玩具配置

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

---

## 图 1：`schedule()` 的主干

一次调度**要么全是 prefill，要么全是 decode**，两者不混批。关键在 `Scheduler.schedule()` 里 prefill 循环结束后那句提前 return——只要抓到了哪怕一条序列，就立刻返回，decode 分支根本不执行。

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

---

## 图 2：轮次快照表（核心）

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

## 图 3：Round 6 的抢占

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

R5 刚在上一轮 prefill 完就被牺牲，KV 全丢，下轮得重算——这就是 recompute 式抢占（vLLM 还有 swap 式，nano-vllm 没实现）。

不过重算通常没那么贵：`deallocate` 只把块还回 `free_block_ids`，**不清 `block.hash`**，`hash_to_block_id` 里的条目也留着（只在块被 `_allocate_block()` 真正复用时才删）。只要这些块还没被别人覆盖，下次 `can_allocate` 走 prefix cache 匹配就能把前缀整段捡回来。

---


## 图 4：BlockManager 的四个数据结构

调度器只管"哪条序列这轮跑",真正管显存的是 `BlockManager`(`nanovllm/engine/block_manager.py`)。它一共只有四个字段:

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

两条贯穿始终的不变量:

```
  block_id ∈ used_block_ids   ⟺   blocks[block_id].ref_count > 0
  free_block_ids ⊎ used_block_ids  =  全部 block_id      (二者互补,不重不漏)
```

`hash` 这一栏是 prefix cache 的关键,它**不是本块内容的哈希,而是"从头到本块"整段前缀的哈希**——`compute_hash()` 把上一块的哈希值当种子链进来:

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

为什么必须串前缀:KV 依赖全部前文。如果只哈希本块内容,两条前缀不同、但恰好第 3 块内容相同的序列就会错误地共享 KV。串上之后,`h2` 标识的是 `token[0:768]` 这整段,而不只是第三块那 256 个 token。

---

## 图 5：跟着图 2 走一遍块管理

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

## 图 6：prefix cache 命中时的块分配路径

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

## 图 7：block_table 到底存了什么

调度器搬来搬去的 `seq.block_table` 里既不是 token 也不是 KV 数值，而是**物理块号**。整个 paged attention 是三层结构：

```
  seq.block_table           BlockManager.blocks[]         model_runner.kv_cache
  (页表, list[int])          (记账簿, CPU 侧元数据)          (物理页, 显存里的大张量)

  逻辑块 0 ─► 7 ──┐         Block(7):                     shape =
  逻辑块 1 ─► 2   │           ref_count = 2   ← 被两条      [2, num_layers,
  逻辑块 2 ─► 9   │           hash      = ...    序列共享     num_blocks,  ◄── block_id
  逻辑块 3 ─► 0   │           token_ids = [...] ← 只用于      block_size,      索引这一维
  逻辑块 4 ─► 5   │                              防哈希碰撞   num_kv_heads,
                  │                                          head_dim]
                  └────────────────────────────────────►  kv_cache[:, :, 7] 这一页
```

- **`kv_cache`** 在 `ModelRunner.allocate_kv_cache()` 里建，是一整块显存，`block_id` 索引第 3 维；同一个方法把每层的切片挂到对应 attention 模块的 `k_cache` / `v_cache` 上。
- **`block_table`** 是逻辑块号 → 物理块号的映射，让一条序列的 KV 可以散落在显存各处而不必连续。写入时 `ModelRunner` 的 `prepare_prefill()` / `prepare_decode()` 把它换算成 `slot_mapping`（每个 token 的扁平槽位），由 `store_kvcache` 这个 triton kernel 落盘；读取时直接把 `block_tables` 张量交给 FlashAttention 的 `block_table` 参数，在 kernel 内部按页取。
- **`Block`** 对象只是 CPU 侧元数据，不含任何 KV 数值。里面的 `token_ids` 唯一用途是 `can_allocate()` 里 `self.blocks[block_id].token_ids != token_ids` 这句做逐 token 比对——光哈希相等不敢直接复用别人的 KV，还要确认 token 真的一样。
