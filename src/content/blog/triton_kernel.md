---
title: 'Triton 学习笔记'
description: '本文介绍了如何编写 Triton 算子，列出了 Triton 的关键语法，然后介绍了 Triton 算子的优化流程，包括如何判断算子的正确性、如何准确测量性能以及如何识别瓶颈也优化。'
pubDate: '2026-08-30'
tags: ['triton', 'gpu', 'kernel', 'optimization', 'cuda', 'performance']
---

Triton 是一门用于编写 GPU kernel 的编程语言和编译器，由 OpenAI 主导开发。传统上写高性能 GPU kernel 要用 CUDA C++，手动管理线程、block、shared memory、内存合并访问、寄存器分配等，开发难度较高。Triton 的目标是用类似 Python 的语法，来写出接近 CUDA 性能的 kernel。

在 CUDA 中，开发者要手动管到 thread/warp，而 Triton 让开发者以 block 为单位思考，编译器自动来处理 thread 级别的调度、内存合并、shared memory 管理等部分。

## 1 如何编写 Triton 算子

学习编写 Triton 算子，可以参考其[官方教程](https://triton-lang.org/main/getting-started/tutorials/index.html) 。我把一些关键点，在下面进行一下归纳。

### 1.1 triton 和 triton.language: 

triton 和 triton.language 是 Triton 提供的两个 python package。

```python
import triton
import triton.language as tl

@triton.jit                      
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)                       
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)     
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output
```

triton: 框架层，提供外壳和工具，用在 host 端，类似 CUDA 代码中的 host 代码 + 工具链。
- @triton.jit：把 Python 函数编译成 GPU kernel 的装饰器
- triton.autotune、triton.heuristics：自动调参
- kernel 的启动语法（kernel[grid](...)）
- 底层的编译器、运行时、和 GPU 打交道的部分

triton.language: 语言层，提供在 kernel 内部实际使用的积木，只能在 @triton.jit 修饰的 kernel 函数体内部使用，
类似 CUDA 中的 device 端代码。
- tl.program_id() / tl.arange()：定位和生成索引
- tl.load() / tl.store()：读写显存（带 mask 做边界保护）
- tl.dot()：矩阵乘
- tl.sum() / tl.max()：规约操作
- tl.constexpr：编译期常量类型
- 各种数据类型、数学函数等

### 1.2 如何加载数据

内存偏移 = 第0维索隐 * 第0维stride + 第1维索隐 * 第1维stride + 第2维索隐 * 第2维stride + ...

stride 表示：在某个维度上索引加 1，需要在底层一维内存里跳过多少个元素。矩阵转置只是把两个维度的 stride 交换了一下。shape 决定"逻辑形状"，stride 决定"内存怎么排布"，两者配合才完整描述一个张量。连续的 shape = (3, 4) 矩阵和转置得到的 shape = (3, 4) 的矩阵，其内部排布是不同的。

x[m, n] 的内存偏移就是 m * stride_m + n * stride_n

```python
@triton.jit
def kernel(x_ptr, stride_m, stride_n, ...):
    # 访问 x[m, n] 就是：
    offset = m * stride_m + n * stride_n
    val = tl.load(x_ptr + offset)
```

下面是更多的例子，注意：block 的每一维都得是 2 的幂，在取/写数据时，注意用 mask 做边界保护。
```python
# x_ptr 指向一维 tensor，长度为 n
pid = tl.program_id(0)
offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
# BLOCK_SIZE 必须是 2 的幂，如果 x 的长度不是 2 的幂，注意使用 mask 做边界保护
mask = offsets < n 
x = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))

# y_ptr 指向二维 tensor，形状如 [T, E]，在 T 这个维度上切分 BLOCK
pid_t = tl.program_id(0)
pid_e = tl.program_id(1)
offs_t = pid_t * BLOCK_T + tl.arange(BLOCK_T)
offs_e = tl.arange(0, E)
mask_t = offs_t < T
y = tl.load(y_ptr + offs_t[:, None] * E + offs_e[None, :], 
        mask=mask_t[:, None], other=0.0) # [BLOCK_T, E]
# 如果想直接加载 y.T (y 的转置)
y_t = tl.load(y_ptr + offs_t[None, :] * E + offs_e[:, None],
        mask=mask_t[None, :], other=0.0) # [E, BLOCK_T]
```

### 1.3 规约函数

假设有一个 shape 为 (M, N) 的 2D block，x[i, j]：

tl.sum(x, axis=0) 沿第 0 维求和：

result[j] = x[0, j] + x[1, j] + ... + x[M-1, j]

结果 shape 变成 (N,)， 相当于就在第 0 维上求和，M 维被压掉了。

```
x: (M, N)
tl.sum(x, axis=0) → (N,)    # 第0维被规约了，没了
tl.sum(x, axis=1) → (M,)    # 第1维被规约了，没了
```

### 1.4 tl.where

```python
# x 是 kernel 内部的一个向量，想把第 idx 个位置换成 value
offsets = tl.arange(0, BLOCK_SIZE)
x = tl.where(offsets == idx, value, x)   # idx 处取 value，其余保持原 x
```

### 1.5 内部 API 判别

```python
inspect.getdoc(fn)     # None → 大概率是内部工具，版本升级可能变
inspect.signature(fn)  # 查参数，比翻文档快
```

常见的内部 API：`kernel.warmup`、`_init_handles`、`n_regs`、`metadata.shared`、`testing.Mark`、`testing.set_gpu_clock`（且带 A100 硬编码）。


## 2 如何优化 Triton 算子

Triton 算子的优化遵循一下工作顺序，重新测后，可以继续判断瓶颈类型、进一步优化，形成循环。
```
写 kernel → 对拍正确性 → 测性能 → 判断瓶颈类型 → 优化 → 重新对拍 → 重新测
```

### 2.1 如何对拍正确性

```python
torch.testing.assert_close(got, ref, rtol=1e-2, atol=1e-2)
```

比 `assert torch.allclose(...)` 好在**失败时告诉你最大偏差和下标**。

**`atol` (Absolute Tolerance，绝对容差)**：允许的**绝对误差**上限。两个元素之间的差值不能超过这个固定值
**`rtol` (Relative Tolerance，相对容差)**：允许的**相对误差**比例。它通常与参考值（`ref`）的大小相乘，数值越大，允许的绝对偏差就越大。

PyTorch 判断 `got`（实际值）和 `ref`（参考值）是否“close”的底层数学公式是：

$$ |got - ref| \le atol + rtol \times |ref| $$

因此，got 和 ref 的参数顺序不要搞错，否则会影响判断的准确性

容差参考：

| dtype | rtol | atol |
|---|---|---|
| fp32 | 1e-5 | 1e-6 |
| fp16 / bf16 | 1e-2 | 1e-2 |

**规约顺序不同必然带来数值差异**，不苛求比特级别的完全一致。判据是否是 bug：
- 误差随规模缓慢增长 → 正常的浮点累积
- 某几个位置差得离谱、或出现 nan/inf → 真 bug

**累加器永远用 fp32**，fp16 累加在 K 稍大时明显掉精度。

### 2.2 如何测量性能

```python
ms = triton.testing.do_bench(lambda: kernel[grid](x, y, n), return_mode='median')
```

`do_bench` 帮你处理了：CUDA event 精确计时、warmup、多次取统计量、**每轮清空 L2**。

**必须包 `lambda`** —— do_bench 需要的是一个能重复跑去测量的函数，lambda: kernel[grid](x, y, n) 定义了一个没有参数的匿名函数，把"该怎么调用 kernel"这件事打包成一个函数对象传进去；如果直接用 kernel[grid](x, y, n)，则会把 kernel 的运行结果传入 do_bench。

**需要每轮清空 L2** —— 现代 GPU 的 L2 很大，数据第一次读进来之后就待在 L2 里了，后续多次运行反复读写全在 L2 上跑，没有碰到显存，测到的不是实际情况。


### 2.3 如何判断瓶颈并进行优化

#### 2.3.1 获得硬件特性

```python
from triton.runtime import driver
props = driver.active.utils.get_device_properties(0)
print(props)
# {'max_shared_mem': 101376, 'max_num_regs': 65536, 'multiprocessor_count': 128, 'warpSize': 32, 
# 'sm_clock_rate': 2520000, 'mem_clock_rate': 10501000, 'mem_bus_width': 384}
```

通过阅读架构白皮书以及在机器上运行代码的方式，获得机器的硬件特性，这对我们之后的估算与优化有参考意义。


#### 2.3.2 首先判断是 memory-bound 还是 compute-bound

kernel 跑的慢，有两类瓶颈：

- memory-bound（访存受限）：计算很少，但要搬很多数据。GPU 的计算单元大部分时间在等数据从显存运过来，算力闲着。瓶颈是显存带宽
- compute-bound（计算受限）：数据不多，但要做大量运算。数据早就到位了，GPU 的计算单元在满负荷算。瓶颈是算力（FLOPS）。

如何判断是哪一类瓶颈，可以使用 Roofline 模型，计算算数强度与硬件的 ridge point（脊点） 进行比较：

**算术强度** = FLOP 数（做多少次浮点运算） / 字节数（搬多少数据）

它衡量的是“每从显存搬一字节数据，能做多少次计算”。强度低，则搬的多、算得少，是 memory-bound；
强度高，则算的多、搬的少，是 compute-bound。**ridge point（脊点）** 是硬件的一个固有分界值 = 硬件峰值算力 ÷ 峰值带宽，
算数强度 < ridge point 是 memory-bound, 算数强度 > ridge point 是 compute-bound。

| 类型 | 特征 | 优化方向 |
|---|---|---|
| **memory-bound** | 算术强度低于 ridge point。elementwise、norm、softmax、绝大多数算子 | **减少 DRAM 往返** = fusion |
| **compute-bound** | matmul、attention | **喂饱 tensor core** = tiling、流水、L2 复用 |

用 Nsight Compute（ncu），它会直接告诉你 kernel 是 memory 还是 compute bound、达到峰值的百分比、以及 roofline 图。
但我们还是了解一下计算的方法，方便我们自己粗判。

**如何计算算数强度？**

以 softmax 算子为例，设输入是一个长度为 N 的 **fp32** 向量 $x$（比如注意力里的一行 logits），softmax 定义（带数值稳定的减最大值版本，实际都这么实现）：

$$m = \max_i(x_i), \quad y_i = \frac{e^{x_i - m}}{\sum_j e^{x_j - m}}$$

第一步：数 FLOP

softmax 分几个阶段，逐个数每个元素贡献多少次运算：

1. **求最大值 $m = \max(x_i)$**
N 个元素求 max，需要 N−1 次比较。比较通常按 1 次运算算 → 约 **N** 次。

2. **算 $e^{x_i - m}$**
每个元素：1 次减法（$x_i - m$）+ 1 次 exp。
这里有个关键点：**exp 不是 1 次 FLOP**。它是超越函数，硬件上要用多条指令近似（多项式展开等），习惯上按较大的等效 FLOP 计（常见按 ~10 FLOP 估，具体看实现）。为了演示，先记作每次 exp 为 $c$ 次运算。
每元素：$1 + c$，共 N 个 → **$N(1+c)$**

3. **求和 $\sum_j e^{x_j - m}$**
N 个数相加，N−1 次加法 → 约 **N**

4. **除法 $y_i = (\cdot)/\text{sum}$**
每元素 1 次除法，共 **N**（实际常优化成算一次倒数再乘，但量级不变）

5. **合计：**

$$\text{FLOP} \approx N + N(1+c) + N + N = N(3 + 1 + c) = N(4 + c)$$

如果把 exp 粗略按 $c \approx 10$ 估：

$$\text{FLOP} \approx N \times 14 = 14N$$

即使把 exp 当成 1（最保守），也就 $5N$。**量级上是几倍到十几倍 N。**

第二步：数字节

看数据进出显存多少次：

- 读 x：一次，4N 字节
- 写 y：一次，4N 字节

$$\text{字节} \approx 4N + 4N = 8N$$

第三步： 计算算数强度

取 exp≈10：

$$\text{算术强度} = \frac{14N}{8N} = \frac{14}{8} \approx 1.75 \text{ FLOP/Byte}$$


**如何估算 ridge point？**

关于峰值算力，可以通过查厂商 datasheet 或架构白皮书获得，但理论峰值实际达不到，真正做 roofline 时更该用实测可达峰值，跑一个大的 matmul（用 cuBLAS，比如 torch.matmul 两个大方阵），算 FLOP ÷ 耗时。白皮书中会有 tensor 算力和 non-tensor 算力，选你这个 kernel 实际会用到的那套单元的算力。

关于峰值带宽，可以通过查厂商的规格得出（Memory Bandwidth），而实测带宽只能达到理论带宽的 70%-90%，可以使用官方工具 bandwidthTest 或者自己写一个纯拷贝 kernel 实测带宽。

两者相除，就得到了 ridge point。

#### 2.3.3 计算带宽百分比

```python
ms = triton.testing.do_bench(lambda: kernel_call(), return_mode='median') # 单次执行的 ms 数
gbps = bytes_moved * 1e-9 / (ms * 1e-3)
print(f"{gbps:.0f} GB/s = {gbps/REF_GBPS:.0%}")   # REF_GBPS：硬件峰值带宽
```
这里的 bytes_moved 可以是理论上的最小值：输入字节数 + 输出字节数，也可以是实际的 DRAM 访存量。

| 指标 | bytes_moved | 用途 |
|---|---|---|
| **有效吞吐** | 理论最小流量 | 同样的理论最小流量，带宽百分比高的算的更快，用于横向比较不同的实现 |
| **实际 DRAM 带宽** | 真实访存量 | 判断某实现还有没有空间 |


我们可以计算有效吞吐，然后除以峰值带宽，≥80% 就收工，50~80% 调参，<50% 往下查。


#### 2.3.4 Fusion：解决 memory-bound 问题

`y = relu(x*a+b)` 在 PyTorch 里是三次 kernel 启动、6 次 DRAM 访问；融合成一个 kernel 后是 2 次。**访存降到 1/3，速度快 3 倍。**

寄存器/shared memory 比 DRAM 快一两个数量级，数据一旦读进寄存器，在里面做多少次运算几乎免费。


#### 2.3.5 检查寄存器 spill

```python
k = kernel.warmup(..., grid=(1,))
k._init_handles()
# n_regs：每个 thread（线程）用的寄存器数量
# n_spills: 每个 thread 的寄存器溢出数量
# smem：每个 block 用的 shared memory 字节数
print(f"n_regs={k.n_regs}, n_spills={k.n_spills}, smem={k.metadata.shared}")
```
**`n_spills > 0` 是性能灾难**——寄存器装不下，溢出到 local memory（其实在显存里）。

补充一下寄存器和 shared memory 分别放什么：
- 寄存器：单个线程私有，存放索引、临时变量、累加器、刚 load 的值，编译器自动管理。
- shared memory：整个 block 共享，存放待复用的数据块 (A/B tile)、线程间交换的中间量。

#### 2.3.6 occupancy

occupancy（占用率）的含义：一个 SM 上能同时驻留多少个 block。

```python
# 按寄存器算能放几个 block
occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
# 按 shared memory 算能放几个 block，取更小值
occupancy = min(occupancy, SIZE_SMEM // size_smem)
```

GPU 在等待一条访存指令时，会切换到另一个已经就绪的 warp/block 去执行，让计算单元不空闲，
这种用大量可执行单元来填满等待空隙的机制，叫做 latency hiding（延迟隐藏），需要 SM 中有足够多的 block/warp 可以切换。

所以当 occupancy 高的时候，访存延迟容易被计算填满，使硬件利用率增加，但是 occupancy 过高的话，意味着一个 SM 上的 block 更多，
那么每个 block、每个线程能分到的寄存器更少，寄存器不够的话，出发 spill，得不偿失。occupancy 需要设一个足够藏延迟又不出发 spill 的值。


#### 2.3.7 L2 复用（compute-bound 专属）

matmul 的 swizzle：把线性 pid 重排成"分组列主序"，让同时活跃的 program 覆盖一个接近**方形**的输出区域。

原理：同样数量的输出 tile，排成方块比排成长条需要的输入数据少得多（周长最小）。9×9 网格算 9 个 tile，行主序要读 90 个 block，3×3 分组只要 54 个。

#### 2.3.8 参数调优：交给 autotune

`BLOCK_SIZE`、`num_warps`、`num_stages` 的最优值**推不出来**，受寄存器压力、occupancy、L2 命中率的耦合影响，还随硬件和 shape 变化。

```python
@triton.autotune(configs=[...], key=['M', 'N', 'K'])
```

粗略直觉（仅供构造 config 列表）：

- `num_warps`：`BLOCK_SIZE` 大就配大的。512→2，2048→4，8192→8
- `num_stages`：shared memory 决定上限。4090（99KB）扫 2~3，H100（228KB）才扫 4~5
- `GROUP_SIZE_M`：8 是常见默认值

#### 2.3.9 访存合并

GPU 访问显存不是一个字节一个字节读的，而是一次搬一整段连续的内存（一个 transaction/事务，通常 32、64 或 128 字节），
哪怕你只要其中 4 个字节（一个 fp32），硬件也必须把整段 128 字节都搬回来。每搬回来一段内存，里面真正被用到的数据越多越好。

而 GPU 又是以 warp（32 个线程）为单位锁步执行的——一个 warp 里 32 个线程会同时发出各自的 tl.load。如果这32个线程要读的地址连续（线程 0 读地址 0，线程 1 读地址 4，线程 2 读地址 8……fp32 步长 4），这 32 次请求正好落在同一段连续内存里，硬件打包成一次（或极少数几次）内存事务就全取回来了。这叫合并访存（coalesced），有效带宽高。

所以尽量让 **stride=1 的那一维是连续维，访存效率最高。** 让 block 的最内层沿着这个方向展开，相邻线程读相邻地址，硬件可以把多次请求打包成一次事务。

对于非连续的输入，要么传 stride 让 kernel 处理（省拷贝，访存可能不合并），或者 `.contiguous()` 物化（多一次拷贝，后续高效）。matmul 对 B 通常选后者，elementwise 通常选前者。

#### 2.3.10 算法层面的重构

比如 FlashAttention 这个算子，使用 online softmax 只扫一趟数据，一边流式地读入、一边维护结果，不需要预先知道全局的 max 和 sum。
对于 S×S 的注意力大矩阵从头到尾没有完整地存在过——每个小块算完、累进结果就扔了，永远不落显存。显存占用从 O(S²) 降到 O(S)。

#### 2.3.11 看看 torch 编译器做到了什么程度

在装了 torch+triton、能连上加速器的机器上:

```bash
TORCH_LOGS="output_code" python your_script.py  2>&1 | tee dump.log
```

或者想要生成文件，而不是一坨 stdout：

```bash
TORCH_COMPILE_DEBUG=1 python your_script.py 
```
会在当前目录下生成 torch_compile_debug/run_.../output_code.py,可以直接拿编辑器打开。

`torch.compile` 生成的就是 Triton 代码。**如果它已经融合得很好，手写优化空间不大。** 这也是极好的学习素材。


#### 2.3.12 查看 TTGIR

你的 Python kernel(@triton.jit)
   ↓
TTIR   (Triton IR)          ← 硬件无关,描述"算什么"
   ↓
TTGIR  (Triton GPU IR)      ← 加入 GPU 硬件信息,描述"怎么在 GPU 上摆布"
   ↓
LLVM IR
   ↓
PTX (NVIDIA) / 汇编
   ↓
机器码(SASS)

TTGIR（这一层）：硬件相关。它在 TTIR 的基础上，加入了数据布局（layout）、线程/warp 如何分工、用不用 shared memory、tensor core 怎么调度等 GPU 专属的决策。
它是"怎么把这个计算高效映射到 GPU 硬件"的蓝图。

```bash
TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=./dump python bench.py
D=$(ls -dt dump/*/ | head -1)
grep -cE "convert_layout|local_alloc" $D/*.ttgir   # layout 转换的真实数量
grep -c  "tt.reduce"                  $D/*.ttgir   # 规约次数，多了就是机会
grep -c  "async_copy"                 $D/*.ttgir   # 流水生效没
grep     "^#blocked\|^#mma\|^#shared" $D/*.ttgir   # 布局决策
grep     "tt.dot"                     $D/*.ttgir   # 确认走了 tensor core
```

上面的代码展示了如何获取 TTGIR，以及一般我们关注的 TTGIR 内容。

关于布局决策，我们看一个例子

```mlir
#blocked = #ttg.blocked<{sizePerThread = [1, 8],
                         threadsPerWarp = [8, 4],
                         warpsPerCTA   = [4, 1],
                         order         = [1, 0]}>
```


| 字段 | 含义 |
|---|---|
| `sizePerThread` | 每个线程持有多大一块（`[1,8]` = 沿 dim1 连续 8 个 → 编译器在准备向量化访存）|
| `threadsPerWarp` | warp 里 32 个线程怎么排（`[8,4]` = 8 行 × 4 列，乘积必须是 32）|
| `warpsPerCTA` | warp 怎么排（`[4,1]` = 4 个 warp 竖着叠）|
| `order` | 哪一维变化最快 |

**覆盖的 tile 大小 = 每维三个数相乘**：dim0 是 `1×8×4=32`，dim1 是 `8×4×1=32`，所以这是一个 `[32,32]` 的 tile。

layout 有几种类型：

- `#blocked` — 普通的访存和 elementwise
- `#mma` — `tl.dot` 的输出，形状由 tensor core 指令决定
- `#dot_op` — `tl.dot` 的输入操作数
- `#shared` — shared memory 里的布局，**含 swizzle 信息**（bank conflict 的规避策略就在这里）



