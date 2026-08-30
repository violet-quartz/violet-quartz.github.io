---
title: 'Triton 学习笔记'
description: ''
pubDate: '2026-08-30'
tags: ['triton', 'gpu', 'kernel']
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

### 2.2 如何测量性能

```python
ms = triton.testing.do_bench(lambda: kernel[grid](x, y, n), return_mode='median')
```

`do_bench` 帮你处理了：CUDA event 精确计时、warmup、多次取统计量、**每轮清空 L2**。

**必须包 `lambda`** —— do_bench 需要的是一个能重复跑去测量的函数，lambda: kernel[grid](x, y, n) 定义了一个没有参数的匿名函数，把"该怎么调用 kernel"这件事打包成一个函数对象传进去；如果直接用 kernel[grid](x, y, n)，则会把 kernel 的运行结果传入 do_bench。

**需要每轮清空 L2** —— 现代 GPU 的 L2 很大，数据第一次读进来之后就待在 L2 里了，后续多次运行反复读写全在 L2 上跑，没有碰到显存，测到的不是实际情况。


### 2.3 如何判断瓶颈并进行优化



