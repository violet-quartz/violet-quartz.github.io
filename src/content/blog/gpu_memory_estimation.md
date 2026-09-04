---
title: '大语言模型推理和训练显存估算'
description: '本文以 Qwen3-8B 为例，我们对其推理/训练占用的显存进行了分类和估算，可依据此来筛选可用的 GPU 机器'
pubDate: '2026-09-04'
tags: ['llm', 'inference', 'training', 'memory estimation']
---

以 Qwen3-8B 为例，我们对其推理/训练所需显存进行估算。

这是 Qwen3-8B 的 [config](https://modelscope.cn/models/Qwen/Qwen3-8B/file/view/master/config.json)：

```json
{
  "architectures": [
    "Qwen3ForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "eos_token_id": 151645,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 12288,
  "max_position_embeddings": 40960,
  "max_window_layers": 36,
  "model_type": "qwen3",
  "num_attention_heads": 32,
  "num_hidden_layers": 36,
  "num_key_value_heads": 8,
  "rms_norm_eps": 1e-06,
  "rope_scaling": null,
  "rope_theta": 1000000,
  "sliding_window": null,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.51.0",
  "use_cache": true,
  "use_sliding_window": false,
  "vocab_size": 151936
}
```

## 1 推理显存估算

Qwen3-8B 的推理显存估算可以拆成三部分：模型权重 + KV cache + 激活值/框架开销。

### 1.1 模型权重

8B (80亿) 参数，精度是 BF16，显存占用 ～16GB。

假设模型有 xB 参数，每个参数占 y 个字节，则显存占用 xy GB。

### 1.2 KV cache

公式：

2 * num_layers * num_kv_heads * head_dim * batch_size * seq_len * 精度字节数

乘以 2 是因为有 K cache， 同时有 V cache。

对于 Qwen3-8b 来说，
- num_layers = 36
- num_kv_heads = 8
- head_dim = 128

以单条请求，BF16（精度字节数 = 2），对于每个 token， 其 KV cache 为

2 * 36 * 8 * 128 * 2 bytes = 147,456 bytes

在不同 seq_len 之下估算：

| seq_len | KV Cache 大小（单条请求） |
|---|---|
| 4K | ~0.56 GB |
| 8K | ~1.13 GB |
| 32K | ~4.5 GB |
| 128K | ~18 GB |

如果是多并发请求的话，还需要乘以并发数 batch_size。

### 1.3 激活值和框架开销

在推理场景里，"激活值"（Activations）指的是模型前向计算过程中，每一层产生的中间张量（intermediate tensors），不是权重，也不是 KV Cache，而是计算过程中"流动"的数据。

具体包括：

以 Transformer 的一次前向计算为例，每一层会产生：

- Attention 部分：Q/K/V 投影后的张量、attention score 矩阵（softmax 前后）、attention 输出
- FFN 部分：升维后的中间张量（比如 hidden_dim → 4×hidden_dim 那一层）、激活函数（SiLU/GELU等）输出
- LayerNorm / RMSNorm 的中间结果
- Embedding 层输出、最后 lm_head 投影到词表维度的 logits（这个在长序列或大词表时其实不小）

这些张量在计算完当前层后，理论上可以释放，但推理框架为了效率通常会保留一部分（比如做 pipeline parallelism，cuda graph 预分配buffer等）。

**推理时，激活值的主要来源：**
- prefill 阶段的 attention score 矩阵 （batch * num_heads * seq_len * seq_len）在 promot 比较长的时候，峰值显存较大
- prefill 阶段的 FFN 中间升维张量 （batch × seq_len × intermediate_size）

激活值会随着 batch_size 和序列长度的变化而变化。

**框架开销包括：** CUDA context（几百 MB）、cuDNN/cuBLAS 的 workspace、显存碎片、算子融合的临时 buffer 等。

这两部分通常预留 1-2GB，粗略按 10-15% 的模型权重量估算即可。

### 1.4 综合估算

| 场景 | 权重 | KV Cache | 激活值和框架开销 | 总计（约） |
|---|---|---|---|---|
| 短文本（4K上下文，batch=1） | 16GB | 0.56GB | ~1GB | **~18GB** |
| 短文本（4K上下文，batch=2） | 16GB | 1.12GB | ~2GB | **~19GB** |
| 中等上下文（32K，batch=1） | 16GB | 4.5GB | ~2GB | **~23GB** |
| 长上下文（128K，batch=1） | 16GB | 18GB | ~2GB | **~36GB** |
| INT4量化（4K上下文） | 5GB | 0.56GB | ~1GB | **~7GB** |

## 2 训练显存估算

这里我们主要考虑微调的显存估算。

总显存 = 模型权重 + 梯度 + 优化器状态 + 激活值 + 框架开销

跟推理相比，微调的显存多出了梯度和优化器状态，没有 KV Cache 的花销（训练不是一步一步的解码，而是一次并行的前向计算），而且激活值也会比推理大得多（因为**反向传播需要用到每一层前向计算时的中间结果**，用于计算梯度）。

### 2.1 关于激活值

粗略估算公式：

```
激活值 ≈ batch_size × seq_len × hidden_size × num_layers × 系数(通常10-20+) × 2 bytes
```

系数取决于模型结构细节（是否用 FlashAttention、算子融合、具体模型结构），这里给一个直观感受：**seq_len 越长、batch 越大，激活值增长非常快**，很多时候训练 OOM 都是卡在这里而不是权重/优化器上。以 seq_len=4K，batch=1 的 Qwen3-8B 来看，激活值为 3GB-6GB。


由于激活值估算不准，如果想要精确值，最靠谱的方法还是：用训练数据在大机器上跑一次实际的 forward+backward，用 torch.cuda.max_memory_allocated() 直接测量峰值显存，而激活值的真实占用，可以用测量峰值减去权重、梯度、优化器状态这几块理论可算的部分来得到。

### 2.2 全量微调

以 8B 参数、**混合精度训练（BF16权重 + FP32优化器状态）**、Adam 优化器为例：

| 项目 | 精度 | 每参数字节数 | 显存占用 |
|---|---|---|---|
| 模型权重 | BF16 | 2 bytes | 16 GB |
| 梯度 | BF16 | 2 bytes | 16 GB |
| Adam 一阶矩(m) | FP32 | 4 bytes | 32 GB |
| Adam 二阶矩(v) | FP32 | 4 bytes | 32 GB |
| FP32 权重副本(可选，用于精度) | FP32 | 4 bytes | 32 GB |

**不带 FP32 权重副本**（纯 BF16 训练）：
```
16 + 16 + 32 + 32 = 96 GB（仅权重+梯度+优化器状态）
```

**带 FP32 权重副本**（更常见，精度更稳，这是 Adam 混合精度训练的标准做法）：
```
16 + 16 + 32 + 32 + 32 = 128 GB
```

再加上激活值（训练时激活值远大于推理，因为要保留反向传播需要的中间结果），**全量微调 8B 模型基本需要多卡（比如 4×A100 80GB 或更多）**，单卡很难跑起来，可以用 **gradient checkpointing**（用重算换显存，可以把激活值降低数倍，但训练速度会慢 20-30%）配合 **ZeRO 等分布式优化**（把优化器状态、梯度切分到多卡）。

### 2.3 LoRA 微调

LoRA 冻结原模型权重，只训练额外插入的低秩矩阵（通常只占原参数量的 0.1%~1%），显存需求断崖式下降：

| 项目 | 说明 | 显存占用（估算） |
|---|---|---|
| 基座模型权重 | 冻结，BF16 | 16 GB |
| LoRA 参数 | 通常几千万到几亿，很小 | ~0.1-0.5 GB |
| LoRA 梯度 | 只对 LoRA 参数 | ~0.1-0.5 GB |
| Adam 优化器状态 | 只对 LoRA 参数（m+v） | ~0.2-1 GB |
| 激活值 | 仍需保留（反向传播要经过冻结层） | 视 batch/seq_len 而定 |

```
基座权重(16GB) + LoRA相关(~1-2GB) + 激活值(视配置) + 框架开销
≈ 20-30GB（典型配置，4K上下文，batch较小）
```

因此 **RTX 4090（24GB）跑 Qwen3-8B 的 LoRA 微调基本可行**，是目前最常见的单卡微调方案。


### 2.4 QLoRA（量化 + LoRA）微调

 LoRA 基础上，**把冻结的基座模型量化成 INT4**，训练时只有 LoRA 部分保持高精度：

| 项目 | 精度 | 显存占用 |
|---|---|---|
| 基座模型权重 | INT4（量化） | ~5 GB |
| LoRA 参数+梯度+优化器状态 | BF16/FP32 | ~1-2 GB |
| 激活值 | 计算时反量化，但激活本身仍是 BF16 | 视配置而定 |

```
5GB(量化基座) + 1-2GB(LoRA) + 激活值 + 框架开销
≈ 10-15GB（典型配置）
```

这样 **12GB 显存的消费级显卡**也有可能跑起来（比如 RTX 3060 12GB、RTX 4070），代价是训练速度会比 LoRA 慢一些（量化反量化有额外开销），且精度略有损失。

说明：基座模型权重 INT4 量化后，显存占用是 5GB 而不是 4GB，原因一是Embedding 层、LM head、LayerNorm 权重这些通常保持原精度（BF16/FP16），二是量化本身也会有一些额外的开销。








