---
title: "术语表"
---

## A

**激活重算（Activation Recomputation）**（又称：*梯度检查点*）：在前向传播中丢弃中间激活、在反向传播中重新计算的技术，以约 33% 的额外计算换取显著的内存节省。

**AllGather（全收集）**：每个进程贡献一个分片、结果为所有分片之连接并出现在所有进程上的集合操作。

**AllReduce（全规约）**：把全部进程的值组合起来并把结果分发给所有进程的集合操作。对满足结合律/交换律的规约而言，等价于先 ReduceScatter 再 AllGather（算法等价，而非语义上的逆运算）。

**All-to-All（全交换）**：执行转置的集合操作；每个进程向其他每个进程发送不同的数据。

**算术强度（Arithmetic Intensity）**：FLOPs 与内存访问字节数之比。决定一个操作是计算受限还是内存受限。

## B

**气泡（Bubble）**：流水线并行中由流水线启动/排空造成的空闲时间。

**桶（Bucket）**：为单次 AllReduce 聚合起来摊销延迟的一批小张量。

**总线带宽（Bus Bandwidth, busbw）**：用算法相关的修正因子归一化集合通信时间的有效带宽指标，使不同集合操作和 GPU 数量之间可以公平比较。

## C

**容量因子（Capacity Factor）**：在 MoE 模型中，专家缓冲区大小与每专家期望 token 数之比。容量因子 1.0 表示每个专家恰好处理其公平份额；大于 1.0 的取值以内存代价容忍路由不均衡。

**Chinchilla 扩展律（Chinchilla Scaling）**：计算最优训练，其中 token 数 ≈ 20× 参数数。

**通信强度（Communication Intensity）**：FLOPs 与网络通信字节数之比。决定一个操作是否受网络限制。

**上下文并行（Context Parallelism, CP）**：专门针对注意力计算沿序列维度跨设备切分的并行策略，使长上下文训练成为可能（例如通过 Ring Attention）。

**临界批大小（Critical Batch Size）**：梯度噪声等于梯度信号时的批大小。超过它，更大批次带来递减的回报。

## D

**数据并行（Data Parallelism, DP）**：在各设备上复制模型、切分数据的并行策略。通过梯度 AllReduce 同步。

**设备网格（Device Mesh）**：用于组织设备的 N 维抽象，每个轴对应一种不同的并行策略。

**DiLoCo**：分布式低通信训练（Distributed Low-Communication）。一种让各工作节点执行本地 SGD 步、仅进行不频繁全局同步的方法，从而减少节点间通信。

**DualPipe**：一种先进的流水线并行技术（DeepSeek-V3），交错两条流水线，使气泡比例相对标准 1F1B 大约减半。

## E

**专家并行（Expert Parallelism, EP）**：把 MoE 专家分布到各设备上的并行策略。

## F

**FP8**：用于训练的 8 位浮点格式。有两种变体：E4M3（4 位指数、3 位尾数）用于前向激活，E5M2（5 位指数、2 位尾数）用于梯度。需要逐张量或逐块的缩放因子。

**FlashAttention**：IO 感知的精确注意力算法，通过对计算分块以利用 GPU SRAM，避免物化完整的 $O(S^2)$ 注意力矩阵，将内存从 $O(S^2)$ 降到 $O(S)$。

**FSDP**：完全分片数据并行（Fully Sharded Data Parallel）。PyTorch 的 ZeRO 式分片（支持多种分片模式，包括类 ZeRO-3）。

## G

**GQA（分组查询注意力）**：多个查询头共享单个键值头的注意力变体，在保持质量的同时减少 KV 缓存内存和参数数量。用于 LLaMA 2/3、Mistral 等。

**梯度累积（Gradient Accumulation）**：在同步前跨多个微批计算梯度，相当于增大批大小。

**梯度检查点（Gradient Checkpointing）**：见*激活重算*。

## H

**HBM**：高带宽内存（High Bandwidth Memory）。带宽最高约 3 TB/s 的 GPU 内存类型（视设备而定）。

## L

**LAMB/LARS**：逐层自适应速率缩放（Layer-wise Adaptive Rate Scaling）。用于大批量训练的优化器修改。

**损失缩放（Loss Scaling）**：混合精度训练中的技术，在反向传播前把损失乘以一个大因子，防止小梯度在 FP16 中下溢为零，然后在优化器步之前还原。

## M

**MFU**：模型 FLOP 利用率（Model FLOP Utilization）。实测模型 FLOP/s（仅有用计算）与理论峰值 FLOP/s 之比。

**微批（Micro-batch）**：为流水线并行而细分的批。

**MLA（多头潜在注意力）**：一种注意力变体（DeepSeek-V2/V3），通过可学习的低秩潜在空间压缩键值投影，大幅减少 KV 缓存内存。

**MoE（专家混合）**：每层有多个并行 FFN「专家」的架构，路由机制为每个 token 选择稀疏子集（top-k）专家，使模型更大而不成比例增加计算。

## N

**NVLink**：节点内高速互连。NVLink 4.0（H100）配合 NVSwitch 每 GPU 提供约 900 GB/s 双向带宽；NVLink 3.0（A100）提供约 600 GB/s。

## P

**流水线并行（Pipeline Parallelism, PP）**：把模型层切分到各设备上的并行策略。

## R

**ReduceScatter（规约散射）**：规约数值并把分片散射给各进程的集合操作。

**Ring Attention**：一种上下文并行技术，把键值块分布到排成逻辑环的各 GPU 上。每块 GPU 针对本地 KV 块计算注意力，同时与邻居收发数据块，从而把通信与计算重叠。

**环形算法（Ring Algorithm）**：把进程组织成逻辑环的带宽最优集合算法。

**RoPE（旋转位置编码）**：对查询和键向量施加旋转矩阵的位置编码方法，实现相对位置感知与长度外推。

**屋顶线模型（Roofline Model）**：以计算上限、内存上限或（扩展的）网络上限来界定吞吐的性能分析框架。

## S

**序列并行（Sequence Parallelism, SP）**：沿序列维度切分的并行策略。

**投机解码（Speculative Decoding）**：一种推理优化，用小「草稿」模型生成候选 token，再由大模型并行验证，在不改变输出的前提下改善延迟。

**SwiGLU**：结合 Swish 与门控线性单元的激活函数：$\text{SwiGLU}(x) = \text{Swish}(xW_1) \odot (xW_2)$。被现代 LLM（LLaMA、Mistral、DeepSeek）广泛采用，以多一次线性投影为代价，质量优于标准 GELU。

## T

**张量并行（Tensor Parallelism, TP）**：把单个张量操作（矩阵乘法）切分到各设备上的并行策略。

## Z

**ZeRO**：零冗余优化器（Zero Redundancy Optimizer）。在数据并行 rank 间分片优化器状态（ZeRO-1）、梯度（ZeRO-2）和参数（ZeRO-3）的内存优化。