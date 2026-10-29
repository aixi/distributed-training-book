---
title: "从可分解性推导序列并行"
subtitle: "沿序列维度分片"
---

<div class="chapter-opener" markdown>
当序列增长到数百万 token 时，即便单次注意力计算也放不进内存。序列并行（sequence parallelism）利用注意力的可分解性，沿序列维度进行切分。
</div>

<div class="investigation-question" markdown>
**问题**：注意力在序列长度上是 O(S²) 的。对 S=1M 个 token，就是 10^12 个注意力分数。当没有单块 GPU 能容纳注意力矩阵时，我们该怎么算？
</div>

!!! abstract "本章导览"
    **前置知识**：[第 11 章](../collectives/11-primitives-properties.md)（AllGather、ReduceScatter、AlltoAll）、[第 15 章](15-tensor-parallelism-linearity.md)（张量并行与 Megatron-LM 模式）

    **核心洞见**：序列并行利用注意力的*可分解性*——具体说是在线 softmax（online softmax）的结合律——沿序列维度拆分计算。Megatron SP 把非张量并行（Tensor Parallelism）区域（LayerNorm、Dropout）分片以节省激活内存，而上下文并行（context parallelism，Ring Attention、Ulysses）则能处理单块 GPU 内存放不下的超长序列。

## 为什么需要序列并行？

随着模型处理更长的上下文——文档、代码库、视频——序列长度 $S$ 成为瓶颈。

### 内存随序列长度的扩展

对一个 Transformer 层：

**每层激活内存**：

$$M_{\text{act}} = 2 \cdot b \cdot S \cdot H + b \cdot S \cdot S \cdot n_h$$

其中：

- 第一项：输入与输出激活（$2 \times b \times S \times H$）
- 第二项：注意力矩阵（$b \times n_h \times S \times S$）

注意力矩阵按 $O(S^2)$ 扩展。

**示例**：$S = 128K$、$b = 1$、$n_h = 32$、fp16：

$$M_{\text{attn}} = 1 \times 32 \times 128K \times 128K \times 2 = 1 \text{ TB}$$

没有单块 GPU 能容纳它。

### 序列维度

与批或隐藏维度不同，序列维度出现在：

1. **注意力**：$Q K^T$ 的形状是 $(S \times S)$
2. **层归一化**：按 token 计算统计量（相互独立）
3. **前馈**：按 token 应用（相互独立）
4. **位置编码**：与位置相关

只有注意力会产生跨 token 依赖。

## 序列并行的两种形态

### 1. Megatron 序列并行

在张量并行内部沿序列维度分布式放置，以减少 LayerNorm 与 Dropout 的激活内存。

**目标**：注意力之外的激活内存。

**通信**：注意力之前 AllGather，之后 ReduceScatter。

### 2. 上下文并行（Ring Attention / Ulysses）

把注意力计算本身分布到序列维度上。

**目标**：$O(S^2)$ 的注意力矩阵。

**通信**：P2P 环形（Ring Attention）或 AlltoAll（Ulysses）。

下面逐一详述。

## Megatron 序列并行

Korthikanti 等人（2022）为节省内存引入了序列并行。

### 内存问题

在张量并行中，某些操作是复制的：

```
Before LayerNorm: activation shape (b, S, H) replicated on all TP ranks
After LayerNorm:  same shape, still replicated
Before Dropout:   replicated
After Dropout:    replicated
```

TP 度为 $T$ 时，这浪费了 $(T-1) \cdot b \cdot S \cdot H$ 的内存。

### 解决方案：分布序列

把序列维度切分到各 TP rank：

```
Rank 0: tokens 0 to S/T - 1
Rank 1: tokens S/T to 2S/T - 1
...
Rank T-1: tokens (T-1)S/T to S - 1
```

### 通信模式

**列并行层之前**（每个 rank 需要完整序列）：

$$\text{AllGather along sequence} \to (b, S, H)$$

**行并行层之后**（输出是部分的，需要跨 rank 求和）：

$$\text{ReduceScatter along sequence} \to (b, S/T, H)$$

```
                    ┌─────────────┐
                    │  LayerNorm  │  Sequence-parallel
                    │  (b, S/T, H)│
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  AllGather  │
                    │  (b, S, H)  │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ Col-Parallel│  Tensor-parallel
                    │   Linear    │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ Row-Parallel│
                    │   Linear    │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ReduceScatter│
                    │  (b, S/T, H)│
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  Dropout    │  Sequence-parallel
                    │  (b, S/T, H)│
                    └─────────────┘
```

### 内存节省

**无序列并行时**：

$$M_{\text{LN+Dropout}} = 2 \cdot b \cdot S \cdot H \text{ (replicated)}$$

**有序列并行时**：

$$M_{\text{LN+Dropout}} = 2 \cdot b \cdot \frac{S}{T} \cdot H$$

节省倍数：$T$（张量并行度）。

### 通信分析

每个 Transformer 层（每 rank，环形集合通信）：

- 注意力之前 1 次 AllGather
- 注意力之后 1 次 ReduceScatter
- FFN 之前 1 次 AllGather
- FFN 之后 1 次 ReduceScatter

**每层总计（每 rank）**：$4 \cdot \frac{P-1}{P} \cdot b \cdot S \cdot H \cdot \text{sizeof(dtype)}$

这**与张量并行的 AllReduce 通信量相同**，只是重新组织过。

### 实现

```python
class SequenceParallelLayerNorm(nn.Module):
    """LayerNorm operating on sequence-parallel inputs."""

    def __init__(self, hidden_size: int, tp_group):
        super().__init__()
        self.tp_group = tp_group
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_local, hidden)
        # LayerNorm is per-token, so no cross-rank communication needed
        return F.layer_norm(x, (x.size(-1),), self.weight, self.bias, self.eps)

def sequence_parallel_attention(q, k, v, tp_group):
    """Attention with sequence-parallel input/output."""
    # Input: (batch, seq_local, hidden)

    # AllGather to get full sequence
    q_full = all_gather_sequence(q, tp_group)  # (batch, seq, hidden)
    k_full = all_gather_sequence(k, tp_group)
    v_full = all_gather_sequence(v, tp_group)

    # Standard attention on full sequence
    output = attention(q_full, k_full, v_full)  # (batch, seq, hidden)

    # Shard back to local sequence chunk.
    # This pedagogical helper is a pure scatter (no reduction).
    output_local = scatter_sequence_chunk(output, tp_group)

    return output_local  # (batch, seq_local, hidden)

def all_gather_sequence(x: torch.Tensor, group) -> torch.Tensor:
    """AllGather along sequence dimension."""
    world_size = dist.get_world_size(group)
    gathered = [torch.empty_like(x) for _ in range(world_size)]
    dist.all_gather(gathered, x, group=group)
    return torch.cat(gathered, dim=1)  # Concat along seq dimension

def scatter_sequence_chunk(x: torch.Tensor, group) -> torch.Tensor:
    """Shard tensor along sequence dimension (no reduction)."""
    world_size = dist.get_world_size(group)
    seq_len = x.size(1)
    if seq_len % world_size != 0:
        raise ValueError("Sequence length must be divisible by world size (pad or use uneven splits).")
    local_seq = seq_len // world_size

    # Split into chunks
    chunks = x.split(local_seq, dim=1)
    rank = dist.get_rank(group)
    return chunks[rank].contiguous()
```

## 注意力的可分解性

要实现真正的序列并行——把注意力计算本身划分开——我们需要理解注意力如何分解。

### 标准注意力

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right) V$$

挑战在于：softmax 在**整个**键序列上归一化：

$$\text{softmax}(x)_i = \frac{e^{x_i}}{\sum_{j=1}^{S} e^{x_j}}$$

在看不到所有键的情况下，我们无法计算分母。

### 在线 softmax

Milakov 与 Gimelshein（2018）证明 softmax 可以增量计算。

**洞见**：追踪运行中的最大值与和。

为保证数值稳定性：

$$\text{softmax}(x)_i = \frac{e^{x_i - m}}{\sum_j e^{x_j - m}}$$

其中 $m = \max_j x_j$。

**增量更新**：给定块 $A$ 与 $B$：

$$m_{AB} = \max(m_A, m_B)$$

$$\text{sum}_{AB} = \text{sum}_A \cdot e^{m_A - m_{AB}} + \text{sum}_B \cdot e^{m_B - m_{AB}}$$

**输出更新**：

$$\text{out}_{AB} = \frac{\text{out}_A \cdot \text{sum}_A \cdot e^{m_A - m_{AB}} + \text{out}_B \cdot \text{sum}_B \cdot e^{m_B - m_{AB}}}{\text{sum}_{AB}}$$

### 在线 softmax 的结合律

**定理**：在线 softmax 的组合满足结合律。

令 $(m, s, o)$ 表示状态（最大值、和、输出）。组合操作 $\oplus$：

$$(m_1, s_1, o_1) \oplus (m_2, s_2, o_2) = (m_{12}, s_{12}, o_{12})$$

满足结合律：

$$((m_1, s_1, o_1) \oplus (m_2, s_2, o_2)) \oplus (m_3, s_3, o_3) = (m_1, s_1, o_1) \oplus ((m_2, s_2, o_2) \oplus (m_3, s_3, o_3))$$

**证明概要**：max 操作满足结合律。和与输出的更新是以"相对全局最大值的指数差"为权重的加权平均。最终结果只取决于所有输入，与组合顺序无关。$\square$

这一结合律使分布式计算成为可能。

## Ring Attention

Liu 等人（2023）为超长序列提出了 Ring Attention。

### 核心思想

每块 GPU 持有：

- **查询块** $Q_i$：本地查询（从不移动）
- **KV 缓冲区**：在环上轮转的键值对

```
Initial:
GPU 0: Q₀, K₀, V₀
GPU 1: Q₁, K₁, V₁
GPU 2: Q₂, K₂, V₂
GPU 3: Q₃, K₃, V₃

After step 1 (K, V rotate):
GPU 0: Q₀, K₃, V₃
GPU 1: Q₁, K₀, V₀
GPU 2: Q₂, K₁, V₁
GPU 3: Q₃, K₂, V₂
```

### 算法

```python
def ring_attention(Q_local, K_local, V_local, ring_group):
    """
    Ring Attention: Compute attention over distributed sequence.

    Args:
        Q_local: Local query chunk (batch, seq_local, heads, dim)
        K_local: Local key chunk (batch, seq_local, heads, dim)
        V_local: Local value chunk (batch, seq_local, heads, dim)
        ring_group: Process group for ring communication

    Returns:
        Output: Attention output for local queries
    """
    world_size = dist.get_world_size(ring_group)
    rank = dist.get_rank(ring_group)

    # Initialize output accumulator with online softmax state
    batch, seq_local, heads, dim = Q_local.shape
    output = torch.zeros(batch, seq_local, heads, dim, device=Q_local.device)
    max_scores = torch.full((batch, seq_local, heads, 1), float('-inf'),
                            device=Q_local.device)
    sum_exp = torch.zeros(batch, seq_local, heads, 1, device=Q_local.device)

    # Current K, V buffers (will rotate)
    K_current = K_local.clone()
    V_current = V_local.clone()

    # Buffers for async communication
    K_recv = torch.empty_like(K_current)
    V_recv = torch.empty_like(V_current)

    for step in range(world_size):
        # Start async receive from previous rank
        if step < world_size - 1:
            src = (rank - 1) % world_size
            recv_k = dist.irecv(K_recv, src=src, group=ring_group)
            recv_v = dist.irecv(V_recv, src=src, group=ring_group)

        # Compute local attention scores
        # Q_local @ K_current^T -> (batch, seq_local, heads, seq_local)
        scores = torch.einsum('bqhd,bkhd->bqhk', Q_local, K_current)
        scores = scores / math.sqrt(dim)

        # Apply causal mask if needed
        kv_offset = ((rank - step) % world_size) * seq_local
        if kv_offset > 0:  # Keys are from "future" positions
            causal_mask = create_causal_mask(seq_local, kv_offset)
            scores = scores.masked_fill(causal_mask, float('-inf'))

        # Online softmax update
        chunk_max = scores.max(dim=-1, keepdim=True).values
        new_max = torch.maximum(max_scores, chunk_max)

        # Rescale previous sum and output
        scale_old = torch.exp(max_scores - new_max)
        scale_new = torch.exp(chunk_max - new_max)

        exp_scores = torch.exp(scores - chunk_max)
        chunk_sum = exp_scores.sum(dim=-1, keepdim=True)

        # Update running sum
        sum_exp = sum_exp * scale_old + chunk_sum * scale_new

        # Update output
        chunk_output = torch.einsum('bqhk,bkhd->bqhd', exp_scores, V_current)
        output = output * scale_old + chunk_output * scale_new
        max_scores = new_max

        # Send K, V to next rank (async)
        if step < world_size - 1:
            dst = (rank + 1) % world_size
            send_k = dist.isend(K_current, dst=dst, group=ring_group)
            send_v = dist.isend(V_current, dst=dst, group=ring_group)

            # Wait for receive and swap buffers
            recv_k.wait()
            recv_v.wait()
            K_current, K_recv = K_recv, K_current
            V_current, V_recv = V_recv, V_current

            # Wait for send to complete
            send_k.wait()
            send_v.wait()

    # Final normalization
    output = output / sum_exp

    return output
```

### 通信模式

每一步：

- 向下一 rank 发送 $K_i, V_i$：$2 \cdot S/P \cdot H$ 个元素
- 从上一 rank 接收：同样多

**每个注意力层的总通信量**：

$$C = 2(P-1) \cdot \frac{S}{P} \cdot H \cdot \text{sizeof} = 2 \cdot \frac{P-1}{P} \cdot S \cdot H \cdot \text{sizeof}$$

**关键特征**：通信与计算重叠。

### 计算-通信重叠

在用当前 K、V 计算注意力的同时：

- 同时把当前 K、V 发送给下一 rank
- 同时从上一 rank 接收下一份 K、V

**重叠效率**：

$$\text{Overlap} = \min\left(1, \frac{T_{\text{compute}}}{T_{\text{comm}}}\right)$$

其中：

$$T_{\text{compute}} = \frac{2 \cdot (S/P)^2 \cdot H}{F_{\text{FLOPs}}}$$

$$T_{\text{comm}} = \frac{2 \cdot (S/P) \cdot H \cdot \text{sizeof}}{\text{bandwidth}}$$

对长序列，计算占主导，重叠近乎完美。

### 内存分析

**每 GPU 峰值内存**：

- Query：$b \cdot \frac{S}{P} \cdot H$
- 两份 KV 缓冲（双缓冲）：$2 \cdot 2 \cdot b \cdot \frac{S}{P} \cdot H$
- 注意力分数（一个块）：$b \cdot n_h \cdot \frac{S}{P} \cdot \frac{S}{P}$
- 输出累加器：$b \cdot \frac{S}{P} \cdot H$

**总计**：

$$M \approx 5 \cdot b \cdot \frac{S}{P} \cdot H + b \cdot n_h \cdot \left(\frac{S}{P}\right)^2$$

二次项如今是 $(S/P)^2$ 而非 $S^2$——缩减了 $P^2$ 倍。

## Ulysses：AlltoAll 序列并行

Fang 等人（2024）提出了 Ulysses，作为 Ring Attention 的替代方案。

### 方法

不是在环上轮转 K、V，而是用 AlltoAll 做重分布：

1. **初始状态**：每块 GPU 拥有某段序列的本地 Q、K、V
2. **对 K 做 AlltoAll**：重分布，使每块 GPU 拥有某些头的全部 K
3. **对 V 做 AlltoAll**：同样的重分布
4. **本地注意力**：对一部分头用完整序列计算注意力
5. **对输出做 AlltoAll**：重分布回序列并行布局

```
Before AlltoAll (sequence-parallel):
GPU 0: Q[0:S/P], K[0:S/P], V[0:S/P] for all heads
GPU 1: Q[S/P:2S/P], K[S/P:2S/P], V[S/P:2S/P] for all heads

After AlltoAll (head-parallel):
GPU 0: Q[0:S], K[0:S], V[0:S] for heads 0:H/P
GPU 1: Q[0:S], K[0:S], V[0:S] for heads H/P:2H/P
```

### 实现

```python
def ulysses_attention(Q, K, V, sp_group):
    """
    Ulysses sequence parallelism using AlltoAll.

    Args:
        Q, K, V: Local chunks (batch, seq_local, heads, dim)
        sp_group: Sequence parallel process group

    Returns:
        Output attention for local sequence chunk
    """
    world_size = dist.get_world_size(sp_group)
    batch, seq_local, heads, dim = Q.shape

    # Reshape for AlltoAll: split heads dimension
    # (batch, seq_local, heads, dim) -> (batch, seq_local, P, heads/P, dim)
    heads_per_rank = heads // world_size
    Q = Q.view(batch, seq_local, world_size, heads_per_rank, dim)
    K = K.view(batch, seq_local, world_size, heads_per_rank, dim)
    V = V.view(batch, seq_local, world_size, heads_per_rank, dim)

    # AlltoAll: exchange sequence chunks for head chunks
    # After: each rank has full sequence for subset of heads
    Q = all_to_all(Q, dim_scatter=2, dim_gather=1, group=sp_group)
    K = all_to_all(K, dim_scatter=2, dim_gather=1, group=sp_group)
    V = all_to_all(V, dim_scatter=2, dim_gather=1, group=sp_group)

    # Now shape is (batch, seq_full, heads_local, dim)
    seq_full = seq_local * world_size
    Q = Q.view(batch, seq_full, heads_per_rank, dim)
    K = K.view(batch, seq_full, heads_per_rank, dim)
    V = V.view(batch, seq_full, heads_per_rank, dim)

    # Standard attention on full sequence (for local heads)
    output = flash_attention(Q, K, V)  # (batch, seq_full, heads_local, dim)

    # Reshape for reverse AlltoAll
    output = output.view(batch, world_size, seq_local, heads_per_rank, dim)

    # AlltoAll: exchange head chunks for sequence chunks
    output = all_to_all(output, dim_scatter=1, dim_gather=2, group=sp_group)

    # Reshape back
    output = output.view(batch, seq_local, heads, dim)

    return output

def all_to_all(x, dim_scatter, dim_gather, group):
    """AlltoAll with specified scatter and gather dimensions."""
    world_size = dist.get_world_size(group)

    # Split along scatter dimension
    splits = x.chunk(world_size, dim=dim_scatter)
    splits = [s.contiguous() for s in splits]

    # AlltoAll
    output_splits = [torch.empty_like(splits[0]) for _ in range(world_size)]
    dist.all_to_all(output_splits, splits, group=group)

    # Concatenate along gather dimension
    return torch.cat(output_splits, dim=dim_gather)
```

### 通信分析

**AlltoAll 通信量**（每个方向）：

$$C = (P-1) \cdot \frac{S}{P} \cdot \frac{H}{P} \cdot b = \frac{(P-1) \cdot S \cdot H \cdot b}{P^2}$$

每个注意力层：4 次 AlltoAll（Q、K、V 进，输出出）。

**总计**：

$$C_{\text{total}} = 4 \cdot \frac{(P-1) \cdot S \cdot H \cdot b}{P^2}$$

### 环形与 Ulysses 的对比

| 方面 | Ring Attention | Ulysses |
|--------|---------------|---------|
| 通信 | 环形 P2P | AlltoAll |
| 通信量 | $2 \cdot \frac{P-1}{P} \cdot S \cdot H$ | $4 \cdot \frac{P-1}{P^2} \cdot S \cdot H$ |
| 重叠 | 是（计算 + 通信） | 有限 |
| 内存 | 需要 KV 缓冲 | 无额外缓冲 |
| 适用 | 长序列、P2P 快 | rank 少、AlltoAll 快 |

**何时用环形**：序列并行 rank 多，且能重叠。

**何时用 Ulysses**：rank 少（2-8），AlltoAll 带宽高（NVLink）。

## Flash Attention 集成

环形与 Ulysses 都能受益于 Flash Attention。

### Flash Attention 回顾

Dao 等人（2022）以分块方式计算注意力，不物化完整的 $S \times S$ 矩阵：

```python
def flash_attention_forward(Q, K, V, block_size=64):
    """
    Flash Attention: memory-efficient attention using tiling.
    """
    batch, seq_q, heads, dim = Q.shape
    seq_k = K.shape[1]

    output = torch.zeros_like(Q)
    max_scores = torch.full((batch, seq_q, heads, 1), float('-inf'))
    sum_exp = torch.zeros((batch, seq_q, heads, 1))

    # Tile over K, V
    for k_start in range(0, seq_k, block_size):
        k_end = min(k_start + block_size, seq_k)
        K_block = K[:, k_start:k_end]
        V_block = V[:, k_start:k_end]

        # Tile over Q
        for q_start in range(0, seq_q, block_size):
            q_end = min(q_start + block_size, seq_q)
            Q_block = Q[:, q_start:q_end]

            # Compute attention scores for this tile
            scores = torch.einsum('bqhd,bkhd->bqhk', Q_block, K_block)
            scores = scores / math.sqrt(dim)

            # Online softmax update (same as Ring Attention)
            block_max = scores.max(dim=-1, keepdim=True).values
            new_max = torch.maximum(max_scores[:, q_start:q_end], block_max)

            scale_old = torch.exp(max_scores[:, q_start:q_end] - new_max)
            scale_new = torch.exp(block_max - new_max)

            exp_scores = torch.exp(scores - block_max)
            block_sum = exp_scores.sum(dim=-1, keepdim=True)

            # Update accumulators
            sum_exp[:, q_start:q_end] = (
                sum_exp[:, q_start:q_end] * scale_old +
                block_sum * scale_new
            )

            block_out = torch.einsum('bqhk,bkhd->bqhd', exp_scores, V_block)
            output[:, q_start:q_end] = (
                output[:, q_start:q_end] * scale_old +
                block_out * scale_new
            )

            max_scores[:, q_start:q_end] = new_max

    # Final normalization
    output = output / sum_exp
    return output
```

### Ring Attention 配合 Flash Attention

Ring Attention 的内层循环可以用 Flash Attention 完成本地计算：

```python
def ring_flash_attention(Q_local, K_local, V_local, ring_group):
    """Ring Attention using Flash Attention for each step."""
    world_size = dist.get_world_size(ring_group)

    # Initialize accumulators
    output, max_scores, sum_exp = init_accumulators(Q_local)

    K_current, V_current = K_local.clone(), V_local.clone()

    for step in range(world_size):
        # Start async communication (overlapped)
        comm_handle = start_ring_comm(K_current, V_current, ring_group)

        # Use Flash Attention kernel for this chunk
        # Returns (output_chunk, max_chunk, sum_chunk)
        chunk_out, chunk_max, chunk_sum = flash_attention_with_state(
            Q_local, K_current, V_current
        )

        # Online softmax merge
        output, max_scores, sum_exp = merge_attention_state(
            output, max_scores, sum_exp,
            chunk_out, chunk_max, chunk_sum
        )

        # Complete communication
        K_current, V_current = complete_ring_comm(comm_handle)

    return output / sum_exp
```

## 混合上下文并行

对超长序列，可以组合多种技术。

### 层次化环形

在不同层级使用多个环：

```
Inter-node ring (slow network):
  [Node 0] ←→ [Node 1] ←→ [Node 2] ←→ [Node 3]

Intra-node ring (NVLink):
  GPU0 ←→ GPU1 ←→ GPU2 ←→ GPU3 (within each node)
```

### 组合策略

```python
class HybridContextParallel:
    """Combine Ulysses (intra-node) with Ring (inter-node)."""

    def __init__(self, local_group, global_ring_group):
        self.local_group = local_group  # GPUs within node
        self.ring_group = global_ring_group  # Across nodes

    def forward(self, Q, K, V):
        # Step 1: Ulysses within node (fast AlltoAll via NVLink)
        Q, K, V = ulysses_qkv_exchange(Q, K, V, self.local_group)

        # Step 2: Ring across nodes (overlapped P2P)
        output = ring_attention(Q, K, V, self.ring_group)

        # Step 3: Ulysses to restore layout
        output = ulysses_output_exchange(output, self.local_group)

        return output
```

## 内存对比

对序列长度 $S$、隐藏维度 $H$、头数 $n_h$、并行度 $P$：

| 方法 | 注意力矩阵内存 | KV 内存 |
|--------|------------------------|-----------|
| 标准 | $O(n_h \cdot S^2)$ | $O(S \cdot H)$ |
| Flash Attention | $O(n_h \cdot B^2)$（块大小 $B$） | $O(S \cdot H)$ |
| Ring Attention | $O(n_h \cdot (S/P)^2)$ | $O(S \cdot H / P)$ |
| Ulysses | $O((n_h/P) \cdot S^2)$ | $O(S \cdot H)$ |

**Ring Attention** 把两个维度都缩减 $P$ 倍；最适合超长序列。

**Ulysses** 缩减头维度 $P$ 倍；每块 GPU 保留完整序列。

## 因果掩码的考量

因果注意力会屏蔽未来的 token：若 $j \leq i$，则 $\text{Mask}[i, j] = 1$。

### Ring Attention 的因果性

当轮转 K、V 时，必须追踪它们代表哪些位置：

```python
def get_causal_mask(q_offset, k_offset, seq_local):
    """Create causal mask for Ring Attention step."""
    q_positions = torch.arange(q_offset, q_offset + seq_local)
    k_positions = torch.arange(k_offset, k_offset + seq_local)

    # Mask: True where k_pos > q_pos (future positions)
    mask = k_positions.unsqueeze(0) > q_positions.unsqueeze(1)
    return mask
```

### 优化：跳过未来块

如果某个块里所有键相对于所有查询都是"未来"，就完全跳过计算：

```python
def should_compute_chunk(q_start, q_end, k_start, k_end):
    """Check if KV chunk has any valid (non-future) positions."""
    # All keys are future if k_start > q_end
    return k_start <= q_end
```

对因果注意力，这可以把计算量最多减少 50%。

## 实现：完整的序列并行层

```python
class SequenceParallelTransformerLayer(nn.Module):
    """Complete transformer layer with sequence parallelism."""

    def __init__(self, config, sp_group, tp_group):
        super().__init__()
        self.sp_group = sp_group
        self.tp_group = tp_group
        self.sp_size = dist.get_world_size(sp_group)

        # Layer components
        self.ln1 = SequenceParallelLayerNorm(config.hidden_size, sp_group)
        self.attention = SequenceParallelAttention(config, sp_group, tp_group)
        self.ln2 = SequenceParallelLayerNorm(config.hidden_size, sp_group)
        self.ffn = SequenceParallelFFN(config, sp_group, tp_group)

    def forward(self, x):
        # x: (batch, seq_local, hidden) - sequence-parallel input

        # Pre-norm attention
        residual = x
        x = self.ln1(x)
        x = self.attention(x)  # Handles AllGather/ReduceScatter internally
        x = residual + x

        # Pre-norm FFN
        residual = x
        x = self.ln2(x)
        x = self.ffn(x)  # Handles AllGather/ReduceScatter internally
        x = residual + x

        return x

class SequenceParallelAttention(nn.Module):
    """Attention with context parallelism (Ring or Ulysses)."""

    def __init__(self, config, sp_group, tp_group, method='ring'):
        super().__init__()
        self.sp_group = sp_group
        self.tp_group = tp_group
        self.method = method
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads

        # Projections (tensor-parallel)
        self.qkv_proj = ColumnParallelLinear(
            config.hidden_size,
            3 * config.hidden_size,
            tp_group
        )
        self.out_proj = RowParallelLinear(
            config.hidden_size,
            config.hidden_size,
            tp_group
        )

    def forward(self, x):
        batch, seq_local, hidden = x.shape

        # Project to Q, K, V
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for attention
        q = q.view(batch, seq_local, self.num_heads, self.head_dim)
        k = k.view(batch, seq_local, self.num_heads, self.head_dim)
        v = v.view(batch, seq_local, self.num_heads, self.head_dim)

        # Context-parallel attention
        if self.method == 'ring':
            output = ring_attention(q, k, v, self.sp_group)
        else:  # ulysses
            output = ulysses_attention(q, k, v, self.sp_group)

        # Reshape and project
        output = output.view(batch, seq_local, hidden)
        output = self.out_proj(output)

        return output
```

## 练习题

1. **内存计算**：对一个 $H = 4096$、$n_h = 32$、$S = 256K$、批 1、bf16 的模型：

   - 计算无序列并行时的注意力矩阵内存
   - 计算 $P = 8$ Ring Attention 时的内存
   - 缩减倍数是多少？

??? success "解答"
    **已知：**

    - 隐藏维度：$H = 4096$
    - 头数：$n_h = 32$
    - 序列长度：$S = 256K = 262,144$
    - 批大小：$b = 1$
    - 数据类型：bf16（2 字节）

    **无序列并行时的注意力矩阵内存：**

    注意力矩阵形状为 $(b, n_h, S, S)$：

    $$M_{\text{attn}} = b \times n_h \times S \times S \times \text{sizeof(bf16)}$$

    $$= 1 \times 32 \times 262144 \times 262144 \times 2 \text{ bytes}$$

    $$= 32 \times 262144^2 \times 2$$

    $$= 32 \times 6.87 \times 10^{10} \times 2$$

    $$= \boxed{4.4 \text{ TB}}$$

    这不可能放进任何单块 GPU。

    **$P = 8$ 的 Ring Attention：**

    每块 GPU 用本地查询对本地 KV 块计算。每 GPU 的注意力矩阵形状为 $(b, n_h, S/P, S/P)$：

    $$M_{\text{attn}}^{\text{Ring}} = b \times n_h \times \frac{S}{P} \times \frac{S}{P} \times 2$$

    $$= 1 \times 32 \times 32768 \times 32768 \times 2$$

    $$= 32 \times 32768^2 \times 2$$

    $$= 32 \times 1.07 \times 10^9 \times 2$$

    $$= \boxed{68.7 \text{ GB}}$$

    这能放进 80GB 的 H100！

    **缩减倍数：**

    $$\text{Reduction} = \frac{M_{\text{attn}}}{M_{\text{attn}}^{\text{Ring}}} = \frac{S^2}{(S/P)^2} = P^2 = 8^2 = \boxed{64\times}$$

    **汇总：**

    | 配置 | 注意力矩阵内存 | 能放进 80GB？ |
    |---------------|------------------------|---------------|
    | 无 SP | 4.4 TB | 否 |
    | 环形（$P=8$） | 68.7 GB | 是 |

    **注意：** Ring Attention 还需要 KV 缓冲（约 $4 \times S/P \times H \times 2 = 2$ GB，双缓冲），轻松放下。

2. **通信量**：对 $S = 1M$、$H = 8192$、$P = 16$，比较 Ring Attention 与 Ulysses 的通信量。哪个带宽用得少？

??? success "解答"
    **已知：**

    - 序列长度：$S = 1M = 1,048,576$
    - 隐藏维度：$H = 8192$
    - 并行度：$P = 16$
    - 假设 bf16（每元素 2 字节）

    **Ring Attention 通信量：**

    Ring Attention 在环上轮转 K、V 张量。每块 GPU 在每一步把自己本地的 K、V 发给下一块 GPU。

    本地 K、V 形状各为 $(S/P, H)$：

    $$\text{KV size per GPU} = 2 \times \frac{S}{P} \times H \times 2 \text{ bytes}$$

    $$= 2 \times \frac{1,048,576}{16} \times 8192 \times 2 = 2 \times 65536 \times 8192 \times 2$$

    $$= 2.15 \text{ GB per step}$$

    在 $P - 1 = 15$ 步内总通信量：

    $$V_{\text{Ring}} = 15 \times 2.15 \text{ GB} = \boxed{32.2 \text{ GB}}$$

    **Ulysses 通信量：**

    Ulysses 执行两次 AlltoAll：
    1. 注意力之前：把 Q、K、V 从 $(S/P, H)$ 重分布到 $(S, H/P)$
    2. 注意力之后：把输出重分布回去

    每次 AlltoAll 会在所有 GPU 之间搬移全部数据。Q、K、V 的每次 AlltoAll 通信量：

    $$V_{\text{AlltoAll}} = 3 \times \frac{S}{P} \times H \times 2 \times \frac{P-1}{P}$$

    $$= 3 \times 65536 \times 8192 \times 2 \times \frac{15}{16}$$

    $$= 3.02 \text{ GB}$$

    输出的 AlltoAll：

    $$V_{\text{out}} = \frac{S}{P} \times H \times 2 \times \frac{P-1}{P} = 1.01 \text{ GB}$$

    Ulysses 总计：

    $$V_{\text{Ulysses}} = V_{\text{AlltoAll}} + V_{\text{out}} = 3.02 + 1.01 = \boxed{4.03 \text{ GB}}$$

    **对比：**

    | 方法 | 总通信量 | 每步延迟 |
    |--------|-------------|------------------|
    | Ring Attention | 32.2 GB | 15 次 P2P 传输 |
    | Ulysses | 4.03 GB | 2 次 AlltoAll 集合通信（collective communication） |

    $$\frac{V_{\text{Ring}}}{V_{\text{Ulysses}}} = \frac{32.2}{4.03} = 8\times$$

    **Ulysses 的总通信量少 8 倍**。不过：

    - **环形**：通信与计算重叠（可隐藏延迟）
    - **Ulysses**：AlltoAll 是阻塞的，但总通信量更低

    **如何选择：**

    - **环形**：计算远大于通信，且 P2P 带宽好时
    - **Ulysses**：P 较小时，且 AlltoAll 很快（节点内 NVLink）

3. **重叠效率**：若每步 Ring 中注意力计算耗时 10 毫秒、KV 传输耗时 2 毫秒，有多少比例的通信被隐藏？在 $P = 8$ 步下，有效通信开销是多少？

??? success "解答"
    **已知：**

    - 注意力计算时间：$T_{\text{compute}} = 10$ 毫秒每步
    - KV 传输时间：$T_{\text{comm}} = 2$ 毫秒每步
    - 步数：$P = 8$

    **Ring Attention 的重叠模型：**

    在 Ring Attention 中，通信通过双缓冲与计算重叠：
    - 用当前 KV 块计算注意力的同时，传输下一份 KV 块
    - 若 $T_{\text{comm}} \leq T_{\text{compute}}$，通信被隐藏

    **被隐藏的通信比例：**

    由于 $T_{\text{comm}} = 2$ 毫秒 $< T_{\text{compute}} = 10$ 毫秒：

    $$\text{Fraction hidden} = \frac{T_{\text{comm}}}{T_{\text{comm}}} = \boxed{100\%}$$

    全部通信都隐藏在计算之后！

    **总执行时间：**

    完全重叠时，总时间由计算主导，再加上一次无法重叠的初始传输：

    $$T_{\text{total}} = T_{\text{initial comm}} + P \times T_{\text{compute}}$$

    $$= 2 + 8 \times 10 = 82 \text{ ms}$$

    无重叠时：

    $$T_{\text{no overlap}} = P \times (T_{\text{compute}} + T_{\text{comm}}) = 8 \times 12 = 96 \text{ ms}$$

    **有效通信开销：**

    $$\text{Overhead} = \frac{T_{\text{total}} - P \times T_{\text{compute}}}{T_{\text{total}}}$$

    $$= \frac{82 - 80}{82} = \frac{2}{82} = \boxed{2.4\%}$$

    **汇总：**

    | 指标 | 数值 |
    |--------|-------|
    | 被隐藏的通信 | 100% |
    | 总时间（有重叠） | 82 ms |
    | 总时间（无重叠） | 96 ms |
    | 重叠带来的加速 | 1.17× |
    | 有效开销 | 2.4% |

    **关键洞见：** 当计算时间超过通信时间时，Ring Attention 能实现近乎完美的重叠。唯一暴露在外的通信是第一计算步之前的初次 KV 获取。

    **重叠失效的情形：**

    若 $T_{\text{comm}} > T_{\text{compute}}$，一部分通信会暴露在外：

    $$\text{Exposed comm} = P \times (T_{\text{comm}} - T_{\text{compute}})$$

4. **因果优化**：对 $P = 8$ 且启用因果掩码的 Ring Attention，平均有多少步（8 步中）因"全未来块"而可被跳过？

??? success "解答"
    **Ring Attention 中的因果掩码：**

    启用因果掩码时，位置 $i$ 只能关注位置 $j \leq i$。当 K、V 块中的 token 相对于所有 Q token 都是未来时，整个计算都可跳过。

    **$P = 8$ 的设定：**

    把序列分成 8 块：$C_0, C_1, \ldots, C_7$

    GPU $r$ 持有 $C_r$ 的查询。经过 $k$ 次轮转后，它收到来自块 $C_{(r-k) \mod 8}$ 的 K、V。

    **什么时候能跳过？**

    当 $C_{(r-k) \mod 8}$ 的所有 token 相对于 $C_r$ 的所有 token 都是未来时，GPU $r$ 可以跳过第 $k$ 步。

    当 $(r - k) \mod 8 > r$ 时，即 KV 块索引大于 Q 块索引时，就会发生。

    **各 GPU 分析：**

    | GPU（Q 块） | 收到的 KV 块 | 可跳过的块 |
    |---------------|-------------------|-----------------|
    | GPU 0（$C_0$） | $C_0 \to C_7 \to C_6 \to \ldots \to C_1$ | $C_1, C_2, \ldots, C_7$（7 个可跳过） |
    | GPU 1（$C_1$） | $C_1 \to C_0 \to C_7 \to \ldots \to C_2$ | $C_2, C_3, \ldots, C_7$（6 个可跳过） |
    | GPU 2（$C_2$） | $C_2 \to C_1 \to C_0 \to \ldots \to C_3$ | $C_3, C_4, \ldots, C_7$（5 个可跳过） |
    | GPU 3（$C_3$） | ... | 4 个可跳过 |
    | GPU 4（$C_4$） | ... | 3 个可跳过 |
    | GPU 5（$C_5$） | ... | 2 个可跳过 |
    | GPU 6（$C_6$） | ... | 1 个可跳过 |
    | GPU 7（$C_7$） | $C_7 \to C_6 \to \ldots \to C_0$ | 0 个可跳过 |

    **每 GPU 平均可跳过步数：**

    $$\text{Avg skippable} = \frac{7 + 6 + 5 + 4 + 3 + 2 + 1 + 0}{8} = \frac{28}{8} = \boxed{3.5 \text{ steps}}$$

    **工作量减少：**

    无优化时：每 GPU $P = 8$ 步
    有因果跳过时：平均 $8 - 3.5 = 4.5$ 步

    $$\text{Speedup} = \frac{8}{4.5} = 1.78\times$$

    **$P$ 块 GPU 的通用公式：**

    平均可跳过步数：

    $$\text{Avg skip} = \frac{\sum_{r=0}^{P-1} (P - 1 - r)}{P} = \frac{(P-1)P/2}{P} = \frac{P-1}{2}$$

    当 $P = 8$ 时：$\frac{8-1}{2} = 3.5$ ✓

    **重要提醒：**

    对角块（GPU $r$ 的 $C_r$）需要部分计算——块内的 token 之间仍有因果关系。这通过在对角块上施加三角掩码来处理。

    | 指标 | 数值 |
    |--------|-------|
    | 无优化时的总步数 | 8 |
    | 平均可跳过步数 | 3.5 |
    | 平均所需步数 | 4.5 |
    | 工作量减少 | 43.75% |
    | 理论加速 | 1.78× |

5. **混合设计**：为 64 块 GPU（8 节点 × 8 GPU）设计一个序列并行策略，$S = 2M$ 个 token。给出分组配置，并估算每 GPU 内存。

??? success "解答"
    **已知：**

    - GPU 总数：64（8 节点 × 每节点 8 GPU）
    - 序列长度：$S = 2M = 2,097,152$ 个 token
    - 假设：$H = 8192$、$n_h = 64$ 头、bf16

    **拓扑考量：**

    - **节点内**：8 块 GPU 通过 NVLink 互联（900 GB/s）
    - **节点间**：GPU 通过 InfiniBand 互联（约 400 Gb/s = 50 GB/s）

    **混合策略：节点内 Ulysses + 节点间环形**

    | 层级 | 并行方式 | 度 | 通信 |
    |-------|-------------|--------|---------------|
    | 节点内 | Ulysses | 8 | 经 NVLink 的快速 AlltoAll |
    | 节点间 | 环形 | 8 | 经 InfiniBand 的 P2P |

    **工作原理：**

    1. 把 64 块 GPU 分成 8 个环形组（每节点一个位置，跨节点）
    2. 每个环形组有跨 8 个节点的 8 块 GPU
    3. 每个节点内部，用 Ulysses 在序列↔头之间重分布
    4. 跨节点之间，用环形轮转 KV 块

    **序列分布：**

    $$S_{\text{per GPU}} = \frac{S}{64} = \frac{2,097,152}{64} = 32,768 \text{ tokens}$$

    **每 GPU 内存估算：**

    *注意力矩阵内存：*

    64 路 SP 下，注意力在 $(S/64) \times (S/64)$ 的块上计算：

    $$M_{\text{attn}} = n_h \times \left(\frac{S}{64}\right)^2 \times 2 = 64 \times 32768^2 \times 2$$

    $$= 64 \times 1.07 \times 10^9 \times 2 = \boxed{137 \text{ GB}}$$

    等一下——这放不进 80GB！我们需要 Flash Attention。

    *使用 Flash Attention（不物化注意力矩阵）：*

    $$M_{\text{Flash}} = O(S/P) = \text{negligible (few MB)}$$

    *每 GPU 的 Q、K、V：*

    $$M_{\text{QKV}} = 3 \times \frac{S}{64} \times H \times 2 = 3 \times 32768 \times 8192 \times 2 = 1.6 \text{ GB}$$

    *环形所需的 KV 缓冲（双缓冲）：*

    $$M_{\text{KV buf}} = 4 \times \frac{S}{64} \times H \times 2 = 2.1 \text{ GB}$$

    **每 GPU 总内存（使用 Flash Attention）：**

    | 组成部分 | 内存 |
    |-----------|--------|
    | 本地 Q、K、V | 1.6 GB |
    | KV 环形缓冲 | 2.1 GB |
    | 输出 | 0.5 GB |
    | Flash 工作区 | ~0.5 GB |
    | **注意力总计** | **~5 GB** |

    这轻松放进 80GB，还为模型权重和激活留出空间。

    **通信分析：**

    *Ulysses（节点内）：*

    - AlltoAll 通信量：$3 \times (S/64) \times H \times 2 \times 7/8 = 1.4$ GB
    - 900 GB/s 下的时间：约 1.5 毫秒

    *环形（节点间）：*

- 每步：$(K+V) = 2 \times (S/8) \times H$ 个元素 → $2 \times (S/8) \times H \times 2 = 8.6$ GB（bf16）
- 50 GB/s 下 7 步：$7 \times 8.6 / 50 \approx 1.2$ 秒

    **替代方案：纯 64 路环形**

    - 63 步 KV 轮转
    - 延迟高得多，但更简单
    - 若计算占主导，可行

    **推荐配置：**

    ```
    Intra-node: Ulysses (P=8)
        - Fast AlltoAll on NVLink
        - Redistributes S/8 → S, H → H/8

    Inter-node: Ring (P=8)
        - Overlapped P2P on InfiniBand
        - Each node rotates as a unit

    Memory per GPU: ~5 GB for attention
    Sequence per GPU: 32K tokens
    Total throughput: 2M token context
    ```

    | 配置 | 优点 | 缺点 |
    |---------------|------|------|
    | 混合 Ulysses+环形 | 兼得两者之长 | 实现复杂 |
    | 纯 Ring-64 | 简单 | 63 步，高延迟 |
    | 纯 Ulysses-64 | 通信量低 | 跨节点 AlltoAll 很慢 |

6. **在线 softmax 验证**：实现并验证：把三个块 $(A, B, C)$ 按 $(A \oplus B) \oplus C$ 组合与按 $A \oplus (B \oplus C)$ 组合，得到的注意力结果相同。

??? success "解答"
    **在线 softmax 的状态表示：**

    对每个块，追踪三元组 $(m, s, o)$：
    - $m$：迄今见过的最大 logit
    - $s$：指数之和（按当前最大值缩放）
    - $o$：加权输出（按当前最大值缩放）

    **组合算子 $\oplus$：**

    给定状态 $(m_1, s_1, o_1)$ 与 $(m_2, s_2, o_2)$：

    $$m_{12} = \max(m_1, m_2)$$

    $$s_{12} = s_1 \cdot e^{m_1 - m_{12}} + s_2 \cdot e^{m_2 - m_{12}}$$

    $$o_{12} = o_1 \cdot e^{m_1 - m_{12}} + o_2 \cdot e^{m_2 - m_{12}}$$

    **实现：**

    ```python
    import torch
    import numpy as np

    def create_chunk_state(logits, values):
        """
        Compute (m, s, o) state for a chunk.

        Args:
            logits: [seq_q, seq_kv] attention logits
            values: [seq_kv, d] value vectors

        Returns:
            m: [seq_q] max logits
            s: [seq_q] sum of exp(logits - m)
            o: [seq_q, d] weighted sum of values
        """
        m = logits.max(dim=-1, keepdim=True).values  # [seq_q, 1]
        exp_logits = torch.exp(logits - m)           # [seq_q, seq_kv]
        s = exp_logits.sum(dim=-1, keepdim=True)     # [seq_q, 1]
        o = exp_logits @ values                       # [seq_q, d]
        return m.squeeze(-1), s.squeeze(-1), o

    def combine_states(state1, state2):
        """
        Combine two (m, s, o) states using the associative operator.
        """
        m1, s1, o1 = state1
        m2, s2, o2 = state2

        m_new = torch.maximum(m1, m2)

        # Correction factors
        alpha1 = torch.exp(m1 - m_new).unsqueeze(-1)
        alpha2 = torch.exp(m2 - m_new).unsqueeze(-1)

        s_new = s1 * alpha1.squeeze(-1) + s2 * alpha2.squeeze(-1)
        o_new = o1 * alpha1 + o2 * alpha2

        return m_new, s_new, o_new

    def finalize_output(state):
        """Convert (m, s, o) state to final attention output."""
        m, s, o = state
        return o / s.unsqueeze(-1)

    # Test associativity
    torch.manual_seed(42)

    seq_q, seq_kv, d = 4, 6, 8

    # Create Q, K, V
    Q = torch.randn(seq_q, d)
    K = torch.randn(seq_kv, d)
    V = torch.randn(seq_kv, d)

    # Split K, V into 3 chunks
    chunk_size = seq_kv // 3
    K_chunks = [K[i*chunk_size:(i+1)*chunk_size] for i in range(3)]
    V_chunks = [V[i*chunk_size:(i+1)*chunk_size] for i in range(3)]

    # Compute logits for each chunk
    logits_A = Q @ K_chunks[0].T
    logits_B = Q @ K_chunks[1].T
    logits_C = Q @ K_chunks[2].T

    # Create states for each chunk
    state_A = create_chunk_state(logits_A, V_chunks[0])
    state_B = create_chunk_state(logits_B, V_chunks[1])
    state_C = create_chunk_state(logits_C, V_chunks[2])

    # Test (A ⊕ B) ⊕ C
    state_AB = combine_states(state_A, state_B)
    state_AB_C = combine_states(state_AB, state_C)
    output_left = finalize_output(state_AB_C)

    # Test A ⊕ (B ⊕ C)
    state_BC = combine_states(state_B, state_C)
    state_A_BC = combine_states(state_A, state_BC)
    output_right = finalize_output(state_A_BC)

    # Ground truth: standard attention
    logits_full = Q @ K.T
    attn_weights = torch.softmax(logits_full, dim=-1)
    output_standard = attn_weights @ V

    # Verify
    print("Max diff (A⊕B)⊕C vs A⊕(B⊕C):",
          (output_left - output_right).abs().max().item())
    print("Max diff vs standard attention:",
          (output_left - output_standard).abs().max().item())
    ```

    **输出：**

    ```
    Max diff (A⊕B)⊕C vs A⊕(B⊕C): 0.0
    Max diff vs standard attention: 1.1920928955078125e-07
    ```

    **验证结果：**

    | 对比 | 最大绝对差 |
    |------------|------------------------|
    | $(A \oplus B) \oplus C$ vs $A \oplus (B \oplus C)$ | $\boxed{0.0}$（精确） |
    | 分块 vs 标准 | ~$10^{-7}$（数值精度） |

    **为什么成立（数学证明）：**

    组合算子满足结合律，因为：

    1. **max 满足结合律**：$\max(\max(a, b), c) = \max(a, \max(b, c))$

    2. **加权和正确组合**：指数重缩放确保和始终以全局最大值为基准归一化。

    3. **Softmax 分解**：对键的任意划分：

       $$\text{softmax}([x_A; x_B; x_C]) = \frac{[e^{x_A}; e^{x_B}; e^{x_C}]}{e^{m_A}s_A + e^{m_B}s_B + e^{m_C}s_C}$$

       这可以通过追踪 $(m, s, o)$ 并做结合式组合来增量计算。

7. **Flash + 环形**：修改 Ring Attention 算法，使其内部使用 Flash Attention。嵌套分块对内存有何影响？

??? success "解答"
    **Ring Attention 内部使用 Flash Attention：**

    Ring Attention 与 Flash Attention 都使用在线 softmax，因此天然可以复合：

    - **环形**：跨 GPU 分块（分布式 KV 块）
    - **Flash**：GPU 内部（SRAM 大小的块）分块

    **嵌套分块结构：**

    ```
    Ring Level (distributed):
        For each KV chunk from ring:
            Flash Level (local):
                For each Q block that fits in SRAM:
                    For each KV block from current chunk:
                        Compute partial attention
                        Update (m, s, o) state
    ```

    **实现：**

    ```python
    import torch
    import torch.distributed as dist

    def flash_attention_forward(q, k, v, block_size=256):
        """
        Flash Attention with online softmax.
        Returns (output, m, lse) for Ring integration.

        Args:
            q: [seq_q, d]
            k: [seq_kv, d]
            v: [seq_kv, d]

        Returns:
            output: [seq_q, d]
            m: [seq_q] max logits per query
            lse: [seq_q] log-sum-exp per query
        """
        seq_q, d = q.shape
        seq_kv = k.shape[0]

        # Initialize online state
        m = torch.full((seq_q,), float('-inf'), device=q.device)
        lse = torch.zeros(seq_q, device=q.device)
        output = torch.zeros(seq_q, d, device=q.device)

        # Process K, V in blocks (simulating SRAM tiling)
        for j in range(0, seq_kv, block_size):
            k_block = k[j:j+block_size]  # [block, d]
            v_block = v[j:j+block_size]  # [block, d]

            # Compute attention scores for this block
            scores = q @ k_block.T  # [seq_q, block]

            # Online softmax update
            m_block = scores.max(dim=-1).values  # [seq_q]
            m_new = torch.maximum(m, m_block)

            # Rescale previous accumulator
            alpha = torch.exp(m - m_new)
            lse = lse * alpha

            # Add new contributions
            exp_scores = torch.exp(scores - m_new.unsqueeze(-1))
            lse = lse + exp_scores.sum(dim=-1)

            # Update output
            output = output * alpha.unsqueeze(-1)
            output = output + exp_scores @ v_block

            m = m_new

        return output, m, lse

    def ring_flash_attention(q_local, k_local, v_local, group):
        """
        Ring Attention using Flash Attention as inner kernel.

        Args:
            q_local: Local Q chunk [seq_local, d]
            k_local: Local K chunk [seq_local, d]
            v_local: Local V chunk [seq_local, d]
            group: Process group for ring communication
        """
        rank = dist.get_rank(group)
        world_size = dist.get_world_size(group)
        device = q_local.device

        seq_local, d = q_local.shape

        # Initialize ring state (using online softmax representation)
        m_global = torch.full((seq_local,), float('-inf'), device=device)
        lse_global = torch.zeros(seq_local, device=device)
        output_global = torch.zeros(seq_local, d, device=device)

        # Double buffering for KV
        k_recv = torch.empty_like(k_local)
        v_recv = torch.empty_like(v_local)

        k_curr, v_curr = k_local, v_local

        for step in range(world_size):
            # Async receive next KV (except last step)
            if step < world_size - 1:
                src = (rank - 1) % world_size
                dst = (rank + 1) % world_size

                recv_k = dist.irecv(k_recv, src=src, group=group)
                recv_v = dist.irecv(v_recv, src=src, group=group)

                send_k = dist.isend(k_curr, dst=dst, group=group)
                send_v = dist.isend(v_curr, dst=dst, group=group)

            # Flash Attention on current KV chunk
            output_chunk, m_chunk, lse_chunk = flash_attention_forward(
                q_local, k_curr, v_curr
            )

            # Combine with global state (online softmax merge)
            m_new = torch.maximum(m_global, m_chunk)

            alpha_global = torch.exp(m_global - m_new)
            alpha_chunk = torch.exp(m_chunk - m_new)

            lse_global = lse_global * alpha_global + lse_chunk * alpha_chunk
            output_global = (output_global * alpha_global.unsqueeze(-1) +
                           output_chunk * alpha_chunk.unsqueeze(-1))

            m_global = m_new

            # Wait for communication and swap buffers
            if step < world_size - 1:
                recv_k.wait()
                recv_v.wait()
                send_k.wait()
                send_v.wait()

                k_curr, k_recv = k_recv, k_curr
                v_curr, v_recv = v_recv, v_curr

        # Final normalization
        output_final = output_global / lse_global.unsqueeze(-1)

        return output_final
    ```

    **内存分析：**

    | 组成部分 | 标准环形 | Flash + 环形 |
    |-----------|--------------|--------------|
    | 注意力矩阵 | 每块 $(S/P)^2$ | 0（从不物化） |
    | 每 GPU 的 Q | $S/P \times d$ | $S/P \times d$ |
    | K、V 缓冲 | $4 \times S/P \times d$ | $4 \times S/P \times d$ |
    | Flash 工作区 | 0 | $O(\text{block\_size} \times d)$ |
    | 输出累加器 | $S/P \times d$ | $S/P \times d$ |

    **嵌套分块的内存影响：**

$$M_{\text{Flash+Ring}} = \underbrace{3 \times \frac{S}{P} \times d}_{\text{Q,K,V local}} + \underbrace{4 \times \frac{S}{P} \times d}_{\text{KV double buffer}} + \underbrace{O(B \times d)}_{\text{Flash SRAM}} + \underbrace{\frac{S}{P} \times d}_{\text{output}}$$

    对 $S = 1M$、$P = 8$、$d = 128$、$B = 256$：

    - 本地 Q、K、V：$3 \times 128K \times 128 \times 2 = 98$ MB
    - KV 缓冲：$2 \times 128K \times 128 \times 2 = 65$ MB
    - Flash 工作区：约 1 MB
    - 输出：$128K \times 128 \times 2 = 33$ MB

    **总计：约 200 MB**（不用 Flash 时为 4+ GB）

    **主要好处：**

    | 好处 | 说明 |
    |---------|-------------|
    | $O(S/P)$ 内存 | 无 $(S/P)^2$ 注意力矩阵 |
    | IO 效率 | Flash 减少 HBM 流量 |
    | 可复合性 | 两者都用在线 softmax |
    | 数值稳定性 | 全程使用 log-sum-exp 技巧 |

    **内存下降：**

    不用 Flash：注意力矩阵为 $M = O((S/P)^2)$
    用 Flash：$M = O(S/P)$，对本地序列线性

    对 $S/P = 128K$：
    - 不用 Flash：$128K^2 \times 2 = 32$ GB（放不下！）
    - 用 Flash：约 200 MB

    $$\text{Reduction} = \frac{(S/P)^2}{S/P \cdot d} = \frac{S/P}{d} = \frac{128K}{128} = \boxed{1024\times}$$

## 旋钮与权衡

| 旋钮 | 主要效果 | 代价 |
|---|---|---|
| 序列并行度 | 减少激活内存 | 更多 AllGather/ReduceScatter |
| 环形 vs AlltoAll（上下文） | 更好的重叠 vs 简单 | 对拓扑敏感 |
| Flash + 分块 | 更低内存与 IO | 更高内核复杂度 |
| 因果掩码捷径 | 更少的块 | 更多边界逻辑 |

## 关键要点

1. **两类序列并行**：

   - Megatron 风格：减少 LayerNorm/Dropout 的激活内存
   - 上下文并行（Context Parallelism）：把注意力计算本身分布出去

2. **在线 softmax 使分解成为可能**：注意力可以用结合式的状态更新增量计算。

3. **Ring Attention**：在环上轮转 K、V；通信与计算重叠。

4. **Ulysses**：用 AlltoAll 在序列与头之间重分布；更简单但重叠更少。

5. **内存按 $O((S/P)^2)$ 扩展**：Ring Attention 把注意力矩阵缩减 $P^2$ 倍。

6. **Flash Attention 集成**：用 Flash 作内层内核以提升内存效率。

7. **因果掩码优化**：跳过所有键都在未来的块。

8. **按拓扑选择**：rank 多且 P2P 好时用环形；rank 少且 AlltoAll 快时用 Ulysses。