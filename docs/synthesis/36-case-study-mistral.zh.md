---
title: "案例研究：Mistral 与 Mixtral"
subtitle: "高效注意力与稀疏专家混合"
---

<div class="chapter-opener" markdown>
Mistral AI 证明了架构效率可以与规模相抗衡。通过滑动窗口注意力、分组查询注意力和稀疏 MoE，他们以远少的资源实现了有竞争力的性能。
</div>

<div class="investigation-question" markdown>
**问题**：Mistral 7B 以一半的参数匹敌 LLaMA 2 13B 的性能。Mixtral 8x7B 只激活 13B 参数却匹敌 LLaMA 2 70B。是什么架构创新实现了这种效率，它们对分布式训练又有何影响？
</div>

!!! tip "引入的新概念"
    本案例研究引入了面向效率的架构创新：

    - **滑动窗口注意力（Sliding Window Attention, SWA）**：把注意力限制在局部 token 窗口内，将内存从 $O(n^2)$ 降到 $O(n \cdot w)$，其中 $w$ 是窗口大小
    - **分组查询注意力（Grouped-Query Attention, GQA）**：在多个查询头之间共享键值头，以极小的质量损失把 KV 缓存减少 4–8 倍
    - **大规模稀疏 MoE**：Mixtral 的 8×7B 和 8×22B 展示了生产级的稀疏专家混合

    这些技术以相乘方式组合——SWA + GQA 合起来可以把 KV 缓存减少 32 倍。

## Mistral 的架构哲学

Mistral AI 的方法反转了典型的扩展叙事。他们不问「我们如何训练更大的模型？」，而问「我们如何最大化每 FLOP 的质量？」这引出了不同的架构选择：

| 模型 | 总参数 | 激活参数 | 性能对标 |
|------|--------|----------|----------|
| Mistral 7B | 7.3B | 7.3B | LLaMA 2 13B |
| Mixtral 8x7B | 46.7B | 12.9B | LLaMA 2 70B |
| Mixtral 8x22B | 141B | 39B | GPT-4 级 |

关键洞察：**稀疏计算实现稠密性能**。

## 滑动窗口注意力

标准注意力具有 $O(n^2)$ 复杂度。对长上下文而言，它主导了计算和内存。

### 局部性假设

对许多任务来说，token 主要关注邻近的上下文。滑动窗口注意力（SWA）正是利用这一点：

$$\text{Attention}(Q, K, V)_{i} = \text{softmax}\left(\frac{Q_i K_{[i-w:i]}^T}{\sqrt{d_k}}\right) V_{[i-w:i]}$$

其中 $w$ 是窗口大小。

### 内存分析

**标准注意力**（序列长度 $n$、批大小 $B$、头数 $h$、维度 $d$）：

$$\text{KV cache} = 2 \cdot B \cdot n \cdot h \cdot d \cdot 2 \text{ bytes}$$

对 $n = 32768$、$B = 1$、$h = 32$、$d = 128$：

$$= 2 \cdot 1 \cdot 32768 \cdot 32 \cdot 128 \cdot 2 = 512 \text{ MB}$$

**滑动窗口注意力**（窗口大小 $w$）：

$$\text{KV cache} = 2 \cdot B \cdot w \cdot h \cdot d \cdot 2 \text{ bytes}$$

对 $w = 4096$：

$$= 2 \cdot 1 \cdot 4096 \cdot 32 \cdot 128 \cdot 2 = 64 \text{ MB}$$

此配置下**内存减少 8 倍**。

### 滚动缓冲区

实现上使用循环缓冲区（circular buffer）：

```python
class SlidingWindowKVCache:
    """
    Rolling KV cache for sliding window attention.

    Memory is fixed at window_size regardless of sequence length.
    """
    def __init__(self, batch_size: int, window_size: int,
                 n_heads: int, head_dim: int, dtype=torch.bfloat16):
        self.window_size = window_size
        # Circular buffer: position i stored at index i % window_size
        self.k_cache = torch.zeros(
            batch_size, n_heads, window_size, head_dim, dtype=dtype
        )
        self.v_cache = torch.zeros(
            batch_size, n_heads, window_size, head_dim, dtype=dtype
        )
        self.position = 0

    def update(self, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update cache with new KV and return valid window."""
        seq_len = k.size(2)

        for i in range(seq_len):
            idx = (self.position + i) % self.window_size
            self.k_cache[:, :, idx, :] = k[:, :, i, :]
            self.v_cache[:, :, idx, :] = v[:, :, i, :]

        self.position += seq_len

        # Return properly ordered window
        return self._get_ordered_cache()

    def _get_ordered_cache(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return cache in correct temporal order."""
        if self.position <= self.window_size:
            return self.k_cache[:, :, :self.position], self.v_cache[:, :, :self.position]

        # Reorder circular buffer
        start = self.position % self.window_size
        indices = [(start + i) % self.window_size for i in range(self.window_size)]
        return self.k_cache[:, :, indices], self.v_cache[:, :, indices]
```

### 堆叠带来的有效上下文

有 $L$ 层、窗口大小 $w$ 时，有效感受野随之增长（上界）：

$$\text{Effective context} \leq L \cdot w$$

对 $L = 32$ 层、$w = 4096$ 的 Mistral 7B：

$$\text{Effective context} \leq 32 \times 4096 = 131072 \text{ tokens}$$

尽管每层只看到 $w$ 个 token，信息仍能通过网络传播。

```
Layer 32: [============================================]
                         ↑
Layer 31: [========================================]
                     ↑
...
Layer 2:  [========]
              ↑
Layer 1:  [====]
            ↑
Input:    [Token at position 0 influences position 131K through layer propagation]
```

### 带滑动窗口的训练

训练期间，我们必须正确地掩蔽注意力：

```python
def sliding_window_mask(seq_len: int, window_size: int) -> torch.Tensor:
    """
    Create attention mask for sliding window.

    Returns mask where mask[i,j] = True if j is in window for query i.
    """
    # Causal: can only attend to previous positions
    causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))

    # Window: can only attend to positions within window
    positions = torch.arange(seq_len)
    window = (positions.unsqueeze(1) - positions.unsqueeze(0)) < window_size

    return causal & window

def sliding_window_attention_forward(
    q: torch.Tensor,  # [batch, heads, seq, dim]
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: int
) -> torch.Tensor:
    """Efficient sliding window attention with FlashAttention."""
    batch, heads, seq_len, dim = q.shape

    # For training: use sparse attention pattern
    # FlashAttention 2 supports this natively
    if seq_len <= window_size:
        # Standard attention for short sequences
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)

    # Block-sparse attention for long sequences
    # Implementation depends on backend (FlashAttention, xFormers, etc.)
    mask = sliding_window_mask(seq_len, window_size)

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(dim)
    scores = scores.masked_fill(~mask, float('-inf'))
    attn = F.softmax(scores, dim=-1)

    return torch.matmul(attn, v)
```

## 分组查询注意力

分组查询注意力（GQA）通过在查询头之间共享键值头来减少 KV 缓存。

### GQA 的设计空间

| 变体 | 查询头 | KV 头 | 内存比例 |
|------|--------|-------|----------|
| 多头（MHA） | $h$ | $h$ | 1.0 |
| 多查询（MQA） | $h$ | 1 | $1/h$ |
| 分组查询（GQA） | $h$ | $g$ | $g/h$ |

Mistral 7B 使用 $h = 32$ 个查询头和 $g = 8$ 个 KV 组：

$$\text{Memory ratio} = \frac{8}{32} = 0.25$$

**仅 GQA 就带来 4 倍的 KV 缓存缩减**。

### 组合节省

SWA + GQA 合在一起：

$$\text{Total reduction} = \frac{w}{n} \cdot \frac{g}{h}$$

对 $n = 32768$、$w = 4096$、$h = 32$、$g = 8$：

$$= \frac{4096}{32768} \cdot \frac{8}{32} = \frac{1}{8} \cdot \frac{1}{4} = \frac{1}{32}$$

KV 缓存**总共缩减 32 倍**。

### 实现

```python
class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention with Sliding Window support.

    Combines GQA for memory efficiency with SWA for long contexts.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        window_size: int,
        head_dim: Optional[int] = None
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim or d_model // n_heads
        self.window_size = window_size

        # Query heads: full count
        self.W_q = nn.Linear(d_model, n_heads * self.head_dim, bias=False)

        # KV heads: reduced count
        self.W_k = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.W_v = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)

        self.W_o = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

        # Repeat factor for broadcasting KV heads to query heads
        self.n_rep = n_heads // n_kv_heads

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: Optional[SlidingWindowKVCache] = None
    ) -> Tuple[torch.Tensor, Optional[SlidingWindowKVCache]]:
        batch, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.W_q(x).view(batch, seq_len, self.n_heads, self.head_dim)
        k = self.W_k(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)
        v = self.W_v(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)

        # Transpose to [batch, heads, seq, dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply RoPE
        q, k = apply_rotary_embedding(q, k, positions)

        # Handle KV cache for inference
        if kv_cache is not None:
            k, v = kv_cache.update(k, v)

        # Expand KV heads to match query heads
        # [batch, n_kv_heads, seq, dim] -> [batch, n_heads, seq, dim]
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        # Compute attention with sliding window
        output = sliding_window_attention_forward(q, k, v, self.window_size)

        # Reshape and project output
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.W_o(output), kv_cache

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Repeat KV heads to match query heads."""
        batch, n_kv_heads, seq_len, head_dim = x.shape

        # [batch, n_kv, seq, dim] -> [batch, n_kv, n_rep, seq, dim]
        x = x.unsqueeze(2).expand(-1, -1, self.n_rep, -1, -1)

        # [batch, n_kv * n_rep, seq, dim] = [batch, n_heads, seq, dim]
        return x.reshape(batch, self.n_heads, seq_len, head_dim)
```

## Mixtral：稀疏专家混合

Mixtral 在 Mistral 的基础上加入稀疏 MoE，每个 token 只激活 8 个专家中的 2 个。

### 架构对比

| 组件 | Mistral 7B | Mixtral 8x7B |
|------|-----------|--------------|
| 层数 | 32 | 32 |
| 隐层维度 | 4096 | 4096 |
| 头数 | 32 | 32 |
| KV 头数 | 8 | 8 |
| FFN | 稠密 14336 | 8 个专家 × 14336 |
| 激活 FFN | 14336 | 2 × 14336 |
| 总参数 | 7.3B | 46.7B |
| 激活参数 | 7.3B | 12.9B |

### 路由机制

Mixtral 使用一个可学习的路由器做 top-2 选择：

```python
class MixtralRouter(nn.Module):
    """
    Expert router for Mixtral-style sparse MoE.

    Selects top-k experts per token with learned routing weights.
    """
    def __init__(
        self,
        d_model: int,
        n_experts: int,
        top_k: int = 2,
        routing_noise: float = 0.0
    ):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.routing_noise = routing_noise

        # Linear router: d_model -> n_experts
        self.gate = nn.Linear(d_model, n_experts, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route tokens to experts.

        Returns:
            expert_weights: [batch, seq, top_k] - normalized routing weights
            expert_indices: [batch, seq, top_k] - selected expert indices
            router_logits: [batch, seq, n_experts] - raw router outputs
        """
        batch, seq_len, _ = x.shape

        # Compute router logits
        router_logits = self.gate(x)  # [batch, seq, n_experts]

        # Add noise during training for exploration
        if self.training and self.routing_noise > 0:
            noise = torch.randn_like(router_logits) * self.routing_noise
            router_logits = router_logits + noise

        # Select top-k experts
        routing_weights = F.softmax(router_logits, dim=-1)
        expert_weights, expert_indices = torch.topk(
            routing_weights, self.top_k, dim=-1
        )

        # Renormalize weights to sum to 1
        expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)

        return expert_weights, expert_indices, router_logits
```

### 负载均衡

没有均衡机制时，路由可能坍缩为总是选择同样的专家。Mixtral 使用辅助损失：

```python
def load_balancing_loss(
    router_logits: torch.Tensor,
    expert_indices: torch.Tensor,
    n_experts: int,
    top_k: int
) -> torch.Tensor:
    """
    Auxiliary loss to encourage balanced expert usage.

    Combines:
    1. Load balancing: all experts should receive similar token counts
    2. Router z-loss: prevent router logits from growing too large
    """
    batch, seq_len, _ = router_logits.shape
    n_tokens = batch * seq_len

    # Compute routing probabilities
    routing_probs = F.softmax(router_logits, dim=-1)  # [batch, seq, n_experts]
    routing_probs = routing_probs.view(-1, n_experts)  # [tokens, n_experts]

    # Expert selection frequency (fraction of tokens routed to each expert)
    expert_mask = F.one_hot(expert_indices, n_experts).float()  # [batch, seq, top_k, n_experts]
    expert_mask = expert_mask.sum(dim=2)  # [batch, seq, n_experts]
    expert_mask = expert_mask.view(-1, n_experts)  # [tokens, n_experts]

    tokens_per_expert = expert_mask.sum(dim=0)  # [n_experts]
    fraction_per_expert = tokens_per_expert / (n_tokens * top_k)

    # Mean routing probability to each expert
    mean_routing_prob = routing_probs.mean(dim=0)  # [n_experts]

    # Load balancing loss: minimize product of fraction and probability
    # Ideally both should be uniform (1/n_experts)
    load_balance_loss = n_experts * (fraction_per_expert * mean_routing_prob).sum()

    # Router z-loss: regularize router logits
    z_loss = torch.logsumexp(router_logits, dim=-1).square().mean()

    return load_balance_loss + 0.001 * z_loss
```

### MoE 块的实现

```python
class MixtralMoEBlock(nn.Module):
    """
    Mixtral Mixture-of-Experts block.

    Replaces the dense FFN with sparse expert routing.
    """
    def __init__(
        self,
        d_model: int,
        ffn_dim: int,
        n_experts: int = 8,
        top_k: int = 2
    ):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k

        # Router
        self.router = MixtralRouter(d_model, n_experts, top_k)

        # Expert FFNs (each is a standard SwiGLU FFN)
        self.experts = nn.ModuleList([
            SwiGLUFFN(d_model, ffn_dim)
            for _ in range(n_experts)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with expert routing.

        Returns:
            output: [batch, seq, d_model]
            aux_loss: scalar load balancing loss
        """
        batch, seq_len, d_model = x.shape

        # Route tokens to experts
        expert_weights, expert_indices, router_logits = self.router(x)

        # Compute auxiliary loss
        aux_loss = load_balancing_loss(
            router_logits, expert_indices, self.n_experts, self.top_k
        )

        # Process through experts
        # Naive implementation: loop over experts
        output = torch.zeros_like(x)

        for expert_idx in range(self.n_experts):
            # Find tokens routed to this expert
            for k in range(self.top_k):
                mask = (expert_indices[:, :, k] == expert_idx)
                if not mask.any():
                    continue

                # Get tokens for this expert
                expert_input = x[mask]  # [n_tokens, d_model]

                # Process through expert
                expert_output = self.experts[expert_idx](expert_input)

                # Weight by routing weight and accumulate
                weights = expert_weights[:, :, k][mask].unsqueeze(-1)
                output[mask] += weights * expert_output

        return output, aux_loss

class SwiGLUFFN(nn.Module):
    """SwiGLU FFN as used in Mistral/Mixtral."""
    def __init__(self, d_model: int, ffn_dim: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, ffn_dim, bias=False)
        self.w2 = nn.Linear(ffn_dim, d_model, bias=False)
        self.w3 = nn.Linear(d_model, ffn_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
```

## 用 Megablocks 实现高效 MoE

朴素基于循环的实现效率很低。Megablocks 提供了高效的批量专家执行：

### Token 到专家的置换

```python
class EfficientMoE(nn.Module):
    """
    Efficient MoE using token permutation and batched matrix multiply.

    Key insight: Group tokens by expert, process in batches, then unpermute.
    """
    def __init__(
        self,
        d_model: int,
        ffn_dim: int,
        n_experts: int,
        top_k: int
    ):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k

        self.router = MixtralRouter(d_model, n_experts, top_k)

        # Experts as fused parameter tensors for efficient batching
        # w1 and w3 combined for gating
        self.w13 = nn.Parameter(torch.randn(n_experts, d_model, 2 * ffn_dim))
        self.w2 = nn.Parameter(torch.randn(n_experts, ffn_dim, d_model))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, d_model = x.shape

        # Route tokens
        expert_weights, expert_indices, router_logits = self.router(x)

        # Flatten batch and sequence dimensions
        x_flat = x.view(-1, d_model)  # [batch*seq, d_model]
        expert_weights_flat = expert_weights.view(-1, self.top_k)
        expert_indices_flat = expert_indices.view(-1, self.top_k)

        # Build permutation indices
        # Group tokens by their assigned experts
        sorted_token_ids, expert_counts = self._compute_permutation(expert_indices_flat)

        # Permute tokens by expert assignment
        x_permuted = x_flat[sorted_token_ids]  # Tokens grouped by expert

        # Compute expert outputs using batched matrix multiply
        expert_outputs = self._batched_expert_forward(x_permuted, expert_counts)

        # Unpermute back to original order
        output_flat = torch.zeros_like(x_flat)
        output_flat = self._weighted_accumulate(
            expert_outputs, sorted_token_ids, expert_weights_flat, expert_indices_flat
        )

        output = output_flat.view(batch, seq_len, d_model)
        aux_loss = load_balancing_loss(
            router_logits, expert_indices, self.n_experts, self.top_k
        )

        return output, aux_loss

    def _compute_permutation(
        self, expert_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute permutation to group tokens by expert."""
        n_tokens = expert_indices.size(0)

        # Flatten expert selections
        flat_indices = expert_indices.view(-1)  # [n_tokens * top_k]
        token_ids = torch.arange(n_tokens, device=expert_indices.device).repeat_interleave(self.top_k)

        # Sort by expert index
        _, perm = flat_indices.sort(stable=True)
        sorted_token_ids = token_ids[perm]

        # Count tokens per expert
        expert_counts = torch.bincount(flat_indices, minlength=self.n_experts)

        return sorted_token_ids, expert_counts

    def _batched_expert_forward(
        self, x: torch.Tensor, expert_counts: torch.Tensor
    ) -> torch.Tensor:
        """Process all tokens through their assigned experts efficiently."""
        outputs = []
        offset = 0

        for expert_idx, count in enumerate(expert_counts.tolist()):
            if count == 0:
                continue

            # Get tokens for this expert
            expert_tokens = x[offset:offset + count]  # [count, d_model]

            # SwiGLU forward: w2(silu(w1(x)) * w3(x))
            w13 = self.w13[expert_idx]  # [d_model, 2*ffn_dim]
            w2 = self.w2[expert_idx]    # [ffn_dim, d_model]

            # Combined w1/w3 projection
            gate_up = expert_tokens @ w13  # [count, 2*ffn_dim]
            gate, up = gate_up.chunk(2, dim=-1)

            # SwiGLU activation
            hidden = F.silu(gate) * up  # [count, ffn_dim]

            # Output projection
            output = hidden @ w2  # [count, d_model]
            outputs.append(output)

            offset += count

        return torch.cat(outputs, dim=0) if outputs else torch.empty(0, x.size(-1))
```

### 分布式 MoE：专家并行

对 Mixtral 规模的模型，专家可以分布到多块 GPU 上：

```python
class DistributedMoE(nn.Module):
    """
    MoE with expert parallelism.

    Each GPU holds a subset of experts. Tokens are routed via all-to-all.
    """
    def __init__(
        self,
        d_model: int,
        ffn_dim: int,
        n_experts: int,
        top_k: int,
        expert_parallel_group: torch.distributed.ProcessGroup
    ):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.ep_group = expert_parallel_group
        self.ep_size = dist.get_world_size(expert_parallel_group)
        self.ep_rank = dist.get_rank(expert_parallel_group)

        # Each GPU holds n_experts // ep_size experts
        self.local_n_experts = n_experts // self.ep_size
        self.expert_start = self.ep_rank * self.local_n_experts

        # Router (replicated on all GPUs)
        self.router = MixtralRouter(d_model, n_experts, top_k)

        # Local experts only
        self.local_experts = nn.ModuleList([
            SwiGLUFFN(d_model, ffn_dim)
            for _ in range(self.local_n_experts)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, d_model = x.shape

        # Route tokens (same routing on all GPUs)
        expert_weights, expert_indices, router_logits = self.router(x)

        # Determine which tokens go to which GPU
        # Each GPU processes experts [expert_start, expert_start + local_n_experts)
        tokens_to_send, send_counts = self._prepare_all_to_all(
            x, expert_indices, expert_weights
        )

        # All-to-all: exchange tokens between GPUs
        tokens_received, recv_counts = self._all_to_all_tokens(
            tokens_to_send, send_counts
        )

        # Process tokens through local experts
        local_outputs = self._process_local_experts(tokens_received, recv_counts)

        # All-to-all: send results back
        outputs = self._all_to_all_results(local_outputs, recv_counts, send_counts)

        # Reconstruct output tensor
        output = self._reconstruct_output(outputs, expert_weights, expert_indices)

        aux_loss = load_balancing_loss(
            router_logits, expert_indices, self.n_experts, self.top_k
        )

        return output, aux_loss

    def _all_to_all_tokens(
        self, tokens: torch.Tensor, send_counts: List[int]
    ) -> Tuple[torch.Tensor, List[int]]:
        """Exchange tokens between GPUs via all-to-all."""
        # Gather receive counts from all GPUs
        all_counts = [torch.tensor(send_counts, device=tokens.device)
                      for _ in range(self.ep_size)]
        dist.all_gather(all_counts, torch.tensor(send_counts, device=tokens.device),
                       group=self.ep_group)

        recv_counts = [c[self.ep_rank].item() for c in all_counts]

        # Prepare send/recv buffers
        send_splits = send_counts
        recv_splits = recv_counts

        output = torch.empty(sum(recv_counts), tokens.size(-1),
                            device=tokens.device, dtype=tokens.dtype)

        # All-to-all
        dist.all_to_all_single(output, tokens,
                               output_split_sizes=recv_splits,
                               input_split_sizes=send_splits,
                               group=self.ep_group)

        return output, recv_counts
```

## 分布式训练分析

### 每 GPU 内存

对带专家并行的 Mixtral 8x7B 训练：

```python
class MixtralMemoryAnalyzer:
    """
    Memory analysis for Mixtral distributed training.
    """
    def __init__(self):
        # Architecture
        self.layers = 32
        self.d_model = 4096
        self.n_heads = 32
        self.n_kv_heads = 8
        self.ffn_dim = 14336
        self.n_experts = 8
        self.vocab_size = 32000

        # Parallelism config
        self.tp = 1  # Tensor parallel
        self.ep = 8  # Expert parallel
        self.dp = 8  # Data parallel (with FSDP)
        self.total_gpus = self.tp * self.ep * self.dp

    def parameter_count(self) -> Dict[str, int]:
        """Count parameters by component."""
        # Attention per layer
        head_dim = self.d_model // self.n_heads
        q_params = self.d_model * self.n_heads * head_dim
        kv_params = 2 * self.d_model * self.n_kv_heads * head_dim
        o_params = self.n_heads * head_dim * self.d_model
        attn_per_layer = q_params + kv_params + o_params

        # MoE FFN per layer
        expert_params = 3 * self.d_model * self.ffn_dim  # w1, w2, w3
        moe_per_layer = self.n_experts * expert_params
        router_per_layer = self.d_model * self.n_experts

        # Layer norms
        norm_per_layer = 2 * self.d_model

        # Embeddings
        embed_params = self.vocab_size * self.d_model

        return {
            'attention': self.layers * attn_per_layer,
            'moe_experts': self.layers * moe_per_layer,
            'router': self.layers * router_per_layer,
            'norms': self.layers * norm_per_layer,
            'embeddings': 2 * embed_params,
            'total': (self.layers * (attn_per_layer + moe_per_layer +
                     router_per_layer + norm_per_layer) + 2 * embed_params)
        }

    def memory_per_gpu(self, batch_size: int, seq_len: int) -> Dict[str, float]:
        """Compute memory usage per GPU in GB."""
        params = self.parameter_count()

        # Parameters sharded across DP (FSDP)
        # Experts sharded across EP
        expert_params_local = params['moe_experts'] / self.ep
        other_params_local = (params['attention'] + params['router'] +
                             params['norms'] + params['embeddings']) / self.dp

        total_params_local = expert_params_local + other_params_local

        # Model states (FP32 optimizer)
        param_bytes = total_params_local * 2  # BF16
        optimizer_bytes = total_params_local * 8  # FP32 moments
        grad_bytes = total_params_local * 2  # BF16

        # Activations (per layer, rough estimate)
        # Attention: 2 * batch * seq * d_model (Q, K/V)
        # MoE: batch * seq * d_model * 2 (input, output)
        act_per_layer = batch_size * seq_len * self.d_model * 4 * 2  # BF16
        activation_bytes = self.layers * act_per_layer

        return {
            'parameters': param_bytes / 1e9,
            'optimizer': optimizer_bytes / 1e9,
            'gradients': grad_bytes / 1e9,
            'activations': activation_bytes / 1e9,
            'total': (param_bytes + optimizer_bytes + grad_bytes +
                     activation_bytes) / 1e9
        }

    def communication_volume(self, batch_size: int, seq_len: int) -> Dict[str, float]:
        """Estimate communication volume per step in GB."""
        n_tokens = batch_size * seq_len

        # Expert routing: all-to-all per layer
        # Send/recv tokens to/from each expert partition
        tokens_moved = n_tokens * self.d_model * 2 * 2  # Input + output, bytes
        all_to_all_per_layer = tokens_moved / self.ep  # Per GPU

        # Gradient sync: FSDP AllGather and ReduceScatter
        params = self.parameter_count()
        non_expert_params = (params['attention'] + params['router'] +
                            params['norms'] + params['embeddings'])
        fsdp_volume = 2 * non_expert_params * 2  # AG + RS, BF16

        return {
            'all_to_all_per_layer_gb': all_to_all_per_layer / 1e9,
            'all_to_all_total_gb': self.layers * all_to_all_per_layer / 1e9,
            'fsdp_volume_gb': fsdp_volume / 1e9,
            'total_gb': (self.layers * all_to_all_per_layer + fsdp_volume) / 1e9
        }
```

### 并行策略

Mixtral 采用混合方法：

| 维度 | 策略 | 理由 |
|------|------|------|
| 张量并行 | TP=1 或 2 | 注意力很小，TP 开销不值得 |
| 专家并行 | EP=8 | 每 GPU 一个专家，无复制 |
| 数据并行 | FSDP | 在副本间分片非专家参数 |
| 序列 | 不使用 | 窗口注意力限制了序列成本 |

### 通信模式

```
Per forward pass (one micro-batch):
├── Layer 1
│   ├── Attention: local compute
│   └── MoE: all-to-all (tokens to experts)
├── Layer 2
│   ├── Attention: local compute
│   └── MoE: all-to-all
├── ...
└── Layer 32
    ├── Attention: local compute
    └── MoE: all-to-all

Backward pass: same pattern in reverse

After backward:
└── FSDP gradient sync (ReduceScatter + AllGather)
```

## 推理优化

Mistral/Mixtral 在推理效率上表现出色。

### KV 缓存预算

在内存约束 $M$ 下服务时：

$$\text{Max batch size} = \frac{M - M_{\text{model}}}{\text{KV per sequence}}$$

采用 SWA 与 GQA 时：

$$\text{KV per sequence} = 2 \cdot L \cdot w \cdot g \cdot d \cdot 2$$

对 Mistral 7B（$L=32$、$w=4096$、$g=8$、$d=128$）：

$$= 2 \cdot 32 \cdot 4096 \cdot 8 \cdot 128 \cdot 2 = 512 \text{ MB}$$

对比无 SWA/GQA 的 LLaMA 2 7B 在 32K 上下文时：

$$= 2 \cdot 32 \cdot 32768 \cdot 32 \cdot 128 \cdot 2 = 16 \text{ GB}$$

Mistral 的架构**可以批处理 32 倍多的序列**。

### 投机解码

Mistral 的效率让带草稿模型的投机解码（speculative decoding）变得可行：

```python
class SpeculativeDecoder:
    """
    Speculative decoding using small draft model.

    Draft model proposes K tokens, target model verifies in parallel.
    """
    def __init__(
        self,
        target_model: nn.Module,  # e.g., Mixtral 8x7B
        draft_model: nn.Module,   # e.g., Mistral 7B
        k: int = 4
    ):
        self.target = target_model
        self.draft = draft_model
        self.k = k

    def generate_step(
        self,
        input_ids: torch.Tensor,
        target_cache: Optional[Any],
        draft_cache: Optional[Any]
    ) -> Tuple[torch.Tensor, int]:
        """
        Generate tokens with speculative decoding.

        Returns:
            new_tokens: accepted tokens
            n_accepted: number of tokens accepted
        """
        # Draft model proposes K tokens autoregressively
        draft_tokens = []
        draft_probs = []

        x = input_ids
        for _ in range(self.k):
            logits = self.draft(x, kv_cache=draft_cache)
            probs = F.softmax(logits[:, -1], dim=-1)
            token = torch.multinomial(probs, 1)

            draft_tokens.append(token)
            draft_probs.append(probs.gather(-1, token))
            x = token

        draft_tokens = torch.cat(draft_tokens, dim=1)  # [batch, k]

        # Target model verifies all K tokens in parallel
        full_input = torch.cat([input_ids, draft_tokens], dim=1)
        target_logits = self.target(full_input, kv_cache=target_cache)

        # Compare probabilities and accept/reject
        n_accepted = 0
        for i in range(self.k):
            target_probs = F.softmax(target_logits[:, -(self.k - i)], dim=-1)
            draft_prob = draft_probs[i]
            target_prob = target_probs.gather(-1, draft_tokens[:, i:i+1])

            # Acceptance probability
            accept_prob = min(1.0, target_prob / draft_prob)

            if torch.rand(1) < accept_prob:
                n_accepted += 1
            else:
                # Reject this and all subsequent tokens
                break

        # Return accepted tokens
        return draft_tokens[:, :n_accepted], n_accepted
```

## 训练效率分析

### 计算效率：稀疏 vs 稠密

对 8 专家、top-2 的 Mixtral：

$$\text{Active FLOPs ratio} = \frac{2}{8} = 0.25$$

但专家计算只是模型的一部分：

```python
def compute_active_ratio(
    n_experts: int,
    top_k: int,
    d_model: int,
    ffn_dim: int,
    n_heads: int,
    n_kv_heads: int
) -> float:
    """Compute ratio of active to total FLOPs."""
    head_dim = d_model // n_heads

    # Attention FLOPs per token (fully activated)
    qkv_flops = 2 * d_model * (n_heads + 2 * n_kv_heads) * head_dim
    attn_flops = 2 * d_model * d_model  # Roughly
    attn_total = qkv_flops + attn_flops

    # Expert FLOPs per token
    expert_flops = 3 * 2 * d_model * ffn_dim  # w1, w2, w3
    active_expert_flops = top_k * expert_flops
    total_expert_flops = n_experts * expert_flops

    # Active ratio
    active = attn_total + active_expert_flops
    total = attn_total + total_expert_flops

    return active / total

# Mixtral 8x7B
ratio = compute_active_ratio(
    n_experts=8, top_k=2,
    d_model=4096, ffn_dim=14336,
    n_heads=32, n_kv_heads=8
)
# Returns ~0.31 (31% of total FLOPs activated)
```

### 训练速度

Mixtral 8x7B 处理 token **比 Mistral 7B 慢**，尽管总参数多 6.5 倍：

$$\text{Speed ratio} \approx \frac{\text{Active params}}{\text{Mistral params}} = \frac{12.9B}{7.3B} \approx 1.8\times$$

每个 token 的训练比 Mistral 7B 慢约 1.8 倍，但达到了 LLaMA 2 70B 的质量。

## 练习题

1. **窗口大小选择**：推导窗口大小 $w$、层数 $L$ 与有效上下文长度之间的关系。若要在 32 层下获得 100K 有效上下文，需要多大的窗口？

??? success "解答"
    **练习 1：窗口大小选择**

    **滑动窗口注意力（SWA）的性质：**

    窗口大小为 $w$、层数为 $L$ 时，信息逐层传播：
    - 第 1 层：每个 token 看到 $w$ 个 token
    - 第 2 层：每个 token 看到各自看过 $w$ 个 token 的 token → 有效 $2w$
    - 第 $L$ 层：有效上下文 $\leq L \times w$

    **有效上下文长度：**
    $$C_{eff} = L \times w$$

    **对 $L=32$ 层获得 100K 有效上下文：**
    $$w = \frac{C_{eff}}{L} = \frac{100,000}{32} = 3,125$$

    为效率起见，向上取整到 2 的幂：

    $$\boxed{w = 4096 \text{ tokens}}$$

    这给出 $32 \times 4096 = 131K$ token 的有效上下文。

    **用 Mistral 的设计验证：**
    - Mistral 7B：$w = 4096$、$L = 32$ → 131K 有效上下文
    - 每位置内存：$O(w)$ 而非 $O(S)$

    **权衡表：**

    | 窗口大小 | 层数 | 有效上下文 | KV 缓存（相对） |
    |----------|------|------------|-----------------|
    | 2048 | 32 | 65K | 1× |
    | 4096 | 32 | 131K | 2× |
    | 8192 | 32 | 262K | 4× |
    | 4096 | 64 | 262K | 2×（更多层） |

2. **GQA 内存分析**：比较 (a) 32 头的 MHA、(b) 1 个 KV 头的 MQA、(c) 8 个 KV 组的 GQA 的 KV 缓存内存。哪一种提供最佳的质量/内存权衡？

??? success "解答"
    **练习 2：GQA 内存分析**

    **设定：**
    - 隐层维度：H = 4096
    - 每头维度：$d_h$ = 128
    - 查询头：$n_q$ = 32
    - 序列长度：S = 8192

    **每 token 的 KV 缓存（FP16）：**
    $$M_{KV} = 2 \times n_{kv} \times d_h \times 2 \text{ bytes}$$

    **(a) MHA（多头注意力）：$n_{kv}$ = 32**
    $$M_{MHA} = 2 \times 32 \times 128 \times 2 = 16,384 \text{ bytes/token}$$

    对 S=8192：$16,384 \times 8192 = 134$ MB 每层

    **(b) MQA（多查询注意力）：$n_{kv}$ = 1**
    $$M_{MQA} = 2 \times 1 \times 128 \times 2 = 512 \text{ bytes/token}$$

    对 S=8192：$512 \times 8192 = 4.2$ MB 每层

    缩减：32×

    **(c) GQA（分组查询注意力）：$n_{kv}$ = 8**
    $$M_{GQA} = 2 \times 8 \times 128 \times 2 = 4,096 \text{ bytes/token}$$

    对 S=8192：$4,096 \times 8192 = 33.5$ MB 每层

    缩减：4×

    **质量对比（经验）：**

    | 方法 | 内存 | 质量（相对 MHA） | 最佳平衡点 |
    |------|------|------------------|------------|
    | MHA | 100% | 100% | 小模型 |
    | MQA | 3.1% | 95–97% | 极致压缩 |
    | GQA-8 | 25% | 99%+ | ✓ 最佳权衡 |
    | GQA-4 | 12.5% | 98–99% | 良好权衡 |

    $$\boxed{\text{GQA with 8 groups: 4× memory reduction with <1\% quality loss}}$$

3. **MoE 计算效率**：对一个 16 专家、top-2 路由的模型，有多大比例的专家 FLOPs 被激活？按 Chinchilla 扩展律，这如何改变最优训练数据预算？

??? success "解答"
    **练习 3：MoE 计算效率**

    **Mixtral 配置：**
    - 专家总数：$E = 16$
    - 每 token 激活专家：$k = 2$
    - 专家规模：与稠密 FFN 相同

    **激活比例：**
    $$\text{Active fraction} = \frac{k}{E} = \frac{2}{16} = 12.5\%$$

    **每 token 有效 FLOPs：**

    设稠密模型 FFN FLOPs 为 $F_{FFN}$：

    $$F_{MoE} = \frac{k}{E} \times E \times F_{FFN} = k \times F_{FFN}$$

    等等——每个 token 使用 $k$ 个完整专家，所以：

    $$F_{MoE} = k \times F_{FFN} = 2 \times F_{FFN}$$

    但模型总共有 $E$ 个专家的参数。

    **参数效率：**
    $$\frac{\text{Active params}}{\text{Total params}} = \frac{k}{E} = 12.5\%$$

    **Chinchilla 扩展律调整：**

    Chinchilla：$D_{opt} = 20N$（最优 token 数 = 20× 参数数）

    对 MoE，质量上的有效参数 ≈ 激活参数 × 缩放因子：

    $$N_{eff} \approx N_{active} \times \alpha$$

    其中 $\alpha \approx 2-3$（路由专门化带来的 MoE 质量乘数）。

    对 Mixtral 8x7B：
    - 总参数：$8 \times 7B + \text{shared} = 46.7B$
    - 激活参数：$\approx 12.9B$
    - 质量等效：$\approx 25-40B$

    **最优训练预算：**
    $$D_{opt,MoE} = 20 \times N_{active} \times \alpha \approx 20 \times 12.9B \times 2.5 = 645B \text{ tokens}$$

    Mistral 用了更多 token 训练（估计 1–2T），因为：
    1. 推理成本低（激活 12.9B）
    2. 更多 token 持续提升质量

    $$\boxed{\text{12.5\% activation → can train on 8× more tokens at same FLOP budget}}$$

4. **专家并行**：你有 64 块 GPU，想训练 Mixtral 8x22B（8 专家）。设计一个考虑 TP、EP、DP 维度的并行策略。有哪些权衡？

??? success "解答"
    **练习 4：专家并行策略**

    **设定：**
    - 64 块 GPU（8 节点 × 8 GPU）
    - Mixtral 8x22B：8 专家，每个约 22B 参数
    - 总参数：约 141B

    **每专家内存：**
    - 参数：$22B \times 2 = 44$ GB（FP16）
    - 优化器（Adam）：$22B \times 12 = 264$ GB（FP32）
    - 每专家静态总量：约 308 GB

    一块 GPU 装不下一个专家！需要分片。

    **策略 1：TP=4, EP=8, DP=2**

    ```
    - Tensor Parallel: 4 GPUs share each expert (within node)
    - Expert Parallel: 8 expert groups (one per TP group)
    - Data Parallel: 2 replicas

    GPU Layout (8 nodes × 8 GPUs = 64):
    Node 0: Expert 0 (TP=4) + Expert 0 replica (TP=4)
    Node 1: Expert 1 (TP=4) + Expert 1 replica (TP=4)
    ...
    ```

    每 GPU 内存：$\frac{44}{4} + \frac{264}{4 \times 2} = 11 + 33 = 44$ GB ✓

    **策略 2：TP=8, EP=8, DP=1**

    ```
    - Full node for each expert (TP=8)
    - Each node handles one expert
    - No DP (batch parallelism only)
    ```

    每 GPU 内存：$\frac{44}{8} + \frac{264}{8} = 5.5 + 33 = 38.5$ GB ✓

    **权衡分析：**

    | 策略 | TP | EP | DP | 通信量 | 批大小 |
    |------|----|----|----|--------|--------|
    | 策略 1 | 4 | 8 | 2 | 更低的 TP 通信 | 大 2× |
    | 策略 2 | 8 | 8 | 1 | 更高的 TP 通信 | 受限 |

    **专家路由的 All-to-All 通信：**
    $$V_{A2A} = 2 \times B \times S \times H = 2 \times B \times 4096 \times 6144$$

    有 DP=2 时，每个副本处理一半批 → A2A 通信量更低。

    $$\boxed{\text{Strategy 1 (TP=4, EP=8, DP=2) balances memory and throughput}}$$

5. **负载均衡**：实现一个带与不带辅助损失的 token 路由模拟。测量 1000 步后专家负载不均衡度（max/min 比）。

??? success "解答"
    **练习 5：负载均衡模拟**

    ```python
    import numpy as np
    from dataclasses import dataclass
    from typing import List, Tuple

    @dataclass
    class RouterState:
        num_experts: int
        logit_bias: np.ndarray  # Auxiliary bias for load balancing

    class MoERouter:
        def __init__(self, num_experts: int, hidden_dim: int, top_k: int = 2):
            self.num_experts = num_experts
            self.top_k = top_k
            self.W_gate = np.random.randn(hidden_dim, num_experts) * 0.02
            self.aux_loss_weight = 0.01
            self.load_history = []

        def route(self, x: np.ndarray, use_aux_loss: bool = True) -> Tuple[np.ndarray, np.ndarray]:
            """
            Route tokens to experts.

            Args:
                x: [batch, hidden] input tokens
                use_aux_loss: whether to apply load balancing

            Returns:
                expert_indices: [batch, top_k]
                expert_weights: [batch, top_k]
            """
            # Compute routing logits
            logits = x @ self.W_gate  # [batch, num_experts]

            # Apply softmax
            probs = self._softmax(logits)

            # Select top-k experts
            top_k_indices = np.argsort(probs, axis=-1)[:, -self.top_k:]
            top_k_probs = np.take_along_axis(probs, top_k_indices, axis=-1)

            # Normalize weights
            top_k_weights = top_k_probs / top_k_probs.sum(axis=-1, keepdims=True)

            # Record load
            load = np.bincount(top_k_indices.flatten(), minlength=self.num_experts)
            self.load_history.append(load)

            # Compute and apply auxiliary loss gradient (simplified)
            if use_aux_loss:
                self._update_for_balance(load)

            return top_k_indices, top_k_weights

        def _softmax(self, x: np.ndarray) -> np.ndarray:
            exp_x = np.exp(x - x.max(axis=-1, keepdims=True))
            return exp_x / exp_x.sum(axis=-1, keepdims=True)

        def _update_for_balance(self, load: np.ndarray):
            """Adjust gate weights to balance load."""
            # Penalize overloaded experts
            avg_load = load.mean()
            imbalance = (load - avg_load) / (avg_load + 1e-6)
            # Push routing away from overloaded experts
            self.W_gate -= self.aux_loss_weight * imbalance

        def get_imbalance_ratio(self) -> float:
            """Max/min load ratio from recent history."""
            if not self.load_history:
                return 1.0
            recent = np.array(self.load_history[-100:])
            avg_load = recent.mean(axis=0)
            return avg_load.max() / (avg_load.min() + 1e-6)

    def simulate_routing(num_steps: int = 1000, use_aux_loss: bool = True):
        """Simulate token routing over many steps."""
        router = MoERouter(num_experts=8, hidden_dim=512, top_k=2)

        for step in range(num_steps):
            # Random batch of tokens (in practice, comes from model)
            batch = np.random.randn(256, 512)
            router.route(batch, use_aux_loss=use_aux_loss)

            if step % 200 == 0:
                ratio = router.get_imbalance_ratio()
                print(f"Step {step}: imbalance ratio = {ratio:.2f}")

        return router.get_imbalance_ratio()

    # Run simulations
    print("With auxiliary loss:")
    ratio_with = simulate_routing(1000, use_aux_loss=True)
    print(f"Final imbalance: {ratio_with:.2f}x\n")

    print("Without auxiliary loss:")
    ratio_without = simulate_routing(1000, use_aux_loss=False)
    print(f"Final imbalance: {ratio_without:.2f}x")
    ```

    **预期结果：**

    | 条件 | Max/Min 负载比 | 效率 |
    |------|----------------|------|
    | 无辅助损失 | 3–10× | 30–50% 浪费 |
    | 有辅助损失 | 1.1–1.5× | <5% 浪费 |

    $$\boxed{\text{Auxiliary loss reduces imbalance from 5× to 1.3×}}$$

6. **投机解码加速**：若草稿模型比目标模型快 5 倍，平均接受 70% 的提议 token，投机深度 K=4，预期加速比是多少？

??? success "解答"
    **练习 6：投机解码加速**

    **已知：**
    - 草稿模型：比目标模型快 5×
    - 接受率：$p = 70\%$ 每 token
    - 投机深度：$K = 4$

    **投机解码过程：**
    1. 草稿模型生成 $K$ 个 token
    2. 目标模型并行验证全部 $K$ 个
    3. 接受匹配的前缀 token

    **每轮预期接受 token 数：**

    每 token 接受概率为 $p$：

    $$E[\text{accepted}] = \sum_{i=0}^{K} i \cdot p^i \cdot (1-p) + K \cdot p^K$$

    对 $K=4$、$p=0.7$：

    $$E[\text{accepted}] = 0 \cdot 0.3 + 1 \cdot 0.7 \cdot 0.3 + 2 \cdot 0.49 \cdot 0.3 + 3 \cdot 0.343 \cdot 0.3 + 4 \cdot 0.2401$$

    $$= 0 + 0.21 + 0.294 + 0.309 + 0.960 = 1.77 \text{ tokens}$$

    其实有更简单的公式：

    $$E[\text{accepted}] = \frac{1 - p^{K+1}}{1-p} - 1 = \frac{1 - 0.7^5}{0.3} - 1 = \frac{1 - 0.168}{0.3} - 1 = 1.77$$

    **时间分析：**

    无投机：
    - 每 token 1 次目标前向
    - 每 token 时间：$T_{target}$

    有投机：
    - $K$ 次草稿前向 + 1 次目标前向（并行验证）
    - 时间：$K \cdot T_{draft} + T_{target} = K \cdot \frac{T_{target}}{5} + T_{target} = T_{target}(1 + K/5)$
    - 生成 token 数：$E[\text{accepted}] + 1 = 2.77$（含被验证的 token）

    **加速比计算：**

    $$\text{Speedup} = \frac{E[\text{accepted}] + 1}{1 + K/5} = \frac{2.77}{1 + 4/5} = \frac{2.77}{1.8} = 1.54\times$$

    **更细致的模型：**

    | 轮次 | 草稿时间 | 目标时间 | Token 数 | 摊薄 |
    |------|----------|----------|----------|------|
    | 1 | 0.8T | 1.0T | 2.77 | 0.65T/token |
    | 基线 | 0 | 1.0T | 1 | 1.0T/token |

    $$\boxed{\text{Speedup} = \frac{1.0}{0.65} \approx 1.54\times}$$

    **敏感性分析：**

    | 接受率 | K | 预期 Token 数 | 加速比 |
    |--------|---|---------------|--------|
    | 60% | 4 | 2.13 | 1.18× |
    | 70% | 4 | 2.77 | 1.54× |
    | 80% | 4 | 3.59 | 2.00× |
    | 70% | 6 | 3.02 | 1.37× |

    接受率越高，加速越显著。K=4 在 p=70% 时接近最优。

## 不变量小结

| 不变量 | 主要压力 | 应对 |
|---|---|---|
| 内存 | 长上下文 KV 缓存 | GQA + 滑动窗口注意力 |
| 计算 | 高效注意力 | Flash 系列内核 |
| 通信 | 适度扩展 | 更简单的并行组合 |

## 关键要点

1. **滑动窗口注意力使长上下文成为可能**：内存与序列长度无关，有效上下文通过层堆叠增长。

2. **GQA 减少 KV 缓存**：在查询组之间共享键值头，带来 4–8 倍的内存缩减。

3. **组合节省是相乘的**：SWA + GQA 合起来可以把 KV 缓存减少 32 倍甚至更多。

4. **稀疏 MoE 以稀疏成本获得稠密质量**：激活 8 个专家中的 2 个即可达到大 3 倍的稠密模型的质量。

5. **专家并行高效分布 MoE**：每 GPU 一个专家，通过 all-to-all 路由 token。

6. **推理效率使实际部署成为可能**：小 KV 缓存意味着更大的批大小和更好的硬件利用率。

7. **架构效率可与规模相抗衡**：Mistral 的创新表明，巧妙的架构胜过蛮力扩展。