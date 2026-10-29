---
title: "架构感知的效率"
subtitle: "GQA、滑动窗口注意力、MLA 与 DualPipe"
---

<div class="chapter-opener" markdown>
影响最大的效率收益，往往不来自系统技巧，而来自在源头减少通信、内存或计算的架构选择。本章介绍四项改变分布式训练格局的架构创新——理解它们，为第八部分的案例研究做好准备。
</div>

<div class="investigation-question" markdown>
**问题**：标准多头注意力为每个头存储独立的 K、V 投影。128 个头、128K 序列长度时，仅 KV 缓存就可能超过 100 GB。我们能否重新设计注意力机制本身，在不损失质量的前提下把它减少 10–50 倍？
</div>

!!! abstract "本章导览"
    **前置知识**：[第 15 章](../parallelism/15-tensor-parallelism-linearity.md)（张量并行）、[第 17 章](../parallelism/17-sequence-parallelism-decomposability.md)（序列并行）、[第 16 章](../parallelism/16-pipeline-parallelism-separability.md)（流水线调度）

    **关键洞察**：架构选择（注意力模式、KV 共享、流水线调度）是杠杆最高的效率旋钮，因为它们减少的是根本性的工作量，而不只是如何分布它。

## 分组查询注意力（GQA）

### KV 缓存问题

在标准多头注意力（Multi-Head Attention, MHA）中，$A$ 个注意力头各自拥有独立的 Key 和 Value 投影：

$$\text{KV cache per token} = 2 \times A \times d_h \times s \text{ bytes}$$

其中 $d_h = H/A$ 是每头维度，$s$ 是每元素字节数。对一个 70B 模型，$A=64$、$H=8192$、BF16：

$$\text{KV cache per token} = 2 \times 64 \times 128 \times 2 = 32{,}768 \text{ bytes} \approx 32 \text{ KB}$$

128K 上下文时：$32 \text{ KB} \times 128{,}000 = 4.1 \text{ GB per layer}$。80 层则是**328 GB**——超过单块 GPU。

### GQA：共享 KV 头

分组查询注意力（Grouped-Query Attention, Ainslie 等人，2023）把 $A$ 个查询头分成 $g$ 组，每组共享一个 KV 头：

$$\text{KV cache (GQA)} = 2 \times g \times d_h \times s$$

相对 MHA 的缩减因子为 $A/g$：

| 变体 | KV 头数（$g$） | 相对 MHA 的缩减 | 质量影响 |
|------|----------------|-----------------|----------|
| MHA | $g = A$（如 64） | 1×（基线） | — |
| GQA-8 | $g = 8$ | 8× | 极小 |
| GQA-1（MQA） | $g = 1$ | 64× | 轻微退化 |

LLaMA 2 70B 和 LLaMA 3 使用 GQA-8（$g = 8$）。这把 KV 缓存减少 8×，质量损失可忽略。

### 对分布式训练的影响

| 方面 | MHA | GQA-8 |
|------|-----|-------|
| KV 缓存内存 | 每 token $2AHs$ | 每 token $2gHs/A \cdot A = 2gd_h s$ |
| TP 通信（KV） | AllGather $A$ 个 KV 头 | AllGather $g$ 个 KV 头（少 8×） |
| 序列并行内存 | $O(S \cdot A \cdot d_h)$ | $O(S \cdot g \cdot d_h)$ |
| 推理批容量 | 受 KV 缓存限制 | 每 GPU 可容纳 8× 序列数 |

!!! note "实践"
    GQA 现在是所有主流 LLM 的默认选择。如果要设计新模型，从 GQA-8 开始，除非你有特定理由需要完整 MHA。

## 滑动窗口注意力（SWA）

### 从全局注意力到局部注意力

标准注意力计算全对（all-pairs）交互：每层 $O(S^2)$。滑动窗口注意力（Sliding Window Attention, SWA）把每个 token 限制为只关注最近的 $w$ 个 token：

$$\text{Attention}(Q_i, K, V) = \text{softmax}\left(\frac{Q_i K_{[i-w:i]}^T}{\sqrt{d_h}}\right) V_{[i-w:i]}$$

**内存**：$O(S \times w)$ 而非 $O(S^2)$

**有效上下文**：有 $L$ 层时，信息传播 $L \times w$ 个 token——一个 $w=4096$ 的 32 层模型，有效上下文为 131K token。

### 与 GQA 结合

SWA 和 GQA 可以相乘式组合：

$$\text{KV cache (SWA + GQA)} = 2 \times g \times d_h \times w \times s$$

这与序列长度 $S$ 无关——缓存被窗口大小所界定。

| 配置 | 每层 KV 缓存（BF16，$H=4096$，$S=32K$） |
|------|----------------------------------------|
| MHA，全局注意力 | $2 \times 32 \times 128 \times 32K \times 2 = 524$ MB |
| GQA-8，全局 | $2 \times 8 \times 128 \times 32K \times 2 = 131$ MB |
| GQA-8，SWA（$w=4096$） | $2 \times 8 \times 128 \times 4096 \times 2 = 16.8$ MB |

两种技术结合后，**缩减 31×**。

### 对分布式训练的影响

- **序列并行（Sequence Parallelism）**：SWA 让 Ring Attention 更便宜——每个分块只需从邻居获取 $w$ 个 token 的上下文，而非完整的前序序列
- **流水线并行（Pipeline Parallelism）**：阶段边界上的激活张量更小
- **内存**：可以用释放的内存增大批大小，提升 MFU

Mistral 7B 证明 SWA + GQA 可以匹敌两倍大的模型——完整分析见[第 36 章](../synthesis/36-case-study-mistral.md)。

## 多头潜在注意力（MLA）

### 超越 GQA：压缩潜在空间

GQA 减少了 KV 头数。MLA（DeepSeek-V2/V3）更进一步，把 KV 表示压缩进低维潜在空间：

$$c_t = W_{\text{DKV}} \cdot h_t \qquad K_t = W_{\text{UK}} \cdot c_t \qquad V_t = W_{\text{UV}} \cdot c_t$$

其中：

- $h_t \in \mathbb{R}^H$ 是隐状态
- $c_t \in \mathbb{R}^{d_c}$ 是压缩后的潜在表示（$d_c \ll H$）
- $W_{\text{DKV}} \in \mathbb{R}^{d_c \times H}$ 负责压缩，$W_{\text{UK}}, W_{\text{UV}}$ 负责解压

**KV 缓存只存 $c_t$**，而非完整的 K、V：

$$\text{KV cache (MLA)} = d_c \times s \text{ per token per layer}$$

对 $d_c = 512$、$H = 7168$（DeepSeek-V3）：潜在维度比隐层维度小 $14\times$。相对 MHA 的实际 KV 缓存压缩更大——约 $65\times$——因为 MHA 每 token 存 $2 \times A \times d_h$ 个值，而 MLA 只存 $d_c$（见下表）。

### 对比

| 方法 | 每 token 每层 KV 缓存（BF16） | 相对 MHA |
|------|-------------------------------|----------|
| MHA（$A=128$，$d_h=128$） | $2 \times 128 \times 128 \times 2 = 65.5$ KB | 1× |
| GQA-8 | $2 \times 8 \times 128 \times 2 = 4.1$ KB | 0.063× |
| MLA（$d_c = 512$） | $512 \times 2 = 1.0$ KB | 0.015× |

MLA 相对 MHA 达到 **~65× 压缩**。这让 DeepSeek-V3 能以实用的 KV 缓存规模服务 671B 参数。

### 训练影响

训练期间，解压 $K_t = W_{\text{UK}} c_t$ 可以被吸收进查询投影（一个矩阵代数技巧），避免显式解压。这意味着：

- **前向传播**：与标准注意力相同的 FLOPs（解压被融合）
- **内存**：反向传播只存 $c_t$ → 激活内存大幅节省
- **通信**：TP 和 SP 的激活张量更小

DeepSeek-V3 对 MLA 的完整运用见[第 35 章](../synthesis/35-case-study-deepseek.md)。

## DualPipe：双向流水线调度

### 重访气泡问题

标准 1F1B 流水线并行的气泡（pipeline bubble）比例为 $(p-1)/(m+p-1)$。$p=16$、$m=32$ 时：气泡 = 32%。零气泡调度（ZB-H1）能解决，但会增加内存压力。

### DualPipe：两条流水线，一次通过

DualPipe（DeepSeek-V3）把微批（micro-batch）分成两股流，以相反方向流过流水线：

```
Stream A: Stage 0 → Stage 1 → ... → Stage P-1   (forward direction)
Stream B: Stage P-1 → Stage P-2 → ... → Stage 0  (reverse direction)
```

每个阶段在两股流的微批之间交替。当一股流处于通信阶段（发送激活）时，另一股处于计算阶段。

### 气泡缩减

$$\text{Bubble}_{\text{DualPipe}} \approx \frac{p-1}{2m}$$

与 1F1B 的 $(p-1)/(m+p-1)$ 相比。$p=16$、$m=32$ 时：

- 1F1B：$15/47 = 31.9\%$
- DualPipe：$15/64 = 23.4\%$

改进随 $m$ 增大：当 $m \to \infty$，1F1B 气泡缓慢趋近 0，而 DualPipe 的气泡以两倍速度趋近 0。

### 关键洞察：跨流重叠通信与计算

DualPipe 的真正威力在于：当流 A 等待激活传输时，流 B 可以计算——反之亦然。这把流水线通信时间转化为有效的计算，有效地隐藏了阶段间延迟。

**约束**：需要足够内存同时容纳两股流的激活。内存开销约为标准 1F1B 的 $2\times$。

DualPipe 在 DeepSeek-V3 训练中的作用见[第 35 章](../synthesis/35-case-study-deepseek.md)。

## 小结：架构即效率

| 创新 | 主要节省 | 缩减因子 | 采用者 |
|------|----------|----------|--------|
| GQA | KV 缓存内存 | 4–64× | LLaMA 2/3、Mistral、Gemma |
| SWA | 注意力内存／计算 | $S/w$ × | Mistral、Mixtral |
| MLA | KV 缓存 + 激活内存 | 相对 MHA 约 65×（$H/d_c$ = 14× 潜在压缩） | DeepSeek-V2/V3 |
| DualPipe | 流水线气泡时间 | ~2× | DeepSeek-V3 |

这些不是系统优化——它们是**改变根本性资源需求的架构选择**。理解它们对后续的案例研究至关重要。

## 关键要点

1. **GQA 是新的默认**：跨查询组共享 KV 头，缓存减少 8×，质量损失可忽略。

2. **SWA 使内存与序列长度无关**：注意力内存从 $O(S^2)$ 变为 $O(S \times w)$。

3. **MLA 进一步压缩 KV 瓶颈**：低秩潜在表示实现 65× 的 KV 缓存缩减。

4. **DualPipe 把有效气泡比例减半**：双向微批流隐藏阶段间通信。

5. **架构与分布协同设计**：最好的效率收益来自在源头减少工作，而不只是更好地分布它。

## 练习题

1. **GQA 内存节省**：一个模型有 $A=64$ 个注意力头、$d_h=128$。比较以下情形每 token 每层的 KV 缓存大小（BF16）：(a) 标准 MHA，(b) 8 个 KV 组的 GQA，(c) 1 个 KV 组的 GQA（MQA）。缩减因子各是多少？

??? success "解答"
    - **MHA**：$2 \times 64 \times 128 \times 2 = 32{,}768$ 字节 = 32 KB
    - **GQA-8**：$2 \times 8 \times 128 \times 2 = 4{,}096$ 字节 = 4 KB → **8× 缩减**
    - **MQA**（GQA-1）：$2 \times 1 \times 128 \times 2 = 512$ 字节 → **64× 缩减**

    缩减因子等于 $A/g$，其中 $g$ 是 KV 组数。

2. **SWA 有效上下文**：一个模型有 $L=32$ 层、滑动窗口大小 $w=4{,}096$。(a) 最大有效上下文是多少？(b) 若输入长度 $S=128{,}000$，有多少层能从位置 $S$ "看到"第一个 token？

??? success "解答"
    **(a)** 最大有效上下文 $= L \times w = 32 \times 4{,}096 = 131{,}072$ 个 token。

    **(b)** 只有当 $S \leq L \times w$ 时，才能从位置 $S$ 到达第一个 token。由于 $128{,}000 < 131{,}072$，第一个 token 可达。把信息从位置 $S$ 传播到位置 1 所需的层数为 $\lceil S/w \rceil = \lceil 128{,}000/4{,}096 \rceil = 32$ 层。所以只有最后一层能"看到"第一个 token，且只是勉强看到。

3. **DualPipe vs 1F1B**：比较 $P=16$ 个阶段、$m=32$ 个微批的流水线在以下情形下的气泡比例：(a) 标准 1F1B，(b) DualPipe。1F1B 需要多少额外的微批才能匹配 DualPipe 的气泡比例？

??? success "解答"
    **(a)** 1F1B 气泡：$\frac{P-1}{m+P-1} = \frac{15}{32+15} = \frac{15}{47} \approx 31.9\%$

    **(b)** DualPipe 气泡：$\approx \frac{P-1}{2m} = \frac{15}{64} \approx 23.4\%$

    要让 1F1B 匹配 DualPipe：$\frac{15}{m'+15} = \frac{15}{64}$，故 $m'+15 = 64$，得 $m' = 49$。1F1B 需要 **49 个微批**（DualPipe 只需 32 个）才能达到相同的气泡比例——增加 53%。