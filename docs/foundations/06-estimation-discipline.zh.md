---
title: "估算作为一种纪律"
subtitle: "分布式训练的费米估算"
---

<div class="chapter-opener" markdown>
在跑昂贵的实验之前，先估算。一名容量工程师应当只用基本算术，就能把训练时间、内存占用和通信成本预测到与实际值相差 2 倍以内。
</div>

<div class="investigation-question" markdown>
**问题**：你计划在 128 块 H100 上训练一个 300 亿参数的模型。什么都不跑，先估算：(1) 每 GPU 内存，(2) 每步时间，(3) 每秒 token 数。在花出去一分钱之前，这些估算就应指导你的硬件选择。
</div>

## 估算心态

好的估算应当：

- **快**：5 分钟的计算，而不是 5 天的 profiling
- **近似**：对规划而言，2 倍以内的误差通常就够了
- **保守**：高估成本，低估吞吐

目标是快速识别不可行的配置，把实验集中在可行的选项上。

!!! note "实践"
    这里的估算使用理想化峰值。务必用你实际技术栈上的 profiler 测量做合理性检查。

示例中使用的默认吞吐与带宽数值见[硬件假设与单位](../appendices/hardware-assumptions.md)。

## 内存估算

### 模型内存

对一个具有以下属性的 transformer：

- $L$ 层
- $H$ 隐层维
- $V$ 词表大小
- $A$ 个注意力头

总参数：

$$\Psi \approx 12LH^2 + 2VH$$

参数内存（混合精度）：

$$M_{\text{params}} = 2\Psi \text{ bytes (FP16)}$$

优化器内存（Adam，FP32）：

$$M_{\text{opt}} = 12\Psi \text{ bytes (params + momentum + variance)}$$

总静态内存：

$$M_{\text{static}} = 2\Psi + 2\Psi + 12\Psi = 16\Psi$$

### 激活内存

每层激活内存（不检查点）：

$$M_{\text{act}}^{\text{layer}} \approx BSH \cdot (34 + 5\frac{AS}{H})$$

其中 $B$ = 批大小，$S$ = 序列长度，$H$ = 隐藏维度，$A$ = 头数。这个系数是基于常见激活张量计数的粗略经验值。

总激活内存：

$$M_{\text{act}} = L \cdot M_{\text{act}}^{\text{layer}}$$

使用激活检查点（每 $k$ 层重算）时，峰值内存由两部分组成：(1) $L/k$ 个边界上的检查点存储（只存隐状态输入，各约 $2BSH$ 字节），以及 (2) 重算期间最多 $k$ 层的完整逐层激活：

$$M_{\text{act}}^{\text{ckpt}} \approx \frac{L}{k} \cdot 2BSH + k \cdot M_{\text{act}}^{\text{layer}}$$

最优的 $k = \sqrt{L}$ 使总内存最小。本章通篇使用一个更简单（也更保守）的近似：

$$M_{\text{act}}^{\text{ckpt}} \approx \left(\frac{L}{k} + k\right) \cdot M_{\text{act}}^{\text{layer}}$$

这会高估检查点项（用完整的 $M_{\text{act}}^{\text{layer}}$ 而非仅 $2BSH$），但更容易计算。

### 快速估算表

| 组成 | 公式 | 70B 模型 |
|------|------|----------|
| 参数 | $2\Psi$ | 140 GB |
| 梯度 | $2\Psi$ | 140 GB |
| 优化器 | $12\Psi$ | 840 GB |
| **静态合计** | $16\Psi$ | **1.12 TB** |

这立刻告诉我们：70B 光静态内存就需要至少 14 块 H100。

## 计算估算

### 每 token 的 FLOPs

前向：

$$F_{\text{fwd}} \approx 2\Psi$$

反向：

$$F_{\text{bwd}} \approx 4\Psi$$

每 token 合计：

$$F_{\text{total}} = 6\Psi$$

### 每步时间

$$T_{\text{step}} = \frac{F_{\text{step}}}{\text{MFU} \times \text{Peak FLOP/s} \times P}$$

其中：

- $F_{\text{step}} = 6\Psi \cdot B \cdot S$（$B \cdot S$ 个 token）
- 对优化良好的训练，MFU ≈ 0.40–0.50
- P = GPU 数量

**例子**：70B 模型，批=100 万 token，128 块 H100，45% MFU：

$$T_{\text{step}} = \frac{6 \times 70 \times 10^9 \times 10^6}{0.45 \times 989 \times 10^{12} \times 128} = 7.37 \text{ seconds}$$

### 每秒 token 数

$$\text{Tokens/s} = \frac{B \cdot S}{T_{\text{step}}} = \frac{10^6}{7.37} \approx 136,000 \text{ tokens/s}$$

### 训练时间

$$T_{\text{train}} = \frac{D}{B \cdot S} \times T_{\text{step}} = \frac{D \times 6\Psi}{\text{MFU} \times F_{\text{peak}} \times P}$$

对 2T token：

$$T_{\text{train}} = \frac{2 \times 10^{12}}{136,000} \approx 14.7 \times 10^6 \text{ seconds} \approx 170 \text{ days}$$

## 通信估算

### 数据并行

每步 AllReduce 通信量：$2\Psi$ 字节

AllReduce 时间（环形，$P$ 块 GPU，带宽 $\beta$）：

$$T_{\text{AR}} = \frac{2(P-1)}{P} \cdot \frac{2\Psi}{\beta}$$

70B 跨 128 块 GPU、50 GB/s：

$$T_{\text{AR}} = 2 \times \frac{127}{128} \times \frac{140 \times 10^9}{50 \times 10^9} \approx 5.6 \text{ seconds}$$

### 张量并行

每层 AllReduce：$2 \times B \times S \times H$ 字节

8 路 TP，B=4，S=4096，H=8192：

$$T_{\text{TP}}^{\text{layer}} = \frac{2 \times 4 \times 4096 \times 8192}{900 \times 10^9} \approx 0.3 \text{ ms}$$

每步合计（80 层，每层 2 次 AllReduce）：约 48ms

### 流水线并行

气泡比例：

$$\text{Bubble} = \frac{p - 1}{m + p - 1}$$

其中 $p$ = 流水线阶段数，$m$ = 微批数。

p=8、m=32 时：Bubble = 7/39 ≈ 18%

## 估算工作流

1. **内存检查**：模型放得下吗？需要多少并行？
2. **计算估算**：理论吞吐是多少？
3. **通信估算**：通信占多大比例的时间？
4. **瓶颈识别**：哪一重上限占主导？
5. **合理性检查**：与类似的已公开训练对比

## 常见估算错误

| 错误 | 后果 | 修正 |
|------|------|------|
| 忘记优化器状态 | 内存低估 3 倍 | 始终计入 12Ψ |
| 忽略激活值 | 训练中 OOM | 计入 $B \cdot S \cdot H$ |
| 假设 100% MFU | 时间低估 2 倍 | 用 40–50% MFU |
| 忽略通信 | 理论可行、实践失败 | 加上 AllReduce/AllGather 时间 |

## 练习题

1. 估算一个 13B 模型在 TP=4、跨 32 块 GPU 的 ZeRO-3 下每 GPU 的内存。假设 batch=8、sequence=4096、hidden=5120、layers=40。

??? success "解答"
    **并行配置：**

    - GPU 总数：32
    - TP = 4（张量并行组）
    - DP = 32/4 = 8（数据并行副本）
    - ZeRO-3 在 DP 维度上分片

    **静态内存（ZeRO-3 跨 32 块 GPU 分片）：**

    | 组成 | 公式 | 每 GPU |
    |------|------|--------|
    | 参数 | $\frac{2\Psi}{32}$ | $\frac{2 \times 13 \times 10^9}{32} = 0.81$ GB |
    | 梯度 | $\frac{2\Psi}{32}$ | 0.81 GB |
    | 优化器 | $\frac{12\Psi}{32}$ | 4.87 GB |
    | **静态合计** | | **6.49 GB** |

    **激活内存：**

    使用公式：$M_{act}^{layer} \approx BSH \times (34 + 5\frac{AS}{H})$

    假设 40 个注意力头（$A = 40$）：

    $$BSH = 8 \times 4096 \times 5120 = 167.8\text{M}$$

    $$34 + 5 \times \frac{40 \times 4096}{5120} = 34 + 160 = 194$$

    $$M_{act}^{layer} = 167.8\text{M} \times 194 \text{ bytes} \approx 32.5 \text{ GB per layer}$$

    **TP=4 时**：激活值被分散，每 GPU 成本约降低 4 倍：

    $$M_{act}^{layer, TP} \approx 8.1 \text{ GB per layer}$$

    **使用激活检查点**（每 4 层检查点）：

    $$M_{act}^{total} \approx \left(\frac{40}{4} + 4\right) \times 8.1 \approx 113 \text{ GB}$$

    这仍然太大，说明在不再增加并行、不减小批大小或不缩短序列的情况下，这个批／序列组合不现实。

    **每 GPU 总内存：**

    | 组成 | 内存 |
    |------|------|
    | 静态（ZeRO-3） | 6.5 GB |
    | 激活（TP=4，检查点） | ~113 GB |
    | 临时缓冲区 | ~5 GB |
    | **合计** | **~125 GB** |

    这**放不进** 80 GB。即便 ZeRO-3 几乎消除了静态内存，这个批／序列组合下的激活内存仍占主导。你需要更多 TP/PP、更小的批／序列，或更激进的重算（例如完全重算会把激活内存降到约 4 × 8.1 = 32 GB，总量约 44 GB——那就放得下了）。

2. 一次训练在 64 块 H100 上为 7B 模型达到 150K tokens/s。计算 MFU。

??? success "解答"
    **每 token FLOPs：**

    $$F_{token} = 6\Psi = 6 \times 7 \times 10^9 = 42 \times 10^9 \text{ FLOPs/token}$$

    **实际达到的 FLOPs/s：**

    $$F_{achieved} = 42 \times 10^9 \times 150 \times 10^3 = 6.3 \times 10^{15} \text{ FLOP/s} = 6,300 \text{ TFLOP/s}$$

    **峰值 FLOPs/s（64 块 H100）：**

    $$F_{peak} = 64 \times 989 \text{ TFLOP/s} = 63,296 \text{ TFLOP/s}$$

    **MFU：**

    $$\text{MFU} = \frac{6,300}{63,296} = \boxed{9.95\% \approx 10\%}$$

    **分析：** 这是非常低的 MFU，说明存在显著低效。可能原因：

    | 问题 | 可能影响 |
    |------|----------|
    | 批大小过小 | 计算利用不足 |
    | 流水线气泡过多 | 阶段间空闲时间 |
    | kernel 未优化 | SM 利用率低 |
    | 通信未重叠 | GPU 等待网络 |

    **优化良好的 7B 训练的预期 MFU：** 40–50%

    在 45% MFU 下，预期吞吐为：

    $$\text{tokens/s} = \frac{0.45 \times 63,296 \times 10^{12}}{42 \times 10^9} \approx 0.68\text{M tokens/s}$$

    观测到的 150K 仅为该潜力的 11%。

3. 你需要在 30 天内、以 200 万美元预算训练一个 30B 模型。估算所需的最少 H100 数量（按 4 美元/小时），以及满足时间线所需的 MFU。

??? success "解答"
    **预算约束：**

    $$\text{Max GPU-hours} = \frac{\$2,000,000}{\$4/\text{hr}} = 500,000 \text{ GPU-hours}$$

    **时间约束：**

    $$T_{max} = 30 \text{ days} = 720 \text{ hours}$$

    **预算给出的 GPU 上限：**

    $$P_{max} = \frac{500,000}{720} \approx 694 \text{ GPUs}$$

    取整到实用值：**P = 640 块 GPU**（或取 2 的幂 512）

    **训练 FLOPs（假设 Chinchilla 最优的 600B token）：**

    $$F_{total} = 6 \times 30 \times 10^9 \times 600 \times 10^9 = 1.08 \times 10^{23} \text{ FLOPs}$$

    **所需吞吐：**

    $$F_{required} = \frac{1.08 \times 10^{23}}{720 \times 3600} = 4.17 \times 10^{16} \text{ FLOP/s}$$

    **640 块 H100 的峰值吞吐：**

    $$F_{peak} = 640 \times 989 \times 10^{12} = 6.33 \times 10^{17} \text{ FLOP/s}$$

    **所需 MFU：**

    $$\text{MFU}_{required} = \frac{4.17 \times 10^{16}}{6.33 \times 10^{17}} = \boxed{6.6\%}$$

    **结论：** 这很容易达到——优化良好的训练能到 40–50% MFU。

    **替代方案：训练更多 token（2T）**

    $$F_{total}^{2T} = 6 \times 30 \times 10^9 \times 2 \times 10^{12} = 3.6 \times 10^{23} \text{ FLOPs}$$

    $$\text{MFU}_{required}^{2T} = \frac{3.6 \times 10^{23} / (720 \times 3600)}{6.33 \times 10^{17}} = 22\%$$

    用标准优化仍然非常可行。

    **小结：**

    | 场景 | Token | GPU | 所需 MFU |
    |------|-------|-----|----------|
    | Chinchilla（600B） | 600B | 640 | 6.6% |
    | 扩展（2T） | 2T | 640 | 22% |
    | 预算受限 | 2T | 512 | ~28% |

## 关键要点

1. **估算是第一等技能**：粗略的数字能避免不可能的计划。
2. **预算、时间和 MFU 三者可互换**：任意两个决定第三个。
3. **始终对照硬件极限做合理性检查**：峰值 FLOPs 与内存上限约束着每一个计划。