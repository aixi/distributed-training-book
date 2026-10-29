---
title: "Chinchilla 最优性"
subtitle: "20:1 比例与计算最优训练"
---

<div class="chapter-opener" markdown>
多年来，这个领域一直在没有足够数据的情况下扩展模型。Chinchilla 揭示了最优平衡：大约每个参数 20 个 token。
</div>

<div class="investigation-question" markdown>
**问题**：GPT-3 有 175B 参数，却只在 300B token 上训练——比例是 1.7:1。Chinchilla 有 70B 参数和 1.4T token（20:1），在相近算力下（$5.9 \times 10^{23}$ 对 $3.15 \times 10^{23}$ FLOPs）达到了*更好*的损失。在 GPT-3 的预算下，一个计算最优模型本可以用小得多的规模匹配它的损失。到底哪里出了问题，我们又该如何计算“最优”？
</div>

<div class="notation-banner" markdown>
**本章符号：** $\Psi$ = 参数数量，$D$ = 训练 token 数，$C$ = 总计算量。参见[符号说明](../appendices/notation.md)。
</div>

## Chinchilla 之前的时代

2022 年之前，主流的扩展策略是：**把模型做得更大**。

这一信念来自 Kaplan 等人（2020），他们提出：

- 模型规模应按 $\Psi \propto C^{0.73}$ 缩放
- 数据应按 $D \propto C^{0.27}$ 缩放

按照这一处方，算力翻倍意味着：

- 参数增加 1.66×
- token 增加 1.20×

结果是：模型越来越欠训练（undertrained）。

| 模型 | 参数 | Token | Token/参数 |
|------|------|-------|------------|
| GPT-2 | 1.5B | 40B | 26.7 |
| GPT-3 | 175B | 300B | 1.7 |
| Gopher | 280B | 300B | 1.1 |
| Megatron-Turing | 530B | 270B | 0.5 |

每个模型都比前代更大，但训练数据量大致相同。到 Megatron-Turing 时，模型平均每个 token 只见过 0.5 次。

## Chinchilla 的方法

Hoffmann 等人（2022）用不同的方式处理这个问题。他们训练了 400 多个模型，覆盖很大的规模范围（70M 到 16B）和 token 数量，**每个都训练到收敛**。

三种独立方法得出了相同的答案：

### 方法 1：固定模型，变化数据

对每个模型规模 $\Psi$，拟合：

$$L(D) = \frac{B}{D^\beta} + L_\infty(\Psi)$$

为每个规模提取 $L_\infty(\Psi)$，然后拟合：

$$L_\infty(\Psi) = \frac{A}{\Psi^\alpha}$$

### 方法 2：等 FLOP 曲线

固定算力预算 $C$。训练许多满足 $C = 6\Psi D$ 的 $(\Psi, D)$ 组合。
绘制损失对 $\Psi$ 的曲线，找最小值。

对每个 $C$，存在一个最优 $\Psi^*(C)$：

```
损失
  │
  │  ●                              C = 10²¹
  │   ●  ●
  │      ↘  ●
  │         ↘ ● ← 最优 \Psi
  │            ● ●
  │               ●
  └──────────────────────────────→ \Psi
```

### 方法 3：参数化拟合

对所有数据同时拟合：

$$L(\Psi, D) = \frac{A}{\Psi^\alpha} + \frac{B}{D^\beta} + L_\infty$$

其中 $L_\infty$ 是不可约损失（无限算力下可达到的最小损失）。加入拉格朗日约束 $C = 6\Psi D$，推导最优缩放。

**三种方法一致**：$\Psi^* \propto C^{0.50}$，$D^* \propto C^{0.50}$。

## 20:1 比例的推导

由最优分配（第 7 章）：

$$\frac{D^*}{\Psi^*} = \frac{B\beta}{A\alpha}$$

代入 Chinchilla 的拟合值：

- $A = 406.4$，$\alpha = 0.34$
- $B = 410.7$，$\beta = 0.28$

$$\frac{D^*}{\Psi^*} = \frac{410.7 \times 0.28}{406.4 \times 0.34} = \frac{115.0}{138.2} \approx 0.83$$

等等——这小于 1，而不是 20。怎么回事？

**解释**：上面的值适用于乘法形式。常被引用的“20:1”来自另一种参数化和拟合流程。

更精确地说，根据论文：

$$\Psi_{\text{opt}} = 0.6225 \cdot C^{0.4957}$$

$$D_{\text{opt}} = 1.8421 \cdot C^{0.5043}$$

在 $C = 10^{21}$ FLOPs 时：

- $\Psi_{\text{opt}} \approx 1.9 \times 10^{10}$（19B）
- $D_{\text{opt}} \approx 4.0 \times 10^{11}$（400B）
- 比例：$D/\Psi \approx 21$

**20:1 规则**是一个有用的近似：

$$D_{\text{optimal}} \approx 20 \cdot \Psi$$

## 计算最优分配

给定算力预算 $C$：

**第 1 步**：估计最优参数量
$$\Psi^* \approx \sqrt{\frac{C}{6 \times 20}} = \sqrt{\frac{C}{120}}$$

**第 2 步**：估计最优 token 数
$$D^* \approx 20 \cdot \Psi^*$$

**算例**：$C = 10^{24}$ FLOPs（约等于 Chinchilla 级别的训练预算）

$$\Psi^* \approx \sqrt{\frac{10^{24}}{120}} = \sqrt{8.33 \times 10^{21}} \approx 9.1 \times 10^{10}$$

$$D^* \approx 20 \times 9.1 \times 10^{10} = 1.82 \times 10^{12}$$

**计算最优**：约 91B 参数，约 1.82T token。

!!! note "真实系统如何偏离"
    作为对比，传闻中 GPT-4 的规格约为 1.8T 参数（混合专家）并在约 13T token 上训练。如果属实，这相对于 Chinchilla 就是严重的**过训练**——而且是有意为之，为了推理效率（一个更小、过训练的模型比计算最优的更大模型服务更快）。这种有意过训练的模式已成为标准做法；下一节将量化这一现象。

## 欠训练的模型

Chinchilla 之前的模型到底“错”了多少？

### GPT-3 分析

- 参数：$\Psi = 175 \times 10^9$
- Token：$D = 300 \times 10^9$
- 计算量：$C = 6\Psi D = 3.15 \times 10^{23}$

该计算量下的 Chinchilla 最优：

$$\Psi^* = \sqrt{\frac{3.15 \times 10^{23}}{120}} \approx 51B$$

$$D^* = 20 \times 51B = 1.02T$$

GPT-3 使用了**多出 3.4 倍的参数**和**少 3.4 倍的 token**。

损失惩罚：

$$\Delta L \approx \frac{A}{\Psi_{\text{GPT-3}}^\alpha} + \frac{B}{D_{\text{GPT-3}}^\beta} - \frac{A}{\Psi^{*\alpha}} - \frac{B}{D^{*\beta}}$$

Chinchilla 用约少 4 倍的算力达到了 GPT-3 的损失。

### Gopher 分析

- 参数：$\Psi = 280 \times 10^9$
- Token：$D = 300 \times 10^9$
- 计算量：$C = 5.04 \times 10^{23}$

Chinchilla 最优：

$$\Psi^* \approx 65B, \quad D^* \approx 1.3T$$

Gopher **过参数化了 4.3 倍**。

## Chinchilla 陷阱

Chinchilla 最优性最小化的是每 FLOP 的损失。但这并不总是正确的目标。

### 推理成本问题

一个 Chinchilla 最优的 70B 模型，损失为 $L$，推理时每 token 需要 70B 次乘加。

一个过训练的 7B 模型（训练数据比 Chinchilla 最优多 10 倍）可能以**少 10 倍的推理 FLOPs** 达到相同的损失 $L$。

总成本对比：

| 方案 | 训练成本 | 推理成本（每 token） |
|------|----------|----------------------|
| Chinchilla 70B | $C$ | $70B$ 次乘加 |
| 过训练 7B | $1.5C$ | $7B$ 次乘加 |

如果你要服务超过 $10^{13}$ 个 token，过训练就是划算的。

### LLaMA 哲学

LLaMA 1 和 2 有意过训练：

| 模型 | 参数 | Token | Token/参数 | 相对 Chinchilla |
|------|------|-------|------------|------------------|
| LLaMA-7B | 7B | 1T | 143 | 过训练 7× |
| LLaMA-13B | 13B | 1T | 77 | 过训练 3.8× |
| LLaMA-65B | 65B | 1.4T | 22 | 约等于 Chinchilla |
| LLaMA-2 7B | 7B | 2T | 286 | 过训练 14× |

7B 模型在 1–2T token 上训练——**比 Chinchilla 最优多 7–14 倍**——以最小化服务成本。

### 数据质量考量

20:1 规则假设存在无限的同质数据。在实践中：

1. **高质量数据有限**：一旦用尽高质量来源，更多 token ≠ 更好
2. **重复有害**：在重复数据上训练收益递减
3. **合成数据**：可以扩展有效数据集规模，但可能有不同的缩放性质

## 何时偏离 Chinchilla

### 场景 1：推理主导的工作负载

如果推理成本 >> 训练成本，就过训练更小的模型。

**盈亏平衡分析**：令 $r = \frac{\text{inference tokens}}{\text{training tokens}}$

按因子 $k$ 过训练（用 $kD$ 个 token 训练 $\Psi/k$ 个参数）在以下条件下有利可图：

$$r > \frac{k^2 \cdot \text{training cost per token}}{\text{inference cost per param}}$$

按典型比例，当服务超过 $10^{13}$ 个 token 时，过训练开始划算。

### 场景 2：数据稀缺

如果高质量数据在 $D_{\text{max}}$ 处耗尽：

$$\Psi^* = \frac{D_{\text{max}}}{20}$$

训练更大的模型会把算力浪费在欠训练的参数上。

### 场景 3：能力阈值

某些能力会在特定模型规模出现，与训练 token 数无关。

思维链推理大约在 60B+ 参数时出现。如果你需要这种能力，70B × 200B（欠训练）在推理任务上可能优于 7B × 2T（过训练）。

### 场景 4：时间约束

训练时间随 token 数增长。如果墙钟时间是约束：

$$\text{Time} \propto \frac{D}{\text{throughput}}$$

更少的 token = 更快的训练。一个欠训练的大模型可能是“尽快跑通”场景下的最优选择。

## Chinchilla 之后的缩放定律

近期工作细化和扩展了 Chinchilla：

### 计算最优 vs 下游最优

Chinchilla 优化的是预训练损失。但下游任务表现可能有不同的缩放：

$$\text{Accuracy}(\Psi, D) \neq f(L(\Psi, D))$$

有些任务更受益于模型规模，另一些更受益于数据。

### 混合专家缩放

MoE 模型有不同的缩放规律：

- $\Psi_{\text{total}}$ 与 $\Psi_{\text{active}}$
- 每 token 只有激活参数贡献 FLOPs
- 固定算力下，更多的总参数可以改善损失

MoE 的缩放定律：

$$L(\Psi_{\text{active}}, \Psi_{\text{total}}, D) = \frac{A}{\Psi_{\text{total}}^{\alpha_1} \Psi_{\text{active}}^{\alpha_2}} + \frac{B}{D^\beta} + L_\infty$$

其中 $\alpha_1$ 和 $\alpha_2$ 分别是总参数和激活参数的拟合指数。

### 重复 Token

如果 $D > D_{\text{unique}}$ 会怎样？Muennighoff 等人（2023）表明：

$$L(\Psi, D, r) = L(\Psi, D_{\text{unique}}) + \gamma \cdot \log(r)$$

其中 $r = D / D_{\text{unique}}$ 是重复比例，$\gamma$ 是经验拟合系数，用来刻画重复数据的价值递减。

4 个 epoch ≈ 4 倍唯一数据通常可以接受；超过 16 个 epoch 后，收益迅速递减。

## 练习题

1. **计算最优**：你有 $C = 10^{22}$ FLOPs。计算 Chinchilla 最优的模型规模和 token 数量。

??? success "解答"
    **使用 Chinchilla 20:1 规则：**

    由 $C = 6\Psi D$ 和 $D^* = 20\Psi^*$：

    $$C = 6 \times \Psi^* \times 20\Psi^* = 120(\Psi^*)^2$$

    **求解最优模型规模：**
    $$\Psi^* = \sqrt{\frac{C}{120}} = \sqrt{\frac{10^{22}}{120}} = \sqrt{8.33 \times 10^{19}}$$

    $$\Psi^* = 9.13 \times 10^9 \approx \boxed{9.1\text{B parameters}}$$

    **最优 token 数量：**
    $$D^* = 20 \times \Psi^* = 20 \times 9.1 \times 10^9 = \boxed{182\text{B token}}$$

    **验证：**
    $$C = 6 \times 9.1 \times 10^9 \times 182 \times 10^9 = 9.94 \times 10^{21} \approx 10^{22} \checkmark$$

    | 参数 | 值 |
    |------|-----|
    | 算力预算 | $10^{22}$ FLOPs |
    | 最优 $\Psi^*$ | 9.1B |
    | 最优 $D^*$ | 182B |
    | Token/参数 | 20 |

2. **欠训练分析**：一个模型有 30B 参数，在 150B token 上训练。
   - 使用了多少计算量？
   - 该计算量下的 Chinchilla 最优分配是什么？
   - 该模型过参数化/欠参数化了多少倍？

??? success "解答"
    **第 1 部分：使用的计算量**

    $$C = 6\Psi D = 6 \times 30 \times 10^9 \times 150 \times 10^9 = \boxed{2.7 \times 10^{22} \text{ FLOPs}}$$

    **第 2 部分：Chinchilla 最优分配**

    $$\Psi^* = \sqrt{\frac{C}{120}} = \sqrt{\frac{2.7 \times 10^{22}}{120}} = \sqrt{2.25 \times 10^{20}}$$

    $$\Psi^* = 1.5 \times 10^{10} = \boxed{15\text{B parameters}}$$

    $$D^* = 20 \times \Psi^* = 20 \times 15 \times 10^9 = \boxed{300\text{B token}}$$

    **第 3 部分：过参数化/欠参数化倍数**

    $$\text{Overparameterization} = \frac{\Psi_{\text{actual}}}{\Psi^*} = \frac{30\text{B}}{15\text{B}} = \boxed{2\times \text{ overparameterized}}$$

    $$\text{Undertraining} = \frac{D^*}{D_{\text{actual}}} = \frac{300\text{B}}{150\text{B}} = 2\times \text{ undertrained on data}$$

    **总结：**

    | 指标 | 实际 | 最优 | 比例 |
    |------|------|------|------|
    | 参数 | 30B | 15B | 过 2× |
    | Token | 150B | 300B | 欠 2× |
    | Token/参数 | 5 | 20 | 比最优低 4× |

    该模型严重欠训练：每参数的 token 数比 Chinchilla 最优少 4 倍。

3. **推理盈亏平衡**：一个 70B 的 Chinchilla 最优模型训练成本为 $10M。一个达到相似损失的 7B 过训练模型训练成本为 $15M。70B 的推理成本为每 1M token $0.001，7B 为每 1M token $0.0001。你需要服务多少 token 后过训练才划算？

??? success "解答"
    **总成本模型：**

    $$C_{\text{total}} = C_{\text{train}} + C_{\text{inference}} \times T$$

    其中 $T$ 是服务的 token 数（以百万计）。

    **对 70B Chinchilla 模型：**
    $$C_{70B} = \$10\text{M} + \$0.001 \times T$$

    **对 7B 过训练模型：**
    $$C_{7B} = \$15\text{M} + \$0.0001 \times T$$

    **盈亏平衡条件：**
    $$\$10\text{M} + \$0.001 \times T = \$15\text{M} + \$0.0001 \times T$$

    $$\$0.001T - \$0.0001T = \$5\text{M}$$

    $$\$0.0009 \times T = \$5\text{M}$$

    $$T = \frac{\$5 \times 10^6}{\$0.0009} = 5.56 \times 10^9 \text{ million tokens}$$

    $$T = \boxed{5.56 \times 10^{15} \text{ tokens} \approx 5.6 \text{ quadrillion tokens}}$$

    **解读：**

    | 服务 token 数 | 更便宜的方案 |
    |---------------|--------------|
    | < 5.6 千万亿 | 70B Chinchilla |
    | > 5.6 千万亿 | 7B 过训练 |

    作为参考，一个高流量推理服务每天可能服务约 1–10T token。按每天 5T token 计算：

    $$\text{Break-even time} = \frac{5.6 \times 10^{15}}{5 \times 10^{12}} \approx 1{,}120 \text{ days} \approx 3 \text{ years}$$

    对超高流量推理，过训练可能划算——但盈亏平衡时间很大程度上取决于实际服务量。

4. **数据预算**：你恰好有 500B 高质量 token。你应该训练的最大模型是多少？

??? success "解答"
    **反向使用 Chinchilla 20:1 规则：**

    如果 $D_{\text{max}} = 500\text{B}$ token，且最优比例为 $D^*/\Psi^* = 20$：

    $$\Psi^* = \frac{D_{\text{max}}}{20} = \frac{500 \times 10^9}{20} = \boxed{25\text{B parameters}}$$

    **所需计算量：**
    $$C = 6\Psi D = 6 \times 25 \times 10^9 \times 500 \times 10^9 = 7.5 \times 10^{22} \text{ FLOPs}$$

    **为什么不更大？**

    | 模型规模 | 问题 |
    |----------|------|
    | > 25B | 欠训练（token/参数 < 20） |
    | < 25B | 浪费数据容量 |
    | = 25B | 数据预算下的 Chinchilla 最优 |

    **替代方案：接受欠训练**

    如果用 500B token 训练 50B 模型：

    - Token/参数 = 10（最优值的一半）
    - 损失比用相同数据训练的 25B 模型高约 20%
    - 但更大模型可能具备涌现能力

    **建议**：就损失而言 25B 最优；更大规模是用损失换能力。

5. **MoE 分析**：一个稠密 70B 模型和一个 70B 激活 / 1T 总参数的 MoE 都在 1.4T token 上训练。哪个达到更低的损失？（假设 MoE 从总参数获得的每参数损失下降约为激活参数的 1.5 倍）

??? success "解答"
    **稠密 70B 模型：**

    使用 $L(\Psi, D) = \frac{A}{\Psi^\alpha} + \frac{B}{D^\beta} + L_\infty$：

    $$L_{\text{dense}} \propto \frac{1}{(70\text{B})^\alpha}$$

    **MoE 模型分析：**

    MoE 具有：
    - 激活参数：$\Psi_{\text{active}} = 70\text{B}$
    - 总参数：$\Psi_{\text{total}} = 1\text{T}$

    总参数带来 1.5× 的损失下降：

    $$L_{\text{MoE}} \propto \frac{1}{(\Psi_{\text{active}})^\alpha \cdot (\Psi_{\text{total}}/\Psi_{\text{active}})^{\alpha/2}}$$

    用于损失缩放（loss scaling）的有效参数量：

    $$\Psi_{\text{eff}} = \Psi_{\text{active}} \cdot \left(\frac{\Psi_{\text{total}}}{\Psi_{\text{active}}}\right)^{0.5} = 70\text{B} \cdot \left(\frac{1000\text{B}}{70\text{B}}\right)^{0.5}$$

    $$\Psi_{\text{eff}} = 70\text{B} \cdot 3.78 = 265\text{B}$$

    **损失对比：**

    | 模型 | 有效 $\Psi$ | 相对损失项 |
    |------|-------------|------------|
    | 稠密 70B | 70B | $(70\text{B})^{-0.34} = 1.00$ |
    | MoE 70B/1T | 265B | $(265\text{B})^{-0.34} = 0.69$ |

    **在每次前向计算量相同的情况下，MoE 的参数相关损失低约 31%。**

    **为什么 MoE 胜出：**

    - 推理成本相同（70B 激活参数）
    - 专家中存储更多知识（1T 总参数）
    - 每个 token 路由到专家
    - 有效容量 >> 激活容量

    **注意事项**：训练 MoE 需要在内存/通信中容纳约 1T 参数，增加了基础设施复杂性。1.5× 系数是经验值，随架构而异。

## 关键要点

1. **Chinchilla 之前的模型欠训练**：GPT-3、Gopher 等使用的参数相对于数据多出 3–4 倍。

2. **20:1 规则**：计算最优训练大约每参数使用 20 个 token。

3. **Chinchilla 优化的是损失/FLOP**：这不等同于最小化推理成本或最大化能力。

4. **过训练常常是理性的**：当推理成本占主导时，更小的过训练模型胜出。

5. **了解你的约束**：数据限制、推理预算、时间压力和能力需求都会移动最优分配。