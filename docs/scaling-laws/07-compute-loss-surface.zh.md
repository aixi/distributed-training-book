---
title: "计算-损失曲面"
subtitle: "刻画资源与性能之间的关系"
---

<div class="chapter-opener" markdown>
损失是两个投入的函数：模型规模与训练数据。理解这个曲面，是高效分配算力的第一步。
</div>

<div class="investigation-question" markdown>
**问题**：你有固定的算力预算 C。应该用更少的数据训练更大的模型，还是用更多的数据训练更小的模型？损失曲面告诉我们存在一个最优分配——而 2022 年之前的大多数模型都选错了。
</div>

<div class="notation-banner" markdown>
**本章符号：** $\Psi$ = 参数数量，$D$ = 训练 token 数，$C$ = 总计算量，$\alpha,\beta$ = 缩放指数。参见[符号说明](../appendices/notation.md)。
</div>

!!! abstract "承上启下：第一部分 基础"
    本部分假定你已熟悉[第 1 章](../foundations/01-scale-imperative.md)的**三堵墙**（内存、时间、成本）、[第 2 章](../foundations/02-extended-roofline.md)的**扩展屋顶线**模型，以及[第 6 章](../foundations/06-estimation-discipline.md)的**估算思维**。现在我们要问：既然训练必须分布式进行，我们该如何在模型规模与训练数据之间最优地分配算力预算？

## 经验发现

2020 年，OpenAI 的研究者观察到一个引人注目的现象：语言模型的损失遵循平滑、可预测的幂律（power law）。把对数损失对对数参数或对数 token 作图，你会得到直线。

这并不显然。复杂系统常常表现出混沌行为。但神经语言模型在多个数量级上遵循：

$$L(\Psi) = \left(\frac{\Psi_c}{\Psi}\right)^{\alpha_{\Psi}}$$

$$L(D) = \left(\frac{D_c}{D}\right)^{\alpha_D}$$

其中 $\Psi_c$、$D_c$ 是临界尺度，$\alpha_{\Psi}$、$\alpha_D$ 是幂律指数。

## 损失曲面

把两种依赖关系结合起来：

$$L(\Psi, D) = \left[\left(\frac{\Psi_c}{\Psi}\right)^{\alpha_{\Psi} / \alpha} + \left(\frac{D_c}{D}\right)^{\alpha_D / \alpha}\right]^\alpha + L_\infty$$

或者使用常见的更简单的加和形式：

$$L(\Psi, D) = \frac{A}{\Psi^{\alpha}} + \frac{B}{D^{\beta}} + L_\infty$$

其中：

| 符号 | 含义 | 典型值 |
|------|------|--------|
| $\Psi$ | 参数数量 | $10^6$ 到 $10^{12}$ |
| $D$ | 训练 token 数 | $10^9$ 到 $10^{13}$ |
| $A$ | 参数缩放常数 | ~400 |
| $B$ | 数据缩放常数 | ~400 |
| $\alpha$ | 参数指数 | 0.076（Kaplan）/ 0.34（Chinchilla） |
| $\beta$ | 数据指数 | 0.095（Kaplan）/ 0.28（Chinchilla） |
| $L_\infty$ | 不可约损失（irreducible loss） | ~1.69 nats |

不可约损失 $L_\infty$ 代表自然语言的熵——即便是完美模型也无法预测不可预测之物。

## 计算约束

训练计算量主要由矩阵乘法决定。对一个 transformer：

$$C \approx 6\Psi D$$

**推导**：每个 token 都要经过各层，其中：

- 前向传播：约 $2\Psi$ FLOPs（矩阵乘法为参数量的 $2 \times$）
- 反向传播：约 $4\Psi$ FLOPs（激活梯度 + 权重梯度）
- 每 token 合计：约 $6\Psi$ FLOPs

对 $D$ 个 token：

$$C = 6\Psi D \text{ FLOPs}$$

!!! note "实践"
    如果预算固定，先从 $C = 6\Psi D$ 计算 $(\Psi, D)$ 组合，再决定你是为损失而训练（Chinchilla），还是为推理成本而训练（过训练更小的模型）。

这在 $(\Psi, D, C)$ 空间中形成了一个**约束曲面**。固定 $C$ 时，我们得到一条双曲线：

$$D = \frac{C}{6\Psi}$$

## 在计算约束下最小化损失

**问题**：在 $C = 6\Psi D$ 的约束下最小化 $L(\Psi, D)$。

!!! question "直觉检查"
    在推导最优解之前：如果你把算力预算翻倍，应该把大部分花在更大的模型上，还是更多的数据上？先做个预测——下面会看到 Chinchilla 令人意外的答案。

**方法**：拉格朗日乘数法（Lagrange multipliers）

拉格朗日函数：

$$\mathcal{L} = \frac{A}{\Psi^\alpha} + \frac{B}{D^\beta} + \lambda(C - 6\Psi D)$$

求偏导：

$$\frac{\partial \mathcal{L}}{\partial \Psi} = -\frac{A\alpha}{\Psi^{\alpha+1}} - 6\lambda D = 0$$

$$\frac{\partial \mathcal{L}}{\partial D} = -\frac{B\beta}{D^{\beta+1}} - 6\lambda \Psi = 0$$

由第一式：

$$\lambda = -\frac{A\alpha}{6D\Psi^{\alpha+1}}$$

代入第二式：

$$\frac{B\beta}{D^{\beta+1}} = \frac{A\alpha \Psi}{D\Psi^{\alpha+1}} = \frac{A\alpha}{\Psi^\alpha D}$$

因此：

$$\frac{B\beta}{D^\beta} = \frac{A\alpha}{\Psi^\alpha}$$

这说明：**在最优点，参数带来的损失下降边际贡献等于数据带来的边际贡献**。

整理得：

$$\frac{\Psi^\alpha}{D^\beta} = \frac{A\alpha}{B\beta}$$

## 最优分配

用 $C = 6\Psi D$ 消去一个变量。把 $D = C/(6\Psi)$ 代入最优性条件 $\Psi^\alpha / D^\beta = A\alpha/(B\beta)$：

$$\Psi^{\alpha+\beta} = \frac{A\alpha}{B\beta} \cdot \left(\frac{C}{6}\right)^\beta$$

$$\Psi^* = \left(\frac{A\alpha}{B\beta}\right)^{\frac{1}{\alpha+\beta}} \left(\frac{C}{6}\right)^{\frac{\beta}{\alpha+\beta}}$$

同理，代入 $\Psi = C/(6D)$：

$$D^* = \left(\frac{B\beta}{A\alpha}\right)^{\frac{1}{\alpha+\beta}} \left(\frac{C}{6}\right)^{\frac{\alpha}{\alpha+\beta}}$$

**关键洞察**：$\Psi^*$ 和 $D^*$ 都是 $C$ 的幂律。

如果 $\alpha \approx \beta$（Chinchilla：$\alpha \approx 0.34$，$\beta \approx 0.28$）：

$$\Psi^* \propto C^{0.45}, \quad D^* \propto C^{0.55}$$

最优比例为：

$$\frac{D^*}{\Psi^*} = \left(\frac{B\beta}{A\alpha}\right)^{\frac{2}{\alpha+\beta}} \left(\frac{C}{6}\right)^{\frac{\alpha-\beta}{\alpha+\beta}}$$

注意，除非 $\alpha = \beta$，该比例会（微弱地）依赖 $C$。对 Chinchilla 的参数，指数 $(\alpha - \beta)/(\alpha + \beta) \approx 0.10$ 很小，因此该比例随算力预算变化缓慢。经验上，Chinchilla 发现，在他们探索的算力预算范围内 $D^*/\Psi^* \approx 20$。

**20:1 规则**：最优训练大约每参数使用 20 个 token。这是 Hoffmann 等人（2022）凭经验观察到的比例，并非普适常数——它微弱地依赖算力预算 $C$。第 8 章会做完整的调和。

## Kaplan 与 Chinchilla

两篇有影响力的论文得出了不同的结论：

| 论文 | $\alpha$ | $\beta$ | 最优 $\Psi^* \propto$ | 最优 $D^* \propto$ | Token/参数 |
|------|----------|---------|----------------------|----------------------|------------|
| Kaplan (2020) | 0.076 | 0.095 | $C^{0.73}$ | $C^{0.27}$ | ~1.7 |
| Chinchilla (2022) | 0.34 | 0.28 | $C^{0.50}$ | $C^{0.50}$ | ~20 |

*注：Kaplan 的最优缩放指数（0.73、0.27）是经验拟合得到的，而非由 α 和 β 推导出来。理论推导 $\Psi^* \propto C^{\beta/(\alpha+\beta)}$ 给出的值不同，说明两者的拟合方法不同。*

**为什么会有差异？**

Kaplan 对每个模型只训练固定步数，而不是训练到收敛。这系统性地让较大的模型欠训练，使拟合偏向于“把模型做得更大”。

Chinchilla 则让每个规模的模型都训练到收敛，从而揭示了真实的缩放关系。

## 可视化曲面

损失曲面可以用等高线可视化：

```
L（损失）
    │
2.5 ├─────────────────────────────
    │   ╲
2.3 ├────╲────────────────────────
    │     ╲   等损失曲线
2.1 ├──────╲──────────────────────
    │       ╲
1.9 ├────────●────────────────────
    │      ↗  ╲    最优路径
1.7 ├─────●────╲──────────────────
    │    ↗      ╲
    └────┴───────┴────────────────→
        10⁹    10¹⁰    10¹¹      \Psi
```

最优路径沿着一条山脊延伸：在脊上，等计算量线与等损失曲线相切。

## 对分布式训练的含义

### 1. 训练效率比模型规模更重要

一个用 2T token 训练的 7B 模型，常常优于用 200B token 训练的 70B 模型，尽管参数量少 10 倍。

**对分布式系统而言**：高效的数据流水线（高吞吐）可能比为了更大模型而扩展到更多 GPU 更有价值。

### 2. 计算最优模型很吃内存

Chinchilla 最优训练意味着更多数据要流过模型。这会增加：

- 前向传播时的激活内存
- 梯度累积需求
- 数据加载带宽需求

### 3. 推理-训练的权衡

Chinchilla 最优模型的服务成本很高：相同质量下，模型更大，推理 FLOPs 更多。

许多从业者会刻意**过训练（overtrain）**更小的模型：

$$L_{\text{overtrained}}(\Psi_{\text{small}}, D_{\text{large}}) = L_{\text{optimal}}(\Psi_{\text{large}}, D_{\text{optimal}})$$

LLaMA 系列模型在 1–2T token 上训练，远超 Chinchilla 比例，以降低服务成本。

## 前沿模型方程

把缩放定律与硬件结合起来：

$$\text{Time} = \frac{6\Psi D}{\text{GPUs} \times \text{FLOPs/GPU} \times \text{MFU}}$$

其中 MFU（模型 FLOP 利用率）通常在 30–50%。

**示例**：Chinchilla（70B 参数，1.4T token）

$$C = 6 \times 70 \times 10^9 \times 1.4 \times 10^{12} = 5.9 \times 10^{23} \text{ FLOPs}$$

在 1000 块 H100（每块约 989 TFLOP/s 稠密 FP16/BF16）、40% MFU 下：

$$\text{Time} = \frac{5.9 \times 10^{23}}{1000 \times 9.89 \times 10^{14} \times 0.4} = \frac{5.9 \times 10^{23}}{3.96 \times 10^{17}} \approx 1.49 \times 10^6 \text{ s} \approx 17.3 \text{ days}$$

## 超越简单缩放

近期研究表明，这个曲面更为复杂：

### 数据质量
$$L(\Psi, D, q) = \frac{A}{\Psi^\alpha} + \frac{B}{(qD)^\beta} + L_\infty$$

其中 $q$ 是数据质量乘子。高质量数据可以移动最优比例。

### 架构效率
不同架构有不同的 $A$ 值。MoE 模型在相同的 $\Psi_{\text{active}}$ 下能达到更低的损失。

### 涌现能力
某些能力会在规模化时突然出现，并不遵循平滑的幂律。曲面上存在不连续点。

## 练习题

1. **最优分配**：给定 $C = 10^{24}$ FLOPs 和经验性的 Chinchilla 20:1 规则（$D^*/\Psi^* \approx 20$），计算最优模型规模和 token 数量。

??? success "解答"
    **使用 Chinchilla 20:1 规则进行最优分配：**

    最优比例为 $D^*/\Psi^* \approx 20$。

    代入计算约束 $C = 6\Psi D$：

    $$C = 6 \times \Psi^* \times 20\Psi^* = 120(\Psi^*)^2$$

    **求解最优模型规模：**
    $$\Psi^* = \sqrt{\frac{C}{120}} = \sqrt{\frac{10^{24}}{120}} = \sqrt{8.33 \times 10^{21}}$$

    $$\Psi^* = 2.89 \times 10^{10.5} \approx \boxed{91\text{B parameters}}$$

    **最优 token 数量：**
    $$D^* = 20 \times \Psi^* = 20 \times 91 \times 10^9 = \boxed{1.82\text{T token}}$$

    **验证：**
    $$C = 6 \times 91 \times 10^9 \times 1.82 \times 10^{12} = 9.94 \times 10^{23} \approx 10^{24} \checkmark$$

    | 参数 | 值 |
    |------|-----|
    | 最优 $\Psi^*$ | 91B |
    | 最优 $D^*$ | 1.82T |
    | Token/参数 | 20 |

2. **训练时间**：你有 512 块 H100，MFU 为 45%。训练练习 1 中的最优模型需要多久？

??? success "解答"
    **使用训练时间公式：**

    $$T = \frac{C}{\text{GPUs} \times \text{Peak FLOP/s} \times \text{MFU}}$$

    **已知：**

    - $C = 10^{24}$ FLOPs
    - GPU 数 = 512
    - H100 峰值 = 989 TFLOP/s = $9.89 \times 10^{14}$ FLOP/s
    - MFU = 45% = 0.45

    **计算：**

    $$T = \frac{10^{24}}{512 \times 9.89 \times 10^{14} \times 0.45}$$

    $$T = \frac{10^{24}}{4.56 \times 10^{17}} = 2.19 \times 10^6 \text{ seconds}$$

    **换算为天数：**
    $$T = \frac{2.19 \times 10^6}{3600 \times 24} = \boxed{25.4 \text{ days}}$$

    **实际考虑：**

    | 因素 | 影响 |
    |------|------|
    | 检查点开销 | 增加约 5–10% |
    | 硬件故障 | 预留约 10% 停机时间 |
    | 现实时间线 | 约 30–32 天 |

3. **过训练分析**：LLaMA-2 7B 在 2T token 上训练。
   - 7B 参数对应的 Chinchilla 最优 token 数是多少？
   - 它被过训练了多少倍？
   - 估算它与“用最优 token 数训练 7B 模型”之间的损失差异。

??? success "解答"
    **第 1 部分：Chinchilla 最优 token 数**

    使用 20:1 规则：

    $$D^* = 20 \times \Psi = 20 \times 7 \times 10^9 = \boxed{140\text{B token}}$$

    **第 2 部分：过训练倍数**

    $$\text{Overtraining factor} = \frac{D_{\text{actual}}}{D^*} = \frac{2 \times 10^{12}}{140 \times 10^9} = \boxed{14.3\times}$$

    **第 3 部分：损失差异估计**

    使用 $L(\Psi, D) = \frac{A}{\Psi^\alpha} + \frac{B}{D^\beta} + L_\infty$，采用 Chinchilla 指数（$\alpha = 0.34$，$\beta = 0.28$）：

    数据相关项的改善：

    $$\Delta L_{\text{data}} = B\left(\frac{1}{D_{\text{opt}}^\beta} - \frac{1}{D_{\text{overtrain}}^\beta}\right)$$

    数据项之比：

    $$\frac{(140\text{B})^{0.28}}{(2\text{T})^{0.28}} = \left(\frac{140}{2000}\right)^{0.28} = 0.07^{0.28} \approx 0.47$$

    与 Chinchilla 最优相比，过训练模型将数据相关损失项降低了约 53%（比例 0.47 意味着只承担 47% 的数据惩罚）。

    **估计改善：过训练可使损失降低约 0.1–0.2 nats。**

    **关键洞察**：过训练用训练算力换取推理效率。LLaMA-2 7B 的表现与约 15–20B 的 Chinchilla 最优模型相当，而服务成本低 2–3 倍。

4. **等损失曲线**：推导 $(\Psi, D)$ 平面上等损失曲线 $L(\Psi, D) = L_0$ 的方程。它是什么形状？

??? success "解答"
    **从损失方程出发：**

    $$L_0 = \frac{A}{\Psi^\alpha} + \frac{B}{D^\beta} + L_\infty$$

    **整理等损失曲线：**

    令 $L' = L_0 - L_\infty$（可约损失）：

    $$L' = \frac{A}{\Psi^\alpha} + \frac{B}{D^\beta}$$

    **求解 $D$ 关于 $\Psi$ 的函数：**

    $$\frac{B}{D^\beta} = L' - \frac{A}{\Psi^\alpha}$$

    $$D^\beta = \frac{B}{L' - A/\Psi^\alpha} = \frac{B\Psi^\alpha}{L'\Psi^\alpha - A}$$

    $$\boxed{D = \left(\frac{B\Psi^\alpha}{L'\Psi^\alpha - A}\right)^{1/\beta}}$$

    **形状分析：**

    | 性质 | 值 |
    |------|-----|
    | 垂直渐近线 | $\Psi = (A/L')^{1/\alpha}$ |
    | 水平渐近线 | $D = (B/L')^{1/\beta}$ |
    | 形状 | 第一象限内类似双曲线的曲线 |
    | 曲率 | 朝向原点凸出 |

    **几何解释：**

    - 对固定损失 $L_0$，存在一族达到它的 $(\Psi, D)$ 组合
    - 更大的模型达到相同损失所需的数据更少（反之亦然）
    - 曲线的渐近线表明：即使对另一个维度无限投入，仍需要的最小资源

5. **边际回报**：在当前训练点 $(\Psi_0, D_0)$，你可以选择把参数翻倍或把数据翻倍。哪个能更多降低损失？推导无差异条件。

??? success "解答"
    **把 $\Psi$ 翻倍带来的损失下降：**

    $$\Delta L_\Psi = \frac{A}{\Psi_0^\alpha} - \frac{A}{(2\Psi_0)^\alpha} = \frac{A}{\Psi_0^\alpha}\left(1 - \frac{1}{2^\alpha}\right) = \frac{A}{\Psi_0^\alpha}(1 - 2^{-\alpha})$$

    **把 $D$ 翻倍带来的损失下降：**

    $$\Delta L_D = \frac{B}{D_0^\beta} - \frac{B}{(2D_0)^\beta} = \frac{B}{D_0^\beta}(1 - 2^{-\beta})$$

    **无差异条件（$\Delta L_\Psi = \Delta L_D$）：**

    $$\frac{A}{\Psi_0^\alpha}(1 - 2^{-\alpha}) = \frac{B}{D_0^\beta}(1 - 2^{-\beta})$$

    $$\boxed{\frac{A(1 - 2^{-\alpha})}{\Psi_0^\alpha} = \frac{B(1 - 2^{-\beta})}{D_0^\beta}}$$

    **对 Chinchilla（$\alpha = \beta$，$A = B$）：**

    条件简化为：

    $$\Psi_0^\alpha = D_0^\alpha \implies \Psi_0 = D_0$$

    但 20:1 规则（$D^*/\Psi^* = 20$）表明 $A \neq B$。实际的无差异点位于 Chinchilla 最优点，在那里边际回报相等。

    **决策规则：**

    | 条件 | 行动 |
    |------|------|
    | $\frac{A(1-2^{-\alpha})}{\Psi_0^\alpha} > \frac{B(1-2^{-\beta})}{D_0^\beta}$ | 参数翻倍 |
    | $\frac{A(1-2^{-\alpha})}{\Psi_0^\alpha} < \frac{B(1-2^{-\beta})}{D_0^\beta}$ | 数据翻倍 |
    | 相等 | 两种选择等价 |

    **实践含义**：如果你处在 Chinchilla 最优点，把任一项翻倍的边际收益相同。2022 年之前的大多数模型在数据上欠训练，因此把 $D$ 翻倍更有价值。

## 关键要点

1. **损失在参数和数据上都遵循幂律**，使有原则的算力分配成为可能。

2. **20:1 规则**：Chinchilla 最优训练大约每参数使用 20 个 token。

3. **最优分配**：$\Psi^*$ 和 $D^*$ 都按 $C^{0.5}$ 缩放——算力翻倍，各乘 $\sqrt{2}$。

4. **边际平衡**：在最优点，最后一 FLOP 花在参数上还是数据上，带来的损失下降相同。

5. **实践权衡**：推理成本常常推动人们过训练更小的模型。