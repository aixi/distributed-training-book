---
title: "批大小与学习动力学"
subtitle: "临界批大小、学习率缩放与 LARS/LAMB"
---

<div class="chapter-opener" markdown>
批大小不只是一个内存参数——它从根本上影响学习动力学。存在一个临界批大小（critical batch size），超过它之后收益会迅速递减。理解这一点，对于扩展到数千块 GPU 至关重要。
</div>

<div class="investigation-question" markdown>
**问题**：你想从 8 块 GPU 扩展到 10,000 块 GPU。朴素做法是：把批大小增加 1,250×。但用 1M 批大小训练的模型常常无法收敛。极限在哪里，我们又该如何突破它？
</div>

## 梯度噪声与批大小

每个小批次都提供真实梯度的一个带噪估计：

$$g_B = \frac{1}{B} \sum_{i=1}^{B} \nabla L(x_i) = \nabla L + \epsilon_B$$

其中 $\epsilon_B$ 是噪声，其方差为：

$$\text{Var}(\epsilon_B) = \frac{\sigma^2}{B}$$

这里 $\sigma^2$ 是逐样本梯度方差（数据和模型的性质）。批越大 → 噪声越低 → 梯度方向越可靠。

但这里有一个关键洞察：**噪声并不总是坏事**。

### 噪声的有益作用

梯度噪声：
1. 帮助逃离尖锐极小值（它们泛化很差）
2. 提供隐式正则化（implicit regularization）
3. 使损失景观的探索成为可能

噪声太小 → 可能收敛到尖锐、不泛化的极小值。

## 临界批大小

McCandlish 等人（2018）推导出临界批大小 $B_{\text{crit}}$：

$$B_{\text{crit}} = \frac{G^2}{H}$$

其中：

- $G^2 = \mathbb{E}[||\nabla L||^2]$：期望梯度范数的平方
- $H = \mathbb{E}[(\nabla L)^T \nabla^2 L (\nabla L)]$：沿梯度方向的曲率

等价地，使用噪声尺度：

$$B_{\text{crit}} = \frac{\text{tr}(\Sigma)}{\text{tr}(H)}$$

其中 $\Sigma$ 是梯度协方差。

### 解读

- **$B < B_{\text{crit}}$**：训练是**噪声主导**的。每一步因梯度噪声而很小。需要很多步。

- **$B \approx B_{\text{crit}}$**：最优平衡。每步可靠，又不浪费算力。

- **$B > B_{\text{crit}}$**：训练是**曲率主导**的。额外样本带来递减的回报。算力被浪费。

### 经验值

临界批大小随任务和训练阶段变化：

| 领域 | 典型 $B_{\text{crit}}$ |
|------|------------------------|
| ImageNet（训练早期） | 2K – 8K |
| ImageNet（训练后期） | 16K – 64K |
| 语言模型（小） | 256 – 2K |
| 语言模型（大） | 2M – 8M |

注意：随着模型接近最小值、曲率下降，$B_{\text{crit}}$ 在训练过程中会**增大**。

## 完美缩放定律

在 $B_{\text{crit}}$ 以下，训练可以完美扩展：

$$\text{Steps}(B) = \frac{S_0 \cdot B_0}{B}$$

其中 $S_0$ 是基线批大小 $B_0$ 下的步数。

批大小翻倍 → 步数减半 → 每步墙钟时间相同 → 训练快 2×。

**总计算量保持不变**：

$$C = S \cdot B \cdot (\text{FLOPs per sample}) = \text{constant}$$

## 收益递减区间

超过 $B_{\text{crit}}$ 后，关系变为：

$$\text{Steps}(B) = S_{\min} + \frac{S_{\text{noise}}}{B}$$

其中 $S_{\min}$ 是不论批大小如何都无法低于的最少步数（曲率极限）。

当 $B \to \infty$：

$$\text{Steps}(B) \to S_{\min}$$

无论批多大，你都无法把步数降到 $S_{\min}$ 以下。

**算力浪费**：

$$\text{Wasted compute} = (B - B_{\text{crit}}) \cdot S(B) \cdot (\text{FLOPs per sample})$$

## 学习率缩放

增大批大小时，必须调整学习率。问题是：如何调整？

### 线性缩放规则（Goyal 等人，2017）

$$\eta(B) = \eta_0 \cdot \frac{B}{B_0}$$

**直觉**：如果批大小翻倍，梯度的可靠性也翻倍，因此可以走两倍大的步。

**适用条件**：

- $B \leq B_{\text{crit}}$
- 使用带动量的 SGD

**推导**：比较批大小 $B_0$ 下 $k$ 步的更新与批大小 $kB_0$ 下 1 步的更新：

小批：

$$\Delta w = -\eta_0 \sum_{i=1}^k g_i \approx -k\eta_0 \bar{g}$$

大批：

$$\Delta w' = -\eta' \cdot g_{kB_0} \approx -\eta' \bar{g}$$

要等价：$\eta' = k\eta_0$

### 平方根缩放

$$\eta(B) = \eta_0 \cdot \sqrt{\frac{B}{B_0}}$$

**直觉**：梯度中的噪声按 $1/\sqrt{B}$ 缩放，因此学习率应随噪声下降而缩放。

**适用条件**：

- 超过 $B_{\text{crit}}$
- 损失景观更复杂

**推导**：从凸优化中 SGD 收敛速度的角度看，对带噪梯度，最优学习率是 $\eta \propto 1/\sqrt{B}$。

### 该用哪个？

| 区间 | 缩放规则 |
|------|----------|
| $B \ll B_{\text{crit}}$ | 线性 |
| $B \approx B_{\text{crit}}$ | 介于线性与平方根之间 |
| $B > B_{\text{crit}}$ | 平方根 |
| $B \gg B_{\text{crit}}$ | 常数（增大没有收益） |

实践方法：**线性缩放 + 预热**，如果不稳定再降下来。

## 预热

训练开始时的大学习率会导致发散。解决方案：预热（warmup）。

### 线性预热

$$\eta(t) = \eta_{\max} \cdot \frac{t}{T_{\text{warmup}}}$$

对于 $t \leq T_{\text{warmup}}$。

### 为什么预热有帮助

训练早期：
1. 梯度大且噪声多
2. 损失景观曲率高
3. 模型远离任何极小值

大步伐会导致：

- 梯度爆炸
- 灾难性更新
- 发散

预热让模型在迈大步之前先“站稳脚跟”。

### 预热时长

经验法则：

$$T_{\text{warmup}} \approx \frac{B}{B_0} \cdot T_0$$

其中 $T_0$ 是基线批大小下的预热步数。

对非常大的批（>64K），可能需要更长的预热。

## 逐层自适应学习率

不同层有不同的梯度幅值。标准学习率在非常深或非常宽的网络、大批大小下表现很差。

### LARS（You 等人，2017）

用于 SGD 的逐层自适应学习率缩放（Layer-wise Adaptive Rate Scaling）：

$$\eta_l = \eta \cdot \phi(||w_l||, ||\nabla w_l||)$$

其中信任比（trust ratio）为：

$$\phi = \frac{||w_l||}{||\nabla w_l|| + \lambda ||w_l||}$$

这里 $\lambda$ 是权重衰减系数（通常为 0.0001 到 0.001）。

**直觉**：用权重范数与梯度范数之比来缩放学习率。防止任何一层相对于其当前尺度更新过多。

更新变为：

$$w_l \leftarrow w_l - \eta \cdot \phi_l \cdot (\nabla w_l + \lambda w_l)$$

**LARS 让 ImageNet 能在 1 小时内用 32K 批大小训练完成**（标准 SGD 需要数天）。

### LAMB（You 等人，2019）

用于批训练的逐层自适应矩估计（Layer-wise Adaptive Moments for Batch training）——把 LARS 与 Adam 结合起来：

$$m_l^{(t)} = \beta_1 m_l^{(t-1)} + (1-\beta_1) \nabla w_l$$

$$v_l^{(t)} = \beta_2 v_l^{(t-1)} + (1-\beta_2) (\nabla w_l)^2$$

$$\hat{m}_l = \frac{m_l}{1-\beta_1^t}, \quad \hat{v}_l = \frac{v_l}{1-\beta_2^t}$$

$$r_l = \frac{\hat{m}_l}{\sqrt{\hat{v}_l} + \epsilon}$$

其中 $\epsilon$ 是用于数值稳定的一个小常数（通常为 $10^{-8}$）。

然后应用 LARS 风格的信任比：

$$\phi_l = \frac{||w_l||}{||r_l + \lambda w_l||}$$

$$w_l \leftarrow w_l - \eta \cdot \phi_l \cdot (r_l + \lambda w_l)$$

**LAMB 让 BERT 能在 76 分钟内用 65K 批大小训练完成**（Adam 需要 3 天）。

### 对比

| 方法 | 基础优化器 | 最大批大小（ImageNet） | 最大批大小（BERT） |
|------|------------|------------------------|--------------------|
| SGD + 线性 LR | SGD | ~8K | 不适用 |
| LARS | SGD | 32K | ~8K |
| Adam + 线性 LR | Adam | ~16K | ~16K |
| LAMB | Adam | ~32K | 65K |

## 批大小与时间的权衡

更大的批大小能让你在更多 GPU 上做数据并行（Data Parallelism）。但收益会递减。

### 扩展效率

定义扩展效率：

$$E(B) = \frac{S(B_0)/S(B)}{B/B_0}$$

- $E = 1$：完美扩展（线性加速）
- $E < 1$：次线性扩展
- $E \to 0$：算力浪费

低于 $B_{\text{crit}}$：$E \approx 1$

高于 $B_{\text{crit}}$：$E$ 迅速下降

### 何时扩展

**扩展批大小**当：

- 墙钟时间是约束
- 你低于 $B_{\text{crit}}$
- GPU 利用率很高

**不要扩展批大小**当：

- 已经高于 $B_{\text{crit}}$
- 最终质量比速度更重要
- 超参数调优很困难

## 梯度累积

大批大小装不进内存？累积梯度：

```python
optimizer.zero_grad()
for i, batch in enumerate(mini_batches):
    loss = model(batch) / accumulation_steps
    loss.backward()  # 累积梯度

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**在数学上等价于**大批大小，但：

- 前向/反向传播次数更多
- 内存占用与小批相同
- 比真正的数据并行慢

使用场景：GPU 内存限制了有效批大小，但尚未达到 $B_{\text{crit}}$。

## 动态批大小

$B_{\text{crit}}$ 会在训练中增大。最优策略是：**在训练过程中增大批大小**。

### LLaMA 的做法

LLaMA 2（以及后来的 LLaMA 3）在训练中途增大批大小：

- 开始：较小的批大小（例如 LLaMA 2 的 2M token）
- 在初始训练阶段后提升到 4M token

*注：LLaMA 1 在整个训练过程中使用约 4M token 的固定批大小。批大小爬升是在 LLaMA 2 中引入的。*

好处：
1. 训练早期：较小批用于探索
2. 训练后期：较大批提高效率
3. 总步数减少

### 自适应缩放

监控梯度噪声尺度：

$$\text{noise scale} = \frac{||\text{Var}(\nabla L)||}{||\mathbb{E}[\nabla L]||^2}$$

当噪声尺度下降时，可以安全地增大批大小。

## 大批训练的实用配方

1. **建立基线**：用小批（256–1024）训练，找到最优 $\eta_0$

2. **估计 $B_{\text{crit}}$**：把批翻倍，检查步数是否减半。当不再减半时停止。

3. **按线性规则缩放**：$\eta = \eta_0 \cdot B/B_0$，直到 $B_{\text{crit}}$

4. **使用预热**：$T_{\text{warmup}} \propto B/B_0$

5. **考虑 LARS/LAMB**：通常对 $B > 8K$ 是必需的

6. **仔细监控**：

   - 损失尖峰 → 降低学习率或增加预热
   - 收敛缓慢 → 可能超过 $B_{\text{crit}}$
   - 逐层梯度范数 → 检查是否不平衡

7. **动态批大小**：随训练进展增大批大小

## 练习题

1. **临界批大小**：一个模型在批大小 256 下需要 100K 步训练完成。批大小 1024 下需要 28K 步。批大小 4096 下需要 15K 步。估计 $B_{\text{crit}}$。

??? success "解答"
    **给定数据：**

    | 批大小 | 收敛步数 |
    |--------|----------|
    | 256 | 100,000 |
    | 1,024 | 28,000 |
    | 4,096 | 15,000 |

    **使用收益递减模型：**

    $$S(B) = S_{\min} + \frac{S_{\text{noise}}}{B}$$

    **检查从 B=256 到 B=1024（4× 增加）的完美缩放：**

    如果完美缩放：$S(1024) = 100,000 / 4 = 25,000$

    实际：28,000 步 → 已经出现收益递减

    **建立方程：**

    由 $S(256) = 100,000$：

    $$S_{\min} + \frac{S_{\text{noise}}}{256} = 100,000$$

    由 $S(1024) = 28,000$：

    $$S_{\min} + \frac{S_{\text{noise}}}{1024} = 28,000$$

    **求解：**

    第一式减第二式：

    $$S_{\text{noise}} \left(\frac{1}{256} - \frac{1}{1024}\right) = 72,000$$

    $$S_{\text{noise}} \times \frac{3}{1024} = 72,000$$

    $$S_{\text{noise}} = 24,576,000$$

    代回：

    $$S_{\min} = 28,000 - \frac{24,576,000}{1024} = 28,000 - 24,000 = 4,000$$

    **用 B=4096 验证：**
    $$S(4096) = 4,000 + \frac{24,576,000}{4096} = 4,000 + 6,000 = 10,000$$

    实际：15,000 步（差异表明模型只是近似）

    **估计 $B_{\text{crit}}$：**

    临界批大小是噪声贡献与曲率贡献相等之处：

    $$\frac{S_{\text{noise}}}{B_{\text{crit}}} = S_{\min}$$

    $$B_{\text{crit}} = \frac{S_{\text{noise}}}{S_{\min}} = \frac{24,576,000}{4,000} = \boxed{6,144}$$

    **解读：** 批大小超过约 6K 后会出现明显的收益递减。数据显示在 4096 时我们已经超过 $B_{\text{crit}}$，确认估计值在合理范围内。

2. **学习率缩放**：你从批 256、$\eta = 0.001$ 扩展到批 4096。在 (a) 线性缩放、(b) 平方根缩放 下应使用什么学习率？

??? success "解答"
    **已知：**

    - 基础批大小：$B_0 = 256$
    - 基础学习率：$\eta_0 = 0.001$
    - 目标批大小：$B = 4096$
    - 缩放因子：$B/B_0 = 16$

    **(a) 线性缩放：**

    $$\eta = \eta_0 \times \frac{B}{B_0} = 0.001 \times 16 = \boxed{0.016}$$

    **(b) 平方根缩放：**

    $$\eta = \eta_0 \times \sqrt{\frac{B}{B_0}} = 0.001 \times \sqrt{16} = 0.001 \times 4 = \boxed{0.004}$$

    **该用哪个？**

    | 批大小 | 相对 $B_{\text{crit}}$ | 推荐缩放 |
    |--------|------------------------|----------|
    | 4,096 | 低于 6K（来自练习 1） | 线性（0.016） |
    | 8,192 | 高于 6K | 平方根 |
    | 16,384 | 远高于 | 常数或平方根 |

    **实践提示：** 从线性缩放（0.016）开始，但使用预热。如果训练不稳定，退回平方根（0.004）。

3. **计算效率**：批大小 512 时，训练需要 50K 步。批大小 8192 时，训练需要 6K 步。计算扩展效率 $E(8192)$。

??? success "解答"
    **已知：**

    - 基础批大小：$B_0 = 512$，步数 $S_0 = 50,000$
    - 目标批大小：$B = 8192$，步数 $S = 6,000$

    **扩展效率公式：**

    $$E(B) = \frac{S_0/S(B)}{B/B_0}$$

    **计算：**

    $$E(8192) = \frac{50,000/6,000}{8,192/512} = \frac{8.33}{16} = \boxed{0.52 = 52\%}$$

    **解读：**

    | 指标 | 值 |
    |------|-----|
    | 批大小增加 | 16× |
    | 步数减少 | 8.33× |
    | 扩展效率 | 52% |
    | “浪费”的算力 | 48% |

    **完美缩放会给出：**
    $$S_{\text{perfect}} = \frac{50,000}{16} = 3,125 \text{ steps}$$

    实际：6,000 步 → 比完美缩放多 1.92× 的步数。

    **结论：** 在批大小 8192 时，我们已远超 $B_{\text{crit}}$。几乎一半算力被“浪费”了，因为它没有减少训练步数。不过，如果墙钟时间是约束，这仍然可能值得。

4. **LARS 推导**：证明 LARS 信任比 $\phi = ||w||/||\nabla w||$ 能确保相对更新量 $||\Delta w||/||w||$ 在各层之间近似恒定。

??? success "解答"
    **LARS 更新规则：**

    $$\Delta w_l = -\eta \cdot \phi_l \cdot \nabla w_l$$

    其中信任比为：

    $$\phi_l = \frac{||w_l||}{||\nabla w_l||}$$

    （为简化忽略权重衰减）

    **相对更新幅值：**

    $$\frac{||\Delta w_l||}{||w_l||} = \frac{\eta \cdot \phi_l \cdot ||\nabla w_l||}{||w_l||}$$

    $$= \frac{\eta \cdot \frac{||w_l||}{||\nabla w_l||} \cdot ||\nabla w_l||}{||w_l||}$$

    $$= \frac{\eta \cdot ||w_l||}{||w_l||}$$

    $$= \boxed{\eta}$$

    **关键洞察：** 相对更新 $||\Delta w||/||w||$ 对**所有层**都等于 $\eta$，与以下因素无关：

    - 权重幅值 $||w_l||$
    - 梯度幅值 $||\nabla w_l||$

    **为什么这很重要：**

    | 层 | 不用 LARS | 用 LARS |
    |----|-----------|---------|
    | 小权重、大梯度 | 巨大的相对更新 | $\eta$ |
    | 大权重、小梯度 | 极小的相对更新 | $\eta$ |
    | 任意层 | $\eta \cdot ||\nabla w|| / ||w||$ | $\eta$ |

    **结果：** 所有层以相同的相对速率更新，从而避免：
    - 浅层更新过慢
    - 输出层更新过猛
    - 大批大小下的训练不稳定

    这就是 LARS 能在标准 SGD 失败的情况下支持 32K+ 批大小训练的原因。

5. **梯度累积（Gradient Accumulation）**：你有 8 块 GPU，每块批 32（共 256），但需要有效批大小 2048。需要多少累积步？如果每次前向-反向需要 100ms，all-reduce 需要 20ms，每个有效步需要多长时间？

??? success "解答"
    **已知：**

    - GPU：8
    - 每 GPU 批大小：32
    - 当前有效批大小：$8 \times 32 = 256$
    - 目标有效批大小：2,048
    - 前向-反向时间：100ms
    - All-reduce 时间：20ms

    **累积步数：**

    $$\text{Accumulation steps} = \frac{\text{Target batch}}{\text{Current batch}} = \frac{2048}{256} = \boxed{8}$$

    **每个有效步的时间分解：**

    | 阶段 | 次数 | 每次时间 | 总计 |
    |------|------|----------|------|
    | 前向-反向传播 | 8 | 100ms | 800ms |
    | All-reduce（仅在最后） | 1 | 20ms | 20ms |
    | **总计** | | | **820ms** |

    **与真正的数据并行（64 块 GPU）对比：**

    如果有 64 块 GPU，每块批 32：
    - 前向-反向：100ms
    - All-reduce：约 40ms（GPU 更多，略高）
    - 总计：约 140ms

    **效率对比：**

    $$\text{Accumulation slowdown} = \frac{820\text{ms}}{140\text{ms}} \approx 5.9\times$$

    **关键洞察：** 梯度累积用时间换内存。它适用于：
    1. 尚未达到 $B_{\text{crit}}$
    2. GPU 内存限制了批大小
    3. 没有更多 GPU 可用

6. **动态批大小**：你希望最初每步训练 1M token，逐步提升到每步 4M token。如果在训练中点切换，相比恒定每步 1M token，你少做多少次梯度更新？

??? success "解答"
    **设定：**

    设总训练 token 数为 $D$（例如 2T token）。

    **恒定批大小（每步 1M token）：**

    $$\text{Updates}_{\text{constant}} = \frac{D}{1\text{M}} = D \text{ updates (in millions)}$$

    **动态批大小：**

    - 前半段（$D/2$ token）按每步 1M token：

    $$\text{Updates}_1 = \frac{D/2}{1\text{M}} = \frac{D}{2\text{M}}$$

    - 后半段（$D/2$ token）按每步 4M token：

    $$\text{Updates}_2 = \frac{D/2}{4\text{M}} = \frac{D}{8\text{M}}$$

    - 动态更新总数：

    $$\text{Updates}_{\text{dynamic}} = \frac{D}{2\text{M}} + \frac{D}{8\text{M}} = \frac{4D + D}{8\text{M}} = \frac{5D}{8\text{M}}$$

    **减少量：**

    $$\text{Reduction} = \frac{D}{\text{M}} - \frac{5D}{8\text{M}} = \frac{8D - 5D}{8\text{M}} = \frac{3D}{8\text{M}}$$

    $$\text{Reduction fraction} = \frac{3D/8}{D} = \boxed{37.5\%}$$

    **D = 2T token 的具体例子：**

    | 策略 | 更新次数 | 墙钟时间（相对） |
    |------|----------|------------------|
    | 恒定 1M | 2,000,000 | 1.0× |
    | 动态（1M→4M） | 1,250,000 | ~0.75× |
    | 减少 | 750,000 | 快 25% |

    **等等——更新次数少了 37.5%，为什么墙钟时间只快 25%？**

    4M 批的步比 1M 批的步耗时约 2×（每步计算量更大）。但通信开销被摊薄到更多 token 上，带来了收益。

    **实际墙钟加速**取决于：
    - 通信开销占比
    - 每个批大小下的 GPU 利用率
    - 两种批大小是否都低于 $B_{\text{crit}}$

    **总结：** 动态批大小把梯度更新减少 37.5%，根据系统特性，这通常转化为约 20–30% 的墙钟加速。

## 关键要点

1. **临界批大小存在**：超过 $B_{\text{crit}}$ 后，算力被浪费。

2. **线性缩放在 $B_{\text{crit}}$ 以下有效**：学习率 $\propto$ 批大小。

3. **预热必不可少**：更大的批需要更长的预热。

4. **LARS/LAMB 支持极端扩展**：逐层自适应使 30K+ 批大小成为可能。

5. **$B_{\text{crit}}$ 在训练中增大**：动态批大小可以利用这一点。

6. **墙钟时间与算力的权衡**：更大的批更快，但在 $B_{\text{crit}}$ 以上效率更低。