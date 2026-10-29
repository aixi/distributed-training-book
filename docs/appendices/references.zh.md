---
title: "参考文献"
---

## 基础论文

- Vaswani, A., et al. (2017). **Attention Is All You Need**. *NeurIPS 2017*. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

- Kaplan, J., et al. (2020). **Scaling Laws for Neural Language Models**. *arXiv preprint*. [arXiv:2001.08361](https://arxiv.org/abs/2001.08361)

- Hoffmann, J., et al. (2022). **Training Compute-Optimal Large Language Models**. *NeurIPS 2022*. [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)

## 并行策略

- Shoeybi, M., et al. (2019). **Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism**. *arXiv preprint*. [arXiv:1909.08053](https://arxiv.org/abs/1909.08053)

- Narayanan, D., et al. (2021). **Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM**. *SC'21*. [arXiv:2104.04473](https://arxiv.org/abs/2104.04473)

- Huang, Y., et al. (2019). **GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism**. *NeurIPS 2019*. [arXiv:1811.06965](https://arxiv.org/abs/1811.06965)

- Narayanan, D., et al. (2019). **PipeDream: Generalized Pipeline Parallelism for DNN Training**. *SOSP'19*. [arXiv:1806.03377](https://arxiv.org/abs/1806.03377)

- Rajbhandari, S., et al. (2020). **ZeRO: Memory Optimizations Toward Training Trillion Parameter Models**. *SC'20*. [arXiv:1910.02054](https://arxiv.org/abs/1910.02054)

- Korthikanti, V., et al. (2022). **Reducing Activation Recomputation in Large Transformer Models**. *MLSys 2023*. [arXiv:2205.05198](https://arxiv.org/abs/2205.05198)

- Liu, H., et al. (2023). **Ring Attention with Blockwise Transformers for Near-Infinite Context**. *arXiv preprint*. [arXiv:2310.01889](https://arxiv.org/abs/2310.01889)

- Jacobs, S. A., et al. (2023). **DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models**. *arXiv preprint*. [arXiv:2309.14509](https://arxiv.org/abs/2309.14509)

## 专家混合

- Shazeer, N., et al. (2017). **Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer**. *ICLR 2017*. [arXiv:1701.06538](https://arxiv.org/abs/1701.06538)

- Fedus, W., et al. (2022). **Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity**. *JMLR*. [arXiv:2101.03961](https://arxiv.org/abs/2101.03961)

- Lepikhin, D., et al. (2021). **GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding**. *ICLR 2021*. [arXiv:2006.16668](https://arxiv.org/abs/2006.16668)

## 高效训练

- Dao, T., et al. (2022). **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness**. *NeurIPS 2022*. [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)

- Dao, T. (2023). **FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning**. *arXiv preprint*. [arXiv:2307.08691](https://arxiv.org/abs/2307.08691)

- Micikevicius, P., et al. (2017). **Mixed Precision Training**. *ICLR 2018*. [arXiv:1710.03740](https://arxiv.org/abs/1710.03740)

- Micikevicius, P., et al. (2022). **FP8 Formats for Deep Learning**. *arXiv preprint*. [arXiv:2209.05433](https://arxiv.org/abs/2209.05433)

- Chen, T., et al. (2016). **Training Deep Nets with Sublinear Memory Cost**. *arXiv preprint*. [arXiv:1604.06174](https://arxiv.org/abs/1604.06174)

- Ainslie, J., et al. (2023). **GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints**. *EMNLP 2023*. [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)

## 大规模系统

- Dubey, A., et al. (2024). **The Llama 3 Herd of Models**. *arXiv preprint*. [arXiv:2407.21783](https://arxiv.org/abs/2407.21783)

- DeepSeek-AI (2024). **DeepSeek-V3 Technical Report**. *arXiv preprint*. [arXiv:2412.19437](https://arxiv.org/abs/2412.19437)

- Jiang, A., et al. (2024). **Mixtral of Experts**. *arXiv preprint*. [arXiv:2401.04088](https://arxiv.org/abs/2401.04088)

- Jiang, A., et al. (2023). **Mistral 7B**. *arXiv preprint*. [arXiv:2310.06825](https://arxiv.org/abs/2310.06825)

- Brown, T., et al. (2020). **Language Models are Few-Shot Learners**. *NeurIPS 2020*. [arXiv:2005.14165](https://arxiv.org/abs/2005.14165)

## 通信与分布式系统

- **NCCL: The NVIDIA Collective Communication Library**. [GitHub](https://github.com/NVIDIA/nccl)

- Thakur, R., et al. (2005). **Optimization of Collective Communication Operations in MPICH**. *International Journal of High Performance Computing Applications*.

- Rabenseifner, R. (2004). **Optimization of Collective Reduction Operations**. *ICCS 2004*.

- Dean, J., et al. (2012). **Large Scale Distributed Deep Networks**. *NeurIPS 2012*.

- Recht, B., et al. (2011). **Hogwild!: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent**. *NeurIPS 2011*. [arXiv:1106.5730](https://arxiv.org/abs/1106.5730)

## 自动并行化

- Zheng, L., et al. (2022). **Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning**. *OSDI'22*. [arXiv:2201.12023](https://arxiv.org/abs/2201.12023)

- Jia, Z., et al. (2019). **Beyond Data and Model Parallelism for Deep Neural Networks**. *MLSys 2019*. [arXiv:1807.05358](https://arxiv.org/abs/1807.05358)

## 书籍与指南

- **How to Scale Your Model** (Google DeepMind/JAX team, 2025). [jax-ml.github.io/scaling-book](https://jax-ml.github.io/scaling-book/)

- **The Ultra-Scale Playbook** (Hugging Face, 2025). [huggingface.co/spaces/nanotron/ultrascale-playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook)

## 硬件参考

- NVIDIA (2022). **H100 Tensor Core GPU Architecture Whitepaper**.
- NVIDIA. **DGX H100 System Architecture Guide**.
- InfiniBand Trade Association. **InfiniBand Architecture Specification**.

## 优化与训练动态

- Goyal, P., et al. (2017). **Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour**. *arXiv preprint*. [arXiv:1706.02677](https://arxiv.org/abs/1706.02677)

- You, Y., et al. (2019). **Large Batch Optimization for Deep Learning: Training BERT in 76 Minutes**. *ICLR 2020*. [arXiv:1904.00962](https://arxiv.org/abs/1904.00962)

- McCandlish, S., et al. (2018). **An Empirical Model of Large-Batch Training**. *arXiv preprint*. [arXiv:1812.06162](https://arxiv.org/abs/1812.06162)

- Smith, S. L., et al. (2018). **Don't Decay the Learning Rate, Increase the Batch Size**. *ICLR 2018*. [arXiv:1711.00489](https://arxiv.org/abs/1711.00489)

## 梯度压缩与异步方法

- Alistarh, D., et al. (2017). **QSGD: Communication-Efficient SGD via Gradient Quantization and Encoding**. *NeurIPS 2017*. [arXiv:1610.02132](https://arxiv.org/abs/1610.02132)

- Vogels, T., et al. (2019). **PowerSGD: Practical Low-Rank Gradient Compression for Distributed Optimization**. *NeurIPS 2019*. [arXiv:1905.13727](https://arxiv.org/abs/1905.13727)

- Lin, Y., et al. (2018). **Deep Gradient Compression: Reducing the Communication Bandwidth for Distributed Training**. *ICLR 2018*. [arXiv:1712.01887](https://arxiv.org/abs/1712.01887)

- Wen, W., et al. (2017). **TernGrad: Ternary Gradients to Reduce Communication in Distributed Deep Learning**. *NeurIPS 2017*. [arXiv:1705.07878](https://arxiv.org/abs/1705.07878)

- Seide, F., et al. (2014). **1-Bit Stochastic Gradient Descent and its Application to Data-Parallel Distributed Training of Speech DNNs**. *Interspeech 2014*.

- Stich, S. U. (2018). **Local SGD Converges Fast and Communicates Little**. *ICLR 2019*. [arXiv:1805.09767](https://arxiv.org/abs/1805.09767)

- Douillard, A., et al. (2023). **DiLoCo: Distributed Low-Communication Training of Language Models**. *arXiv preprint*. [arXiv:2311.08105](https://arxiv.org/abs/2311.08105)

## 检查点

- Young, J. W. (1974). **A First Order Approximation to the Optimum Checkpoint Interval**. *Communications of the ACM*.

- Daly, J. T. (2006). **A Higher Order Estimate of the Optimum Checkpoint Interval for Restart Dumps**. *Future Generation Computer Systems*.

## 扩展与涌现能力

- Wei, J., et al. (2022). **Emergent Abilities of Large Language Models**. *TMLR*. [arXiv:2206.07682](https://arxiv.org/abs/2206.07682)

- Schaeffer, R., et al. (2023). **Are Emergent Abilities of Large Language Models a Mirage?**. *NeurIPS 2023*. [arXiv:2304.15004](https://arxiv.org/abs/2304.15004)

## 经典分布式计算

- Lamport, L. (1978). **Time, Clocks, and the Ordering of Events in a Distributed System**. *Communications of the ACM*.

- Little, J. D. C. (1961). **A Proof for the Queuing Formula: L = λW**. *Operations Research*.

- Williams, S., et al. (2009). **Roofline: An Insightful Visual Performance Model for Multicore Architectures**. *Communications of the ACM*.