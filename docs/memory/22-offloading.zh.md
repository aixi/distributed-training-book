---
title: "卸载：把内存扩展到 GPU 之外"
subtitle: "用 CPU 与 NVMe 扩展内存层次"
---

<div class="chapter-opener" markdown>
当 GPU 内存不够时，我们并不只是加更多 GPU，而是扩展内存层次。卸载（offloading）把 CPU 内存和 NVMe 存储当作 GPU 内存中更慢的层级来使用，从而让超出单块 GPU 容量的模型也能训练。
</div>

<div class="investigation-question" markdown>
**问题**：一个 175B 参数的模型，仅 fp16 参数就需要 350GB。最大的 GPU 只有 80GB。我们如何在单个节点上训练这个模型？带宽约束是什么，又如何隐藏延迟？
</div>

## 内存层次

现代训练系统拥有三层内存层次：

```
┌─────────────────────────────────────────────────────────────┐
│                    Memory Hierarchy                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────┐                                              │
│  │  GPU HBM   │  80GB, 3.35TB/s bandwidth (H100)              │
│  │  (Hot)     │  Compute happens here                        │
│  └─────┬──────┘                                              │
│        │ PCIe Gen4: 32GB/s                                   │
│        │ NVLink: 900GB/s (intra-node, H100)                   │
│        ▼                                                     │
│  ┌────────────┐                                              │
│  │ CPU DRAM   │  512GB-2TB, 200GB/s                          │
│  │  (Warm)    │  Optimizer states, gradients                 │
│  └─────┬──────┘                                              │
│        │ NVMe: 3-7GB/s per drive                             │
│        │ RAID: 10-30GB/s aggregate                           │
│        ▼                                                     │
│  ┌────────────┐                                              │
│  │   NVMe     │  10-100TB                                    │
│  │  (Cold)    │  Parameters, checkpoints                     │
│  └────────────┘                                              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 带宽分析

关键洞见：**带宽沿层次结构呈指数下降**。

| 层级 | 容量 | 带宽 | 延迟 |
|------|----------|-----------|---------|
| GPU HBM | 80GB | 3,350 GB/s | ~1μs |
| CPU DRAM | 1TB | 200 GB/s | ~100ns |
| NVMe SSD | 10TB | 7 GB/s | ~10μs |
| HDD | 100TB | 0.2 GB/s | ~10ms |

带宽比 GPU:CPU:NVMe 大致为 300:30:1。

### 卸载原则

**原则**：把*此刻*不需要的数据移到更便宜的存储上，在需要之前预取，并让传输与计算重叠。

$$\text{Effective Bandwidth} = \min(\text{Transfer Rate}, \text{Compute Time} / \text{Data Size})$$

如果我们能把传输隐藏在计算背后，有效带宽就是无穷大。

## CPU 卸载

CPU 卸载把优化器状态和梯度移到 CPU 内存中。

### 卸载什么

**适合卸载的对象**：

1. **优化器状态**：Adam 的 m 和 v（每步只访问一次）
2. **主权重**：fp32 副本（只在更新时需要）
3. **梯度**：归约之后、优化器步之前

**不适合卸载的对象**：

1. **激活值**：反向传播期间被访问（频率很高）
2. **参数**：每次前向/反向都需要（关键路径）

### ZeRO-Offload

DeepSpeed 的 ZeRO-Offload 把优化器步切分开来：

```
┌─────────────────────────────────────────────────────────────┐
│                    ZeRO-Offload                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  GPU Side:                     CPU Side:                     │
│  ┌──────────────────┐          ┌──────────────────┐          │
│  │ Forward Pass     │          │ Optimizer States │          │
│  │ (activations)    │          │ (m, v, fp32 w)   │          │
│  └────────┬─────────┘          └────────▲─────────┘          │
│           │                              │                   │
│  ┌────────▼─────────┐          ┌────────┴─────────┐          │
│  │ Backward Pass    │   ──────▶│ Optimizer Step   │          │
│  │ (gradients)      │ gradients│ (Adam update)    │          │
│  └────────┬─────────┘          └────────┬─────────┘          │
│           │                              │                   │
│  ┌────────▼─────────┐          ┌────────▼─────────┐          │
│  │ Update Weights   │◀──────── │ New Weights      │          │
│  │ (fp16 params)    │  weights │ (fp32 → fp16)    │          │
│  └──────────────────┘          └──────────────────┘          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 实现

```python
import torch
import torch.nn as nn
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
import threading
from concurrent.futures import ThreadPoolExecutor

@dataclass
class OffloadConfig:
    """Configuration for CPU offloading."""
    offload_optimizer: bool = True
    offload_gradients: bool = True
    offload_parameters: bool = False  # Only for ZeRO-3 + offload
    pin_memory: bool = True  # Use pinned memory for faster transfers
    num_threads: int = 4  # CPU threads for optimizer step
    overlap_comm: bool = True  # Overlap CPU-GPU transfers

class CPUOffloadOptimizer:
    """
    Optimizer that offloads states to CPU memory.

    Memory layout:

    - GPU: fp16 parameters, gradients (temporary)
    - CPU: fp32 master weights, Adam states (m, v)

    Update flow:
    1. Gradients computed on GPU (fp16)
    2. Gradients transferred to CPU, cast to fp32
    3. Adam update on CPU (fp32)
    4. Updated weights transferred to GPU, cast to fp16
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        config: Optional[OffloadConfig] = None
    ):
        self.model = model
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.config = config or OffloadConfig()

        self.step_count = 0

        # Create CPU copies and optimizer states
        self._init_offload_buffers()

        # Thread pool for parallel CPU operations
        self.executor = ThreadPoolExecutor(max_workers=self.config.num_threads)

        # CUDA streams for overlapped transfers
        self.d2h_stream = torch.cuda.Stream()  # Device to host
        self.h2d_stream = torch.cuda.Stream()  # Host to device

    def _init_offload_buffers(self):
        """Initialize CPU buffers for parameters and optimizer states."""
        self.cpu_params: Dict[str, torch.Tensor] = {}
        self.cpu_m: Dict[str, torch.Tensor] = {}  # First moment
        self.cpu_v: Dict[str, torch.Tensor] = {}  # Second moment
        self.gpu_params: Dict[str, nn.Parameter] = {}

        # Gradient buffers (pinned for faster transfer)
        self.cpu_grads: Dict[str, torch.Tensor] = {}

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            self.gpu_params[name] = param

            # CPU copies of parameters in fp32
            cpu_param = param.data.float().cpu()
            if self.config.pin_memory:
                cpu_param = cpu_param.pin_memory()
            self.cpu_params[name] = cpu_param

            # Initialize Adam states
            self.cpu_m[name] = torch.zeros_like(cpu_param)
            self.cpu_v[name] = torch.zeros_like(cpu_param)

            # Pinned gradient buffer
            grad_buffer = torch.empty_like(cpu_param)
            if self.config.pin_memory:
                grad_buffer = grad_buffer.pin_memory()
            self.cpu_grads[name] = grad_buffer

    def _transfer_gradients_to_cpu(self):
        """Asynchronously transfer gradients from GPU to CPU."""
        with torch.cuda.stream(self.d2h_stream):
            for name, param in self.gpu_params.items():
                if param.grad is not None:
                    # Non-blocking copy to pinned memory
                    self.cpu_grads[name].copy_(
                        param.grad.float(),
                        non_blocking=True
                    )

    def _cpu_adam_step(self, name: str):
        """Perform Adam update on CPU for a single parameter."""
        beta1, beta2 = self.betas

        param = self.cpu_params[name]
        grad = self.cpu_grads[name]
        m = self.cpu_m[name]
        v = self.cpu_v[name]

        # Bias correction
        bias_correction1 = 1 - beta1 ** self.step_count
        bias_correction2 = 1 - beta2 ** self.step_count

        # Update moments
        m.mul_(beta1).add_(grad, alpha=1 - beta1)
        v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        # Compute update
        denom = (v.sqrt() / (bias_correction2 ** 0.5)).add_(self.eps)
        step_size = self.lr / bias_correction1

        # Apply weight decay (decoupled, like AdamW)
        if self.weight_decay != 0:
            param.add_(param, alpha=-self.lr * self.weight_decay)

        # Apply update
        param.addcdiv_(m, denom, value=-step_size)

    def _transfer_weights_to_gpu(self):
        """Asynchronously transfer updated weights from CPU to GPU."""
        with torch.cuda.stream(self.h2d_stream):
            for name, gpu_param in self.gpu_params.items():
                cpu_param = self.cpu_params[name]
                # Cast back to fp16/bf16 and copy to GPU
                gpu_param.data.copy_(
                    cpu_param.to(gpu_param.dtype),
                    non_blocking=True
                )

    def step(self):
        """Perform optimization step with CPU offloading."""
        self.step_count += 1

        # Step 1: Transfer gradients to CPU (async)
        self._transfer_gradients_to_cpu()

        # Synchronize D2H stream before CPU computation
        self.d2h_stream.synchronize()

        # Step 2: Perform Adam updates on CPU (parallel across parameters)
        if self.config.num_threads > 1:
            futures = [
                self.executor.submit(self._cpu_adam_step, name)
                for name in self.cpu_params.keys()
            ]
            for future in futures:
                future.result()  # Wait for completion
        else:
            for name in self.cpu_params.keys():
                self._cpu_adam_step(name)

        # Step 3: Transfer updated weights back to GPU (async)
        self._transfer_weights_to_gpu()

        # Synchronize H2D stream (can be deferred to next forward)
        # self.h2d_stream.synchronize()

    def synchronize(self):
        """Ensure all async operations are complete."""
        self.d2h_stream.synchronize()
        self.h2d_stream.synchronize()

    def zero_grad(self):
        """Clear GPU gradients."""
        for param in self.gpu_params.values():
            if param.grad is not None:
                param.grad.zero_()
```

### 内存节省

对于使用 Adam 优化器的 fp16 训练，设 $\Psi$ 为模型参数量：

| 组件 | 不卸载 | 卸载后 |
|-----------|-----------------|--------------|
| 参数（fp16） | $2\Psi$ 字节 | $2\Psi$ 字节（GPU） |
| 梯度（fp16） | $2\Psi$ 字节 | $2\Psi$ 字节（反向传播期间的峰值）→ 0（传输之后） |
| 主权重（fp32） | $4\Psi$ 字节 | $4\Psi$ 字节（CPU） |
| Adam m（fp32） | $4\Psi$ 字节 | $4\Psi$ 字节（CPU） |
| Adam v（fp32） | $4\Psi$ 字节 | $4\Psi$ 字节（CPU） |
| **GPU 总计（稳态）** | **$16\Psi$ 字节** | **$2\Psi$ 字节** |
| **GPU 总计（峰值）** | **$16\Psi$ 字节** | **$4\Psi$ 字节** |

**内存下降**：GPU 内存降为原来的 1/4（4×）。

### 带宽需求

要让一步在时间 $T$ 内完成，设 $\Psi$ 为参数量：

$$\text{Gradient transfer}: \frac{4\Psi}{B_{\text{PCIe}}} < T_{\text{backward}}$$

$$\text{Weight transfer}: \frac{2\Psi}{B_{\text{PCIe}}} < T_{\text{forward}}$$

其中 $B_{\text{PCIe}}$ 是 PCIe 带宽（Gen4 为 32 GB/s）。

以 PCIe Gen4（32 GB/s）和 10B 模型（40GB 梯度）为例：

$$\text{Gradient transfer time} = \frac{40 \text{ GB}}{32 \text{ GB/s}} = 1.25s$$

如果反向传播耗时 2 秒，传输就可以被隐藏。

## NVMe 卸载

当 CPU 内存也不够时，NVMe 提供了下一层。

### ZeRO-Infinity

DeepSpeed 的 ZeRO-Infinity 把卸载扩展到 NVMe：

```
┌─────────────────────────────────────────────────────────────┐
│                    ZeRO-Infinity                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  GPU:                                                        │
│  ┌────────────────────────────────────────────────────┐      │
│  │ Working Set: Current layer parameters + activations│      │
│  │ Size: O(hidden_dim²) per layer                     │      │
│  └────────────────────────────────────────────────────┘      │
│        ▲                                                     │
│        │ Prefetch                                            │
│  CPU:  ▼                                                     │
│  ┌────────────────────────────────────────────────────┐      │
│  │ Prefetch Buffer: Next $N$ layers                   │      │
│  │ Optimizer State Partition (if CPU offload)         │      │
│  └────────────────────────────────────────────────────┘      │
│        ▲                                                     │
│        │ Async I/O                                           │
│  NVMe: ▼                                                     │
│  ┌────────────────────────────────────────────────────┐      │
│  │ All Parameters (sharded)                           │      │
│  │ All Optimizer States (sharded)                     │      │
│  │ Checkpoint Data                                    │      │
│  └────────────────────────────────────────────────────┘      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### NVMe 异步 I/O

标准文件 I/O 会阻塞 CPU。为了高效访问 NVMe，我们需要异步 I/O：

```python
import os
import io
import asyncio
import aiofiles
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import torch

@dataclass
class NVMeConfig:
    """Configuration for NVMe offloading."""
    nvme_path: str = "/mnt/nvme/offload"
    block_size: int = 1024 * 1024  # 1MB blocks
    aio_queue_depth: int = 8  # Concurrent I/O operations
    prefetch_depth: int = 2  # Layers to prefetch
    pin_memory: bool = True

class NVMeOffloadManager:
    """
    Manages parameter offloading to NVMe storage.

    Uses memory-mapped files and async I/O for efficient access.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[NVMeConfig] = None
    ):
        self.model = model
        self.config = config or NVMeConfig()

        # Ensure offload directory exists
        os.makedirs(self.config.nvme_path, exist_ok=True)

        # Parameter metadata
        self.param_info: Dict[str, Tuple[torch.Size, torch.dtype, int]] = {}

        # Memory-mapped files
        self.mmap_files: Dict[str, np.memmap] = {}

        # Prefetch buffers (pinned CPU memory)
        self.prefetch_buffers: Dict[str, torch.Tensor] = {}

        # Track which parameters are currently in GPU
        self.gpu_resident: set = set()

        self._init_nvme_storage()

    def _init_nvme_storage(self):
        """Initialize NVMe storage for all parameters."""
        offset = 0

        for name, param in self.model.named_parameters():
            size_bytes = param.numel() * param.element_size()

            # Store metadata
            self.param_info[name] = (param.shape, param.dtype, offset)
            offset += size_bytes

            # Create memory-mapped file
            filepath = os.path.join(self.config.nvme_path, f"{name.replace('.', '_')}.bin")
            mmap = np.memmap(
                filepath,
                dtype=self._torch_to_numpy_dtype(param.dtype),
                mode='w+',
                shape=param.shape
            )

            # Write initial values
            if param.dtype == torch.bfloat16:
                mmap[:] = param.data.cpu().view(torch.uint16).numpy()
            else:
                mmap[:] = param.data.cpu().numpy()
            mmap.flush()

            self.mmap_files[name] = mmap

            # Create pinned prefetch buffer
            buffer = torch.empty(
                param.shape,
                dtype=param.dtype,
                device='cpu'
            )
            if self.config.pin_memory:
                buffer = buffer.pin_memory()
            self.prefetch_buffers[name] = buffer

            # Clear GPU parameter (replace with placeholder)
            param.data = torch.empty(0, dtype=param.dtype, device=param.device)

    def _torch_to_numpy_dtype(self, dtype: torch.dtype) -> np.dtype:
        """Convert PyTorch dtype to NumPy dtype."""
        mapping = {
            torch.float32: np.float32,
            torch.float16: np.float16,
            torch.bfloat16: np.uint16,  # Store raw bits
            torch.int32: np.int32,
            torch.int64: np.int64,
        }
        return mapping.get(dtype, np.float32)

    async def _async_load(self, name: str) -> torch.Tensor:
        """Asynchronously load parameter from NVMe."""
        mmap = self.mmap_files[name]
        buffer = self.prefetch_buffers[name]
        shape, dtype, _ = self.param_info[name]

        # Async copy from mmap to buffer
        # In practice, use aiofiles or io_uring
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: buffer.copy_(
                torch.from_numpy(np.array(mmap)).view(dtype) if dtype == torch.bfloat16 else torch.from_numpy(np.array(mmap))
            )
        )

        return buffer

    async def _async_save(self, name: str, data: torch.Tensor):
        """Asynchronously save parameter to NVMe."""
        mmap = self.mmap_files[name]
        _, dtype, _ = self.param_info[name]

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: np.copyto(mmap, data.cpu().view(torch.uint16).numpy() if dtype == torch.bfloat16 else data.cpu().numpy())
        )
        mmap.flush()

    def prefetch(self, layer_names: List[str]):
        """Prefetch parameters for upcoming layers."""
        async def _prefetch_all():
            tasks = [self._async_load(name) for name in layer_names]
            await asyncio.gather(*tasks)

        asyncio.run(_prefetch_all())

    def load_to_gpu(self, name: str, device: torch.device) -> torch.Tensor:
        """Load parameter to GPU from prefetch buffer or NVMe."""
        if name in self.gpu_resident:
            return  # Already loaded

        buffer = self.prefetch_buffers[name]
        shape, dtype, _ = self.param_info[name]

        # Copy from pinned CPU to GPU
        gpu_tensor = buffer.to(device, non_blocking=True)

        self.gpu_resident.add(name)
        return gpu_tensor

    def offload_from_gpu(self, name: str, data: torch.Tensor):
        """Offload parameter from GPU back to NVMe."""
        if name not in self.gpu_resident:
            return

        # Async save to NVMe
        asyncio.run(self._async_save(name, data))

        self.gpu_resident.discard(name)

    def get_memory_stats(self) -> Dict[str, int]:
        """Get current memory usage statistics."""
        gpu_memory = sum(
            self.param_info[name][0].numel() *
            torch.tensor([], dtype=self.param_info[name][1]).element_size()
            for name in self.gpu_resident
        )

        total_memory = sum(
            info[0].numel() * torch.tensor([], dtype=info[1]).element_size()
            for info in self.param_info.values()
        )

        return {
            'gpu_memory': gpu_memory,
            'nvme_memory': total_memory,
            'offload_ratio': 1 - (gpu_memory / total_memory) if total_memory > 0 else 0
        }
```

### 带宽优化

NVMe 带宽十分宝贵。优化策略如下：

**1. 顺序访问**

随机 I/O 很慢。按顺序读取来组织数据：

```python
def pack_layers_sequentially(model: nn.Module, filepath: str):
    """
    Pack model parameters sequentially for optimal NVMe access.

    Instead of one file per parameter, pack all parameters
    in layer order for sequential prefetching.
    """
    layer_order = []
    total_bytes = 0

    with open(filepath, 'wb') as f:
        for name, param in model.named_parameters():
            data = param.data.cpu().numpy()
            offset = f.tell()
            f.write(data.tobytes())

            layer_order.append({
                'name': name,
                'offset': offset,
                'size': data.nbytes,
                'shape': data.shape,
                'dtype': str(data.dtype)
            })
            total_bytes += data.nbytes

    return layer_order, total_bytes
```

**2. 预读**

处理当前层的同时预取接下来的层：

```python
class LayerPrefetcher:
    """Prefetch layers ahead of computation."""

    def __init__(
        self,
        layer_order: List[str],
        prefetch_depth: int = 2
    ):
        self.layer_order = layer_order
        self.prefetch_depth = prefetch_depth
        self.current_idx = 0
        self.prefetched: Dict[str, torch.Tensor] = {}

    def get_layer(self, name: str) -> torch.Tensor:
        """Get layer, triggering prefetch of next layers."""
        # Return prefetched if available
        if name in self.prefetched:
            tensor = self.prefetched.pop(name)
        else:
            # Synchronous load (cache miss)
            tensor = self._load_layer(name)

        # Trigger prefetch for upcoming layers
        self._prefetch_upcoming()

        return tensor

    def _prefetch_upcoming(self):
        """Prefetch next layers in background."""
        for i in range(self.prefetch_depth):
            idx = self.current_idx + i + 1
            if idx < len(self.layer_order):
                name = self.layer_order[idx]
                if name not in self.prefetched:
                    # Async load
                    self.prefetched[name] = self._async_load_layer(name)

        self.current_idx += 1
```

**3. 压缩**

通过压缩减少 I/O 量：

```python
import lz4.frame

def compress_parameter(tensor: torch.Tensor) -> bytes:
    """Compress parameter for NVMe storage."""
    data = tensor.numpy().tobytes()
    return lz4.frame.compress(data, compression_level=0)  # Fast mode

def decompress_parameter(
    compressed: bytes,
    shape: Tuple[int, ...],
    dtype: np.dtype
) -> torch.Tensor:
    """Decompress parameter from NVMe."""
    data = lz4.frame.decompress(compressed)
    array = np.frombuffer(data, dtype=dtype).reshape(shape)
    return torch.from_numpy(array.copy())
```

对于 fp16 模型权重，LZ4 通常能实现 1.3–1.5× 的压缩率，且 CPU 开销很小。

## 让传输与计算重叠

高效卸载的关键：**重叠**。

### 重叠原则

```
Without overlap:
├──Transfer──┤├──Compute──┤├──Transfer──┤├──Compute──┤
             ▲            ▲             ▲            ▲
         Transfer     Compute       Transfer     Compute
           Wait                       Wait

With overlap:
├──Transfer─1─┤
     ├──Compute─0─┤
          ├──Transfer─2─┤
               ├──Compute─1─┤
                    ├──Transfer─3─┤
```

有效时间 = max(传输时间, 计算时间)，而不是两者之和。

### 双缓冲

使用两个缓冲区：一个用于当前计算，一个用于下一次传输：

```python
class DoubleBufferedLoader:
    """
    Double-buffered parameter loading for overlapped transfers.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.buffers = [None, None]  # Two GPU buffers
        self.current_buffer = 0
        self.load_stream = torch.cuda.Stream()
        self.pending_load: Optional[str] = None

    def get_current_buffer(self) -> torch.Tensor:
        """Get buffer containing current layer's parameters."""
        return self.buffers[self.current_buffer]

    def start_next_load(
        self,
        cpu_tensor: torch.Tensor,
        shape: Tuple[int, ...]
    ):
        """Start async load of next layer's parameters."""
        next_buffer = 1 - self.current_buffer

        # Allocate or reuse buffer
        if (self.buffers[next_buffer] is None or
            self.buffers[next_buffer].shape != shape):
            self.buffers[next_buffer] = torch.empty(
                shape,
                dtype=cpu_tensor.dtype,
                device=self.device
            )

        # Async copy
        with torch.cuda.stream(self.load_stream):
            self.buffers[next_buffer].copy_(cpu_tensor, non_blocking=True)

    def swap_buffers(self):
        """Swap active buffer, synchronizing pending load."""
        self.load_stream.synchronize()
        self.current_buffer = 1 - self.current_buffer
```

### 计算-通信重叠分析

对于第 $l$ 层：

- 参数大小：$P_l$ 字节
- 计算时间：$T_l^{compute}$
- 传输带宽：$B$

传输时间：$T_l^{transfer} = P_l / B$

**重叠条件**：$T_l^{transfer} \leq T_{l-1}^{compute}$

对于一个 Transformer 层：

- 参数：约 $12H^2$（$H$ 为隐藏维度）
- 计算：约 $24BSH^2$ FLOPs（$B$ 为批大小，$S$ 为序列长度）

以 PCIe Gen4（32 GB/s）和 A100（312 TFLOPs fp16）为例：

$$\frac{12H^2 \cdot 2}{32 \times 10^9} \leq \frac{24BSH^2}{312 \times 10^{12}}$$

解得：

$$BS \geq \frac{312 \times 10^{12}}{32 \times 10^9} \approx 9{,}750$$

当批大小为 1、序列长度为 2048 时，重叠**无法**实现；你需要更大的 $B \cdot S$ 或更快的互联。

## 完整的卸载系统

下面是一个把 CPU 卸载与 NVMe 卸载结合起来的完整实现：

```python
from enum import Enum
from typing import Dict, List, Optional, Tuple, Set
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from collections import OrderedDict
import threading
import queue

class OffloadTier(Enum):
    GPU = "gpu"
    CPU = "cpu"
    NVME = "nvme"

@dataclass
class TensorLocation:
    """Tracks where a tensor is stored."""
    tier: OffloadTier
    offset: int = 0  # For NVMe
    size: int = 0
    is_loading: bool = False
    is_saving: bool = False

@dataclass
class OffloadSystemConfig:
    """Configuration for the complete offloading system."""
    # Memory limits
    gpu_memory_limit: int = 40 * (1024 ** 3)  # 40GB
    cpu_memory_limit: int = 256 * (1024 ** 3)  # 256GB

    # Offloading policy
    offload_optimizer_states: bool = True
    offload_gradients: bool = True
    offload_parameters: bool = False
    parameter_offload_tier: OffloadTier = OffloadTier.CPU

    # Performance
    prefetch_count: int = 2
    pin_memory: bool = True
    num_io_threads: int = 4

    # NVMe settings
    nvme_path: str = "/mnt/nvme/offload"
    use_compression: bool = False

class UnifiedOffloadManager:
    """
    Unified manager for GPU/CPU/NVMe offloading.

    Manages the complete memory hierarchy with automatic
    tier selection and overlap optimization.
    """

    def __init__(
        self,
        model: nn.Module,
        config: OffloadSystemConfig
    ):
        self.model = model
        self.config = config

        # Memory tracking
        self.gpu_used = 0
        self.cpu_used = 0

        # Tensor locations
        self.locations: Dict[str, TensorLocation] = {}

        # Buffers for each tier
        self.gpu_buffers: Dict[str, torch.Tensor] = {}
        self.cpu_buffers: Dict[str, torch.Tensor] = {}
        self.nvme_manager: Optional[NVMeOffloadManager] = None

        # Async operation tracking
        self.pending_loads: Dict[str, threading.Event] = {}
        self.pending_saves: Dict[str, threading.Event] = {}

        # I/O thread pool
        self.io_queue = queue.Queue()
        self.io_threads: List[threading.Thread] = []

        # CUDA streams for overlapped transfers
        self.load_stream = torch.cuda.Stream()
        self.save_stream = torch.cuda.Stream()

        self._init_offloading()
        self._start_io_threads()

    def _init_offloading(self):
        """Initialize offloading based on configuration."""
        for name, param in self.model.named_parameters():
            param_bytes = param.numel() * param.element_size()

            # Determine initial tier based on memory limits
            if self.gpu_used + param_bytes <= self.config.gpu_memory_limit:
                tier = OffloadTier.GPU
                self.gpu_used += param_bytes
                self.gpu_buffers[name] = param.data
            elif self.cpu_used + param_bytes <= self.config.cpu_memory_limit:
                tier = OffloadTier.CPU
                self.cpu_used += param_bytes
                cpu_tensor = param.data.cpu()
                if self.config.pin_memory:
                    cpu_tensor = cpu_tensor.pin_memory()
                self.cpu_buffers[name] = cpu_tensor
            else:
                tier = OffloadTier.NVME
                if self.nvme_manager is None:
                    self.nvme_manager = NVMeOffloadManager(
                        self.model,
                        NVMeConfig(nvme_path=self.config.nvme_path)
                    )

            self.locations[name] = TensorLocation(
                tier=tier,
                size=param_bytes
            )

    def _start_io_threads(self):
        """Start background I/O threads."""
        for _ in range(self.config.num_io_threads):
            t = threading.Thread(target=self._io_worker, daemon=True)
            t.start()
            self.io_threads.append(t)

    def _io_worker(self):
        """Background worker for I/O operations."""
        while True:
            try:
                op, name, data, event = self.io_queue.get(timeout=1.0)

                if op == 'load':
                    self._do_load(name)
                elif op == 'save':
                    self._do_save(name, data)

                if event:
                    event.set()

            except queue.Empty:
                continue

    def _do_load(self, name: str):
        """Perform synchronous load from lower tier."""
        location = self.locations[name]

        if location.tier == OffloadTier.CPU:
            # Load from CPU to GPU
            cpu_tensor = self.cpu_buffers[name]
            with torch.cuda.stream(self.load_stream):
                gpu_tensor = cpu_tensor.cuda(non_blocking=True)
            self.load_stream.synchronize()
            self.gpu_buffers[name] = gpu_tensor

        elif location.tier == OffloadTier.NVME:
            # Load from NVMe to CPU, then GPU
            cpu_tensor = self.nvme_manager.prefetch_buffers.get(name)
            if cpu_tensor is not None:
                self.cpu_buffers[name] = cpu_tensor
                with torch.cuda.stream(self.load_stream):
                    gpu_tensor = cpu_tensor.cuda(non_blocking=True)
                self.load_stream.synchronize()
                self.gpu_buffers[name] = gpu_tensor

    def _do_save(self, name: str, data: torch.Tensor):
        """Perform synchronous save to lower tier."""
        location = self.locations[name]

        if location.tier == OffloadTier.CPU:
            # Save to CPU
            with torch.cuda.stream(self.save_stream):
                self.cpu_buffers[name].copy_(data, non_blocking=True)
            self.save_stream.synchronize()

        elif location.tier == OffloadTier.NVME:
            # Save to CPU, then NVMe
            cpu_tensor = data.cpu()
            self.cpu_buffers[name] = cpu_tensor
            if self.nvme_manager:
                self.nvme_manager.offload_from_gpu(name, cpu_tensor)

    def ensure_gpu(self, name: str) -> torch.Tensor:
        """Ensure parameter is on GPU, loading if necessary."""
        location = self.locations[name]

        if name in self.gpu_buffers:
            return self.gpu_buffers[name]

        # Wait for any pending load
        if name in self.pending_loads:
            self.pending_loads[name].wait()
            del self.pending_loads[name]
            return self.gpu_buffers[name]

        # Synchronous load
        self._do_load(name)
        return self.gpu_buffers[name]

    def prefetch(self, names: List[str]):
        """Prefetch parameters asynchronously."""
        for name in names:
            if name not in self.gpu_buffers and name not in self.pending_loads:
                event = threading.Event()
                self.pending_loads[name] = event
                self.io_queue.put(('load', name, None, event))

    def offload(self, name: str, data: torch.Tensor):
        """Offload parameter from GPU."""
        location = self.locations[name]

        if location.tier == OffloadTier.GPU:
            return  # Already on GPU, no offload needed

        event = threading.Event()
        self.io_queue.put(('save', name, data, event))

        # Remove from GPU buffers
        if name in self.gpu_buffers:
            del self.gpu_buffers[name]

    def get_memory_summary(self) -> Dict[str, Dict[str, int]]:
        """Get memory usage summary by tier."""
        summary = {
            'gpu': {'count': 0, 'bytes': 0},
            'cpu': {'count': 0, 'bytes': 0},
            'nvme': {'count': 0, 'bytes': 0},
        }

        for name, location in self.locations.items():
            tier = location.tier.value
            summary[tier]['count'] += 1
            summary[tier]['bytes'] += location.size

        return summary
```

## 性能分析

### 吞吐模型

带卸载的总步时间：

$$T_{step} = \max(T_{compute}, T_{transfer}) + T_{sync}$$

其中：

- $T_{compute}$：前向 + 反向传播时间
- $T_{transfer}$：总数据搬运时间
- $T_{sync}$：不可避免的同步开销

### 效率指标

定义卸载效率：

$$\eta_{offload} = \frac{T_{compute}}{T_{step}} = \frac{T_{compute}}{\max(T_{compute}, T_{transfer}) + T_{sync}}$$

目标：$\eta_{offload} > 0.9$（开销低于 10%）

### 何时使用哪一层

| 场景 | 推荐策略 |
|----------|---------------------|
| 模型能放进 GPU | 不卸载 |
| 模型能放进 GPU + 优化器卸载 | CPU 卸载优化器 |
| 模型放不进 GPU | ZeRO-3 + CPU 卸载 |
| 模型放不进 CPU | ZeRO-Infinity（NVMe） |

### 案例研究：单节点上的 175B 模型

配置：

- 8× A100 80GB GPU
- 2TB CPU 内存
- 10TB NVMe 阵列（RAID 0，4 块盘）

内存需求：

- 参数：350GB（fp16）
- 优化器状态：1.4TB（fp32 Adam）
- 梯度：350GB（fp16）
- 激活值：可变

策略：
1. 在 8 块 GPU 上做 ZeRO-3：350GB / 8 = 每块 GPU 44GB 参数
2. 优化器卸载到 CPU：1.4TB / 8 = 每块 GPU 那份 175GB → 放到 CPU
3. 梯度卸载：ReduceScatter 到 CPU，优化器步在 CPU 上执行
4. 激活值：检查点（无需卸载）

结果：用那些连参数都放不下的 80GB GPU 也能训练。

## 与其他技术的集成

### 卸载 + ZeRO

| ZeRO 阶段 | 卸载目标 | 收益 |
|------------|----------------|---------|
| ZeRO-1 | 优化器到 CPU | GPU 上的优化器内存 → 0 |
| ZeRO-2 | 优化器 + 梯度到 CPU | 只留参数在 GPU 上 |
| ZeRO-3 | 全部到 CPU/NVMe | GPU 内存占用最小 |

### 卸载 + 张量并行

TP 降低单 GPU 内存；卸载进一步降低：

$$\text{Memory per GPU} = \frac{\text{Total}}{TP \times \text{ZeRO Stage Factor}} + \text{Activations}$$

在 175B 模型上使用 TP=8 和 ZeRO-3：

$$= \frac{350 \text{ GB}}{8 \times 8} + \text{Activations} = 5.5 \text{ GB} + \text{Activations}$$

### 卸载 + 流水线并行

PP 天然地把内存按时序分阶段。与卸载结合：

1. 只加载当前流水线阶段所需的参数
2. 卸载空闲阶段的参数
3. 在计算期间预取下一阶段的参数

## 练习题

1. **带宽计算**：一个模型有 70B 参数（fp16）。你的系统有 PCIe Gen4（32 GB/s）。前向传播耗时 2 秒。你能完全重叠参数加载吗？需要多大的批大小？

??? success "解答"
    **参数大小：**

    $$M_{params} = 70 \times 10^9 \times 2 \text{ bytes} = 140 \text{ GB}$$

    **传输全部参数的时间：**

    $$T_{transfer} = \frac{140 \text{ GB}}{32 \text{ GB/s}} = 4.375 \text{ seconds}$$

    **能完全重叠吗？**

    不能！传输时间（4.375s）> 前向传播时间（2s）。

    $$\text{Overlap ratio} = \frac{T_{transfer}}{T_{compute}} = \frac{4.375}{2} = 2.19$$

    我们需要 2.19× 更多的计算时间来隐藏传输。

    **所需批大小：**

    计算量随批大小线性增长，因此：

    $$B_{required} = B_{current} \times \frac{T_{transfer}}{T_{compute}} = B_{current} \times 2.19$$

    如果当前前向传播（2s）对应的批大小为 $B$，我们需要：

    $$B_{new} \geq 2.19 \times B$$

    **替代方案：逐层流式加载**

    如果我们逐层流式加载（70B 模型约 80 层）：

    $$M_{layer} = \frac{140 \text{ GB}}{80} = 1.75 \text{ GB}$$

    $$T_{layer\_transfer} = \frac{1.75 \text{ GB}}{32 \text{ GB/s}} = 54.7 \text{ ms}$$

    $$T_{layer\_compute} = \frac{2000 \text{ ms}}{80} = 25 \text{ ms}$$

    逐层来看仍然受限于传输。需要双缓冲 + 2× 批大小。

    | 配置 | 批倍数 | 完全重叠？ |
    |---------------|------------------|-------------------|
    | 当前 | 1× | 否 |
    | 2× 批大小 | 2× | 几乎（80ms vs 55ms/层） |
    | 2.5× 批大小 | 2.5× | $\boxed{\text{是}}$ |

2. **内存分层**：为一个 500B 参数模型设计卸载策略，节点上有 8× 80GB GPU、2TB CPU RAM 和 20TB NVMe。计算每一层的内存需求。

??? success "解答"
    **系统容量：**

    | 层级 | 容量 | 带宽 | 延迟 |
    |------|----------|-----------|---------|
    | GPU | 8 × 80 = 640 GB | 3.35 TB/s (HBM3) | ~ns |
    | CPU | 2 TB | ~200 GB/s (DDR5) | ~100ns |
    | NVMe | 20 TB | ~28 GB/s (4× Gen4) | ~μs |

    **模型内存需求（用 AdamW 训练）：**

    $$M_{total} = 16\Psi = 16 \times 500 \times 10^9 = 8 \text{ TB}$$

    | 组件 | 大小 | 公式 |
    |-----------|------|---------|
    | 参数（fp16） | 1 TB | $2\Psi$ |
    | 梯度（fp16） | 1 TB | $2\Psi$ |
    | 优化器（fp32） | 6 TB | $12\Psi$ |
    | **总计** | **8 TB** | $16\Psi$ |

    **分层策略：**

    ```
    ┌─────────────────────────────────────────────────────────┐
    │  GPU (640 GB total, ~80 GB per GPU)                     │
    │  ├── Active layer parameters: ~12 GB (2 layers)         │
    │  ├── Active gradients: ~12 GB                           │
    │  ├── Activation memory: ~40 GB                          │
    │  └── Working buffers: ~16 GB                            │
    ├─────────────────────────────────────────────────────────┤
    │  CPU RAM (2 TB)                                         │
    │  ├── Full parameters (prefetch buffer): 1 TB            │
    │  ├── Full gradients (accumulation buffer): 1 TB         │
    │  └── Reserved for OS/framework: 200 GB                  │
    ├─────────────────────────────────────────────────────────┤
    │  NVMe (20 TB)                                           │
    │  ├── Optimizer states: 6 TB                             │
    │  ├── Checkpoint storage: ~10 TB                         │
    │  └── Spare capacity: ~4 TB                              │
    └─────────────────────────────────────────────────────────┘
    ```

    **训练期间的数据流：**

    1. **前向传播**：参数流式经过 GPU ← CPU ← NVMe
    2. **反向传播**：计算梯度，在 CPU 中累积
    3. **优化器步**：在 NVMe 上（分块）更新，再应用到参数上

    **时序分析（每步）：**

    | 操作 | 数据大小 | 带宽 | 时间 |
    |-----------|-----------|-----------|------|
    | 参数：NVMe → CPU | 1 TB | 28 GB/s | 36s |
    | 参数：CPU → GPU | 1 TB | 32 GB/s | 31s |
    | 梯度：GPU → CPU | 1 TB | 32 GB/s | 31s |
    | 优化器更新（NVMe） | 6 TB | 28 GB/s | 214s |

    **优化：让一切重叠**

    配合恰当的预取与流式加载：

    $$T_{step} \approx \max(T_{compute}, T_{transfer})$$

    | 配置 | 可行？ |
    |---------------|------------|
    | 全部放 GPU | 否（需要 8 TB，只有 640 GB） |
    | 仅 GPU + CPU | 否（需要 8 TB，只有 2.6 TB） |
    | GPU + CPU + NVMe | $\boxed{\text{是}}$ |

3. **重叠效率**：如果计算时间为 100ms、传输时间为 80ms、同步开销为 5ms，卸载效率是多少？你会如何改进？

??? success "解答"
    **卸载效率定义：**

    $$\eta_{offload} = \frac{T_{compute}}{T_{total}}$$

    其中 $T_{total}$ 是每步的实际墙钟时间。

    **情况 1：不重叠（串行）**

    $$T_{total}^{seq} = T_{compute} + T_{transfer} + T_{sync} = 100 + 80 + 5 = 185 \text{ ms}$$

    $$\eta_{offload}^{seq} = \frac{100}{185} = 54\%$$

    **情况 2：完全重叠（流水化）**

    在完美重叠下，传输发生在计算期间：

    $$T_{total}^{overlap} = \max(T_{compute}, T_{transfer}) + T_{sync}$$

    $$T_{total}^{overlap} = \max(100, 80) + 5 = 105 \text{ ms}$$

    $$\eta_{offload}^{overlap} = \frac{100}{105} = \boxed{95.2\%}$$

    **改进策略：**

    | 策略 | 效果 | 新效率 |
    |----------|--------|----------------|
    | 异步传输 | 把 80ms 藏在 100ms 之后 | 95.2% |
    | 降低同步开销 | 5ms → 1ms | 99.0% |
    | 更大的批 | 增大 $T_{compute}$ | 更高 |
    | 锁页内存（pinned memory） | 降低同步开销 | ~98% |
    | 双缓冲 | 启用完全重叠 | 95.2% |

    **实现最大重叠：**

    ```python
    class OverlappedOffloader:
        def __init__(self, model, device):
            self.stream_compute = torch.cuda.Stream()
            self.stream_transfer = torch.cuda.Stream()
            self.buffers = [None, None]  # Double buffer

        def step(self, layer_idx, input_tensor):
            current_buf = layer_idx % 2
            next_buf = (layer_idx + 1) % 2

            # Prefetch next layer (overlapped with compute)
            with torch.cuda.stream(self.stream_transfer):
                self.buffers[next_buf] = self.load_layer(layer_idx + 1)

            # Compute on current layer
            with torch.cuda.stream(self.stream_compute):
                output = self.forward_layer(layer_idx, input_tensor,
                                           self.buffers[current_buf])

            # Minimal sync point
            self.stream_compute.synchronize()
            return output
    ```

    **降低同步开销：**

    5ms 的同步开销可以通过以下方式降低：

    1. **CUDA 事件**取代流同步：约 0.1ms
    2. **锁页内存**：消除缺页开销
    3. **减少同步点**：每 $N$ 层同步一次，而不是每层都同步

4. **双缓冲**：为参数加载实现三缓冲。什么情况下它比双缓冲更有优势？

??? success "解答"
    **三缓冲实现：**

    ```python
    import torch
    from collections import deque
    from typing import Optional, List, Callable

    class TripleBufferedLoader:
        """
        Triple buffering for parameter loading.

        Maintains 3 GPU buffers:
        - Buffer 0: Currently being used for compute
        - Buffer 1: Next layer (ready for compute)
        - Buffer 2: Being filled from CPU/NVMe
        """

        def __init__(
            self,
            layer_size: int,
            dtype: torch.dtype = torch.float16,
            device: str = "cuda"
        ):
            # Allocate 3 GPU buffers
            self.buffers = [
                torch.empty(layer_size, dtype=dtype, device=device)
                for _ in range(3)
            ]

            # Transfer streams
            self.transfer_streams = [
                torch.cuda.Stream() for _ in range(2)
            ]
            self.compute_stream = torch.cuda.Stream()

            # Buffer state tracking
            self.buffer_ready = [False, False, False]
            self.current_transfer_stream = 0

        def prefetch_layer(
            self,
            layer_idx: int,
            buffer_idx: int,
            source_tensor: torch.Tensor
        ):
            """Async prefetch a layer into buffer."""
            stream = self.transfer_streams[self.current_transfer_stream]
            self.current_transfer_stream = 1 - self.current_transfer_stream

            with torch.cuda.stream(stream):
                self.buffers[buffer_idx].copy_(source_tensor, non_blocking=True)

            self.buffer_ready[buffer_idx] = True
            return stream

        def forward_with_triple_buffer(
            self,
            layers: List[Callable],
            cpu_params: List[torch.Tensor],
            input_tensor: torch.Tensor
        ) -> torch.Tensor:
            """
            Execute forward pass with triple buffering.
            """
            num_layers = len(layers)
            x = input_tensor

            # Initial prefetch: load first 2 layers
            events = [None, None]
            events[0] = self.prefetch_layer(0, 0, cpu_params[0])
            if num_layers > 1:
                events[1] = self.prefetch_layer(1, 1, cpu_params[1])

            for i in range(num_layers):
                compute_buf = i % 3
                prefetch_buf = (i + 2) % 3

                # Wait for current layer's transfer
                if events[i % 2] is not None:
                    events[i % 2].synchronize()

                # Start prefetching layer i+2
                if i + 2 < num_layers:
                    events[i % 2] = self.prefetch_layer(
                        i + 2, prefetch_buf, cpu_params[i + 2]
                    )

                # Compute layer i
                with torch.cuda.stream(self.compute_stream):
                    x = layers[i](x, self.buffers[compute_buf])

            self.compute_stream.synchronize()
            return x
    ```

    **三缓冲何时有优势？**

    | 场景 | 双缓冲 | 三缓冲 | 胜者 |
    |----------|---------------|---------------|--------|
    | $T_{transfer} < T_{compute}$ | 完全重叠 | 无收益 | 平手 |
    | $T_{transfer} = T_{compute}$ | 完全重叠 | 无收益 | 平手 |
    | $T_{transfer} > T_{compute}$ | 出现计算空档 | 流水更平滑 | 三缓冲 |
    | 传输时间可变 | 可能停顿 | 额外缓冲吸收抖动 | 三缓冲 |
    | 高抖动/竞争 | 遇延迟就停顿 | 容忍延迟 | 三缓冲 |

    **数学分析：**

    设 $T_c$ = 计算时间，$T_t$ = 传输时间。

    **双缓冲：**

    $$T_{step}^{double} = \max(T_c, T_t)$$

    当第 $i$ 层完成时若第 $i+1$ 层的传输尚未完成，就会停顿。

    **三缓冲：**

    $$T_{step}^{triple} = \max(T_c, \frac{T_t}{2})$$

    有两个传输在途时，我们实际上拥有两倍的传输带宽。

    **三缓冲何时有效：**

    $$T_t > T_c \text{ (transfer-bound scenario)}$$

    **示例：**

    | 指标 | 双缓冲 | 三缓冲 |
    |--------|---------------|---------------|
    | $T_c = 50$ ms | | |
    | $T_t = 80$ ms | | |
    | 每层时间 | 80 ms | 50 ms |
    | 加速比 | 1× | $\boxed{1.6\times}$ |

    三缓冲在传输时间超过计算时间时最有价值，它让流水线保持受限于计算，而不是受限于传输。

5. **压缩权衡**：LZ4 以 2GB/s 的压缩吞吐实现 1.5× 压缩。对于 NVMe 上（7GB/s）的 10GB 张量，压缩值得吗？计算压缩与不压缩的总时间。

??? success "解答"
    **不压缩：**

    $$T_{no\_compress} = \frac{10 \text{ GB}}{7 \text{ GB/s}} = 1.43 \text{ seconds}$$

    **压缩：**

    压缩后大小：

    $$M_{compressed} = \frac{10 \text{ GB}}{1.5} = 6.67 \text{ GB}$$

    压缩时间（在 CPU 上）：

    $$T_{compress} = \frac{10 \text{ GB}}{2 \text{ GB/s}} = 5.0 \text{ seconds}$$

    传输时间：

    $$T_{transfer} = \frac{6.67 \text{ GB}}{7 \text{ GB/s}} = 0.95 \text{ seconds}$$

    压缩后的总时间：

    $$T_{with\_compress} = T_{compress} + T_{transfer} = 5.0 + 0.95 = 5.95 \text{ seconds}$$

    **比较：**

    | 方案 | 时间 | 加速比 |
    |----------|------|---------|
    | 不压缩 | 1.43s | 1× |
    | 压缩 | 5.95s | 0.24×（更慢！） |

    $$\boxed{\text{Compression is NOT worth it}}$$

    **盈亏平衡分析：**

    当满足下式时，压缩才有利：

    $$T_{compress} + T_{transfer}^{compressed} < T_{transfer}^{uncompressed}$$

    $$\frac{M}{C_{throughput}} + \frac{M/R}{B_{NVMe}} < \frac{M}{B_{NVMe}}$$

    其中 $R$ = 压缩比，$C$ = 压缩吞吐。

    解出压缩有益的条件：

    $$\frac{1}{C_{throughput}} < \frac{1}{B_{NVMe}} \times (1 - \frac{1}{R})$$

    $$C_{throughput} > \frac{B_{NVMe}}{1 - 1/R}$$

    对本例（$R = 1.5$，$B = 7$ GB/s）：

    $$C_{required} > \frac{7}{1 - 1/1.5} = \frac{7}{0.33} = 21 \text{ GB/s}$$

    我们需要压缩吞吐 > 21 GB/s，而 LZ4 只提供 2 GB/s。

    **压缩确实有利的场景：**

    | 场景 | 原因 |
    |----------|-----|
    | 极慢的存储（HDD ~200 MB/s） | 传输占主导 |
    | 高压缩比（>10×） | 体积显著缩小 |
    | GPU 加速压缩 | 快到可以重叠 |
    | 稀疏张量 | 可实现极端压缩 |

    **GPU 加速压缩示例：**

    使用 nvCOMP（GPU LZ4，约 50 GB/s）：

    $$T_{compress}^{GPU} = \frac{10}{50} = 0.2 \text{s}$$

    $$T_{total}^{GPU} = 0.2 + 0.95 = 1.15 \text{s} < 1.43\text{s}$$

    GPU 压缩能带来 $\boxed{1.24\times}$ 的加速。

6. **预取深度**：给定层计算时间 $T_c$、传输带宽 $B$ 和层大小 $S$，推导最优预取深度。什么时候增加预取深度不再有帮助？

??? success "解答"
    **定义：**

    - $T_c$ = 每层计算时间
    - $B$ = 传输带宽（字节/秒）
    - $S$ = 层大小（字节）
    - $k$ = 预取深度（提前预取的层数）

    **每层传输时间：**

    $$T_t = \frac{S}{B}$$

    **最优预取深度的推导：**

    要做到不停顿，我们需要在被需要之前有足够时间传输下一层：

    $$k \times T_c \geq T_t$$

    解出最小 $k$：

    $$k^* = \left\lceil \frac{T_t}{T_c} \right\rceil = \left\lceil \frac{S}{B \times T_c} \right\rceil$$

    $$\boxed{k^* = \left\lceil \frac{S}{B \cdot T_c} \right\rceil}$$

    **示例计算：**

    | 参数 | 值 |
    |-----------|-------|
    | 层大小 $S$ | 2 GB |
    | PCIe 带宽 $B$ | 32 GB/s |
    | 计算时间 $T_c$ | 50 ms |

    $$T_t = \frac{2}{32} = 62.5 \text{ ms}$$

    $$k^* = \left\lceil \frac{62.5}{50} \right\rceil = \left\lceil 1.25 \right\rceil = 2$$

    需要预取深度为 2 才能避免停顿。

    **增加预取深度不再有帮助的场景：**

    | 场景 | 原因 |
    |----------|-----|
    | 已经受限于计算（$T_t < T_c$） | $k=1$ 就足够 |
    | 受限于内存 | 无法分配更多缓冲区 |
    | 收益递减 | 超过 $k^*$ 后没有改善 |
    | 层大小可变 | 可能需要自适应 $k$ |

    **收益递减的数学分析：**

    有效吞吐作为 $k$ 的函数：

    $$\text{Throughput}(k) = \begin{cases}
    \frac{1}{T_t / k} & \text{if } k < k^* \text{ (transfer-bound)} \\
    \frac{1}{T_c} & \text{if } k \geq k^* \text{ (compute-bound)}
    \end{cases}$$

    一旦 $k \geq k^*$，我们就受限于计算，更多预取并无帮助。

    **内存代价：**

    $$M_{prefetch} = (k + 1) \times S$$

    即 $k$ 个预取缓冲区加一个计算缓冲区带来的额外内存。

    | 预取深度 | 吞吐 | 内存代价 |
    |----------------|------------|-------------|
    | $k = 0$ | $\frac{1}{T_c + T_t}$ | $S$ |
    | $k = 1$ | $\frac{1}{\max(T_c, T_t)}$ | $2S$ |
    | $k = k^*$ | $\frac{1}{T_c}$ | $(k^*+1)S$ |
    | $k > k^*$ | $\frac{1}{T_c}$（无增益） | 浪费内存 |

    **自适应预取策略：**

    ```python
    def compute_optimal_prefetch(
        layer_sizes: List[int],
        bandwidth: float,
        compute_times: List[float]
    ) -> List[int]:
        """
        Compute per-layer optimal prefetch depth.
        Handles variable layer sizes and compute times.
        """
        prefetch_depths = []

        for size, t_compute in zip(layer_sizes, compute_times):
            t_transfer = size / bandwidth
            k_optimal = max(1, int(np.ceil(t_transfer / t_compute)))
            prefetch_depths.append(k_optimal)

        return prefetch_depths
    ```

    **关键洞见：** 最优预取量恰好足以隐藏传输延迟——再多只会浪费内存，并不会提升吞吐。

## 关键要点

1. **内存是分层的**：GPU → CPU → NVMe，带宽递减、容量递增。

2. **重叠至关重要**：没有重叠，卸载会毁掉性能；有了重叠，它几乎免费。

3. **锁页内存很重要**：锁页（page-locked）的 CPU 内存才能支持异步 GPU 传输。

4. **预取隐藏延迟**：计算当前层的同时加载下一层。

5. **ZeRO + 卸载威力强大**：它把跨 GPU 的内存削减与通过 CPU/NVMe 的内存扩展结合起来。

6. **了解你的带宽**：PCIe（32 GB/s）、NVMe（7 GB/s）。据此设计。

7. **压缩有时有用**：对于配合 GPU 加速压缩（如 nvCOMP）的 NVMe，压缩能提升有效带宽。基于 CPU 的压缩（如 LZ4 约 2 GB/s）通常比不压缩的 NVMe（约 7 GB/s）更慢。一定要针对你的具体硬件做基准测试。