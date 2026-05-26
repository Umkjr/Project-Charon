# Project Charon
**Hierarchical Tiered Cache & Dynamic KV Quantization Engine**
*Phase 2 Prototyping Blueprint: Storage Disaggregation & Compression*

## Executive Summary
While Phase 1 successfully masks PCIe latency via HMM-driven preemptive swapping, it is still bound by the physical capacity of system RAM. Phase 2 breaks the memory wall entirely. By integrating on-the-fly 3-bit vector quantization and a 3-tier storage lifecycle (VRAM $\leftrightarrow$ DRAM $\leftrightarrow$ NVMe SSD), Charon can expand a GPU's effective context window capacity by up to 10x without requiring additional RAM hardware.

**Objective:** Mathematically prove that 3-bit Product Quantization (PQ) combined with asynchronous `io_uring` NVMe storage can sustain high-frequency multi-agent execution with zero measurable degradation in output quality.

---

## Prototyping Phases

### Phase 2.1: The Compression Engine (Triton / CUDA)
Standard 16-bit (FP16/BF16) KV-cache scales linearly and exhausts VRAM rapidly. We will build custom Triton kernels to apply dynamic vector quantization on the fly.
* **The Method:** Implement a PolarQuant/TurboQuant style rotation-based coordinate transform. 
* **The Math:** Convert the high-dimensional KV vectors from Cartesian to polar coordinates (separating magnitude and angle). Because angular distributions are predictable, we apply 3-bit scalar quantization with zero overhead for normalization constants.
* **The Execution (`charon_quant.triton`):**
    * Intercept the attention layer output during the forward pass.
    * Crush the generated $K$ and $V$ vectors down to 3 bits per coordinate (a ~5.3x memory reduction).
    * Store the compressed representation natively in the PagedAttention blocks.

### Phase 2.2: Just-In-Time (JIT) Dequantization
To maintain rapid Time-To-First-Token (TTFT), the vectors cannot be decompressed back in global memory.
* **The Fusion:** Write a fused FlashAttention-style kernel. 
* **The Pipeline:** The 3-bit vectors are pulled from VRAM into the GPU's ultra-fast SRAM. They are instantly exploded back into FP16/BF16 inside the registers strictly for the microsecond required to compute the $QK^T$ matrix multiplication, minimizing the global memory bandwidth footprint.

### Phase 2.3: The 3-Tier Storage Lifecycle
Expand the Phase 1 `predictive_core.cpp` HMM Oracle to manage a three-state transition matrix instead of two.

**1. Hot Tier (GPU VRAM)**
* *Criteria:* `TrafficRegime::AGENT_CONCURRENT` (High-frequency, sub-800ms API calls).
* *Storage:* 3-bit compressed KV vectors. 

**2. Warm Tier (System DRAM)**
* *Criteria:* `TrafficRegime::HUMAN_INTERACTIVE` (Low-frequency, paused interactions).
* *Transfer Mechanism:* Standard `cudaMemcpyAsync` over the PCIe bus, managed by the Phase 1 polling hook.

**3. Cold Tier (Gen5 NVMe SSD)**
* *Criteria:* `TrafficRegime::DORMANT` (No requests received in > 5 minutes).
* *Transfer Mechanism:* Serialize the compressed blocks and write them to the local SSD. 

### Phase 2.4: Asynchronous SSD I/O (`io_uring`)
Standard Python file I/O will block the CPU and bottleneck the prefetch. We must write C++ bindings to write directly to the SSD bypassing the kernel page-cache.
* **The Implementation:** Utilize Linux `io_uring` (or GPU Direct Storage if enterprise drivers are available). 
* **The Workflow:** When the HMM Oracle detects an incoming request for a `DORMANT` session, it issues an asynchronous `io_uring` read command to pull the serialized KV blocks directly from the SSD to System RAM, and immediately triggers the Phase 1 CUDA stream to push them into VRAM.

### Phase 2.5: The Antigravity Crucible (Phase 2 Benchmarking)
* **The Baseline:** Run standard vLLM on the 8GB RTX 3070 Ti using a 128K context window prompt. Watch it immediately OOM.
* **The Test:** Deploy the custom Triton kernels and the 3-Tier engine. Slam the server with 10 concurrent agents utilizing massive context histories. 
* **The Metrics to Record:**
    1. Maximum Sustained Context Length before VRAM exhaustion.
    2. Read/Write bandwidth saturation on the local NVMe SSD.
    3. Accuracy/Perplexity degradation (ensuring the 3-bit compression does not destroy the agent's logic).

---

## Technical Debt & Architecture Isolation
* **Triton Modularity:** The compression kernels must be designed as drop-in replacements for standard attention layers. They should be packaged as an independent `.whl` extension to avoid hard-forking the vLLM core engine more than necessary.
* **Wear Leveling:** NVMe SSDs have finite write cycles (TBW). The HMM Oracle must implement a strict admission control policy—only evicting blocks to the SSD if the session is predicted to remain dormant for extended periods, avoiding SSD wear from rapid thrashing.