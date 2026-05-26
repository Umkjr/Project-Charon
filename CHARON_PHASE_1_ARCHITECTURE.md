# Project Charon
**Predictive KV-Cache Arbitrage & VRAM Eviction Engine**
*Phase 1 Prototyping Blueprint: Single-Node Architecture*

## Executive Summary
Project Charon is a deep-tech C++ memory scheduler designed to sit inside open-source inference servers (vLLM). By utilizing a Hidden Markov Model (HMM), Charon treats API traffic as a market microstructure, predicting when high-frequency multi-agent loops are active and preemptively ferrying their KV-Cache into hot VRAM via the PCIe bus before the request arrives.

**Objective:** Mathematically prove that predictive memory arbitrage increases concurrent token throughput by 15%+ on heavily constrained hardware, effectively masking PCIe transfer latency.

---

## Prototyping Phases

### Phase 1.1: The Local Sandbox Constraint
Before writing the C++ hooks, the testing environment must be ruthlessly constrained to force the memory wall.
* **Target Hardware:** RTX 3070 Ti (8GB VRAM).
* **Model Selection:** Qwen 1.5B (Quantized to INT4).
* **VRAM Allocation:** Restrict the model weights to consume exactly 3GB of VRAM, leaving a hard ceiling of ~4GB strictly for the KV-Cache.
* **Goal:** Create an environment that will violently Out-Of-Memory (OOM) or throttle when 5 concurrent orchestration agents hit it simultaneously under standard vLLM logic.

### Phase 1.2: The C++ HMM Oracle (`predictive_core.cpp`)
Build the standalone, zero-dependency mathematical engine that calculates the traffic regime.
* **Input Parameters:** `session_id`, `timestamp_ms`, `payload_size`.
* **State Definitions:** * `S1` (Human / Low Frequency)
  * `S2` (Agent / High Frequency)
* **The Math:** Calculate the Bayesian update for transition probability $P(X_{t+1} = S_2 \mid X_t = S_1, O_t)$.
* **Output:** A sorted priority queue of active `session_id`s ranked by their probability of striking the server in the next 100ms.

### Phase 1.3: The Python/Pybind11 Intercept
Wrap the C++ HMM Oracle and inject it into vLLM’s existing Python memory allocator.
* **Compilation:** Use `pybind11` to compile `charon_engine.so`.
* **The Hook:** Locate `vllm/core/block_manager/cpu_gpu_block_allocator.py`.
* **The Override:** Intercept the `can_swap_out` and `swap_out` methods. Instead of waiting for the VRAM to hit 100% capacity to trigger a reactive swap, script a polling loop that asks the Charon Oracle for preemptive eviction targets every 50ms.

### Phase 1.4: Asynchronous PCIe Execution
The core proprietary advantage. Moving the memory without pausing the GPU's compute cores.
* **CUDA Streams:** Hijack vLLM's `csrc/cache_kernels.cu` to assign Charon its own dedicated, non-blocking CUDA stream.
* **The Trade Execution:** When the Oracle predicts an agent loop is incoming, execute a background `cudaMemcpyAsync`. 
* **The Swap:** Move the lowest-probability Human KV-Cache to system RAM, and pull the highest-probability Agent KV-Cache into the hot VRAM.

### Phase 1.5: The Antigravity Crucible (Benchmarking)
Use Google Antigravity to orchestrate a highly controlled, simulated production environment that proves the engine's value.
* **The Baseline:** Run standard vLLM.
* **The Traffic:** Deploy 1 "Human" agent script (sending requests with 30-second delays) and 4 "High-Frequency Bot" agents (firing massive context windows back-to-back).
* **The Metrics to Record:** 1. Average Time-To-First-Token (TTFT).
  2. Number of concurrent sessions sustained before VRAM OOM.
  3. Total tokens generated per second across the GPU.
* **The Win State:** Charon successfully manages all 5 agents on the 8GB GPU without crashing, logging a >15% lower TTFT for the High-Frequency Bots compared to the baseline vLLM run.

---

## Technical Debt & IP Protection Strategy
* **Zero Dependencies:** The core `predictive_core.cpp` must never include PyTorch or vLLM headers. It must remain a pure mathematical calculation.
* **Graceful Degradation:** If the Charon binary fails to initialize (or the DRM license fails), the system must instantly fall back to standard vLLM FIFO reactive scheduling without crashing the host server.