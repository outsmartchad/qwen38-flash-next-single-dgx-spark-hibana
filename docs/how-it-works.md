# How Qwen3.8 Flash Next and Hibana work

[Setup](setup.md) · [Troubleshooting](troubleshooting.md) · [Measurement method and raw results](benchmarking.md)

## How the model works

Flash Next combines a sparsely activated language backbone with an unusually large lookup table. That makes it an interesting single-Spark target. The router limits the feed-forward work done for each token, and the runtime can serve lookup rows from host memory as needed. The original model also has a vision encoder. Both Hibana runs in this recipe used the text path. [Qwen model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next#model-overview) · [Qwen technical report](https://arxiv.org/html/2608.30320v1#S2) · [Exact runtime record](../results/exact.server.json) · [Fast runtime record](../results/fast.server.json)

### Parameters and token flow

Qwen reports a 125B language backbone with about 6B parameters active per token, **plus** 51B parameters in n-gram embeddings and 4B in a one-layer MTP module. The main stack has 48 layers of width 2,560. Each mixture-of-experts (MoE) block routes a token to 10 of 512 learned feed-forward experts and also runs a shared expert. A token therefore makes 480 routed expert selections across the main stack. Those are selections over layers, so the same expert index in two layers points to different weights. The runtime needs access to the full expert bank because different tokens select different subsets. [Model specifications](https://huggingface.co/Qwen/Qwen3.8-Flash-Next#model-overview) · [Pinned RadixArk model config](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4/blob/7b719225242aacd3dbd3f9407468c2ee9a9d2594/config.json)

```text
Chat template → token IDs → token embeddings
                               ↓
             early n-gram lookup and gated feature injection
                               ↓
      12 groups of [GDN → MoE] ×3, then [QSA → MoE]
              carried by a four-branch residual stream
                               ↓
        output head → next-token logits → selected token
                               └─ hidden state also feeds the MTP proposer
```

This diagram shows the text model's data flow. The n-gram lookup happens inside the early stack, and the residual branches carry layer inputs and outputs around the attention and MoE sublayers. [Qwen architecture and ablations](https://arxiv.org/html/2608.30320v1#S2) · [Pinned Hibana model implementation](https://github.com/sharar-project/hibana/blob/d474b74d5a892b2cd1cc8fc50cb17946e2a539bb/docs/flashnext.md#the-model-as-served)

### Recurrent state and sparse attention

- **Gated DeltaNet (GDN)** handles 36 of the 48 layers. It updates a fixed-size state per sequence. A learned gate controls how much old state survives, and a delta update corrects what that state associates with the new key. Decode can update this state incrementally, and chunkwise implementations process a prefill block in parallel. [Gated Delta Networks](https://arxiv.org/abs/2412.06464) · [Qwen's GDN design](https://arxiv.org/html/2608.30320v1#S2.SS1.SSS1)
- **Qwen Sparse Attention (QSA)** handles the other 12 layers. A lightweight indexer scores four-token micro-blocks of the retained context. The long-context path selects up to 512 completed blocks, representing 2,048 positions, and also considers the current partial block. The selected blocks can come from earlier parts of the conversation. The server still stores past keys and values for future selection. [Qwen's QSA design](https://arxiv.org/html/2608.30320v1#S2.SS1.SSS2) · [Hibana sparse attention path](https://github.com/sharar-project/hibana/blob/d474b74d5a892b2cd1cc8fc50cb17946e2a539bb/docs/flashnext.md#sparse-attention-past-2048-tokens)
- **Gated Residual** widens the residual stream to four branches with learned, input-dependent reads and per-branch writes. The n-gram table adds another source of features. Recent token IDs form hashed bigram and trigram indices, and a gate injects the returned features into an early layer. The size of this learned table increases storage capacity while each token reads only its selected rows. [Gated Residual and n-gram embeddings](https://arxiv.org/html/2608.30320v1#S2)

### Why an MTP head helps decode

Autoregressive decode makes each new token depend on the preceding tokens. Flash Next's multi-token prediction (MTP) module uses the recent hidden state to propose upcoming tokens at lower cost than running the entire model for each proposal. The target then evaluates a block of proposals together. When a draft agrees with the verification rule, the engine can commit several tokens from one target pass. Rejected proposals require a target token and state correction. The speedup depends on the accepted length, propose and verify cost, and request batch. [Qwen model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next#model-overview) · [Multi-token prediction research](https://arxiv.org/abs/2404.19737) · [Speculative decoding algorithm](https://arxiv.org/abs/2211.17192) · [Hibana's MTP path](https://github.com/sharar-project/hibana/blob/d474b74d5a892b2cd1cc8fc50cb17946e2a539bb/docs/flashnext.md#the-mtp-draft)

## How this recipe runs it on one DGX Spark

### Build, weights and request path

The launcher builds Hibana 0.1.0 at `d474b74` in an ARM64 CUDA container. It stages the RadixArk base checkpoint with NVFP4 routed experts and an FP8 n-gram table. Fast also stages a separately calibrated W3 overlay. NVFP4 uses 4-bit values grouped with higher precision scales, so its packed size also includes the scales. The Docker image provides the build tools and locked Python dependencies. Serving mounts the weight cache read-only, uses offline Hub mode, and binds the API to `127.0.0.1:18904`. [Dockerfile](../Dockerfile) · [Launcher](../bin/recipe.sh) · [Pinned RadixArk checkpoint](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4/tree/7b719225242aacd3dbd3f9407468c2ee9a9d2594) · [Pinned W3 overlay](https://huggingface.co/yemara/Qwen3.8-Flash-Next-NVFP4-sharar-w3-qad/tree/5c4c62a62853caebea8cedda7fbed952efa1effe) · [NVIDIA's NVFP4 format](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)

```text
Spark client → localhost chat API → tokenization and scheduler
                                          ↓
                  chunked prefill → KV pages and recurrent state slots
                                          ↓
                 MTP proposals → GB10 target verification → stream
                      ↑                ↑
       mmap-backed n-gram rows     resident model weights
```

The Grace CPU and Blackwell GPU share the Spark's advertised 128 GB of physical memory. On the pinned engine's single-GPU path, the n-gram table is memory mapped from the checkpoint. Hibana gathers the rows it needs and prefetches for later prompt chunks. The OS can keep frequently touched rows in the page cache. The SSD tier for **saved inference cache blocks** is disabled by the recipe, while the checkpoint files still live on storage. [DGX Spark specifications](https://www.nvidia.com/en-us/products/workstations/dgx-spark/#specifications) · [Pinned PLE implementation](https://github.com/sharar-project/hibana/blob/d474b74d5a892b2cd1cc8fc50cb17946e2a539bb/hibana/model/flashnext/ple.py) · [Hibana operations](https://github.com/sharar-project/hibana/blob/d474b74d5a892b2cd1cc8fc50cb17946e2a539bb/docs/operations.md#serving-profiles) · [Recipe launcher](../bin/recipe.sh)

The API hands ready requests to the scheduler. For decode and verify, Hibana can replay captured CUDA graphs for supported batch shapes, reducing CPU kernel-launch overhead. Its sparse path scores and assembles the selected KV pages on the device inside the graph. The retained KV pages also carry the indexer data used by QSA. For a cached prefix, the runtime needs both the matching attention pages and the recurrent snapshot from which it can continue. [Hibana architecture](https://github.com/sharar-project/hibana/blob/d474b74d5a892b2cd1cc8fc50cb17946e2a539bb/docs/architecture.md) · [QSA graphs](https://github.com/sharar-project/hibana/blob/d474b74d5a892b2cd1cc8fc50cb17946e2a539bb/docs/flashnext.md#sparse-attention-past-2048-tokens) · [Prefix-cache design](https://github.com/sharar-project/hibana/blob/d474b74d5a892b2cd1cc8fc50cb17946e2a539bb/docs/cache_design.md)

### What Exact and Fast change

| Knob | Exact | Fast |
|---|---|---|
| Routed expert weights | Base NVFP4 | Calibrated W3 expert weights |
| Model trunk | FP8 at load | FP8 plus calibrated W3 copies for eligible small decode and verify calls |
| Attention KV pool | FP8 | NVFP4 |
| MTP draft depth | Fixed cap of 4 | Up to 8, with per-batch caps and a recent-acceptance policy |
| Greedy verification | Target top-1 | Target top-3 within a 2-logit gap |
| Sampled verification | `accept_soft=0` | `accept_soft=0.5` |
| Expert selection during decode and verify | Normal model routing | Budget on distinct experts read in the forward pass |
| QSA prefill backend | Original | Optimized path where eligible |

The settings come from the pinned [profile implementation](https://github.com/sharar-project/hibana/blob/d474b74d5a892b2cd1cc8fc50cb17946e2a539bb/hibana/launch/settings.py) and the [Exact](../results/exact.server.json) and [Fast](../results/fast.server.json) runtime records. The W3 **weight** format changes expert and projection reads. The NVFP4 **KV** format compresses attention history. The recurrent GDN state uses FP32 in both profiles. [Hibana W3 calibration](https://github.com/sharar-project/hibana/blob/d474b74d5a892b2cd1cc8fc50cb17946e2a539bb/docs/w3-calibration.md) · [Operations and fallback rules](https://github.com/sharar-project/hibana/blob/d474b74d5a892b2cd1cc8fc50cb17946e2a539bb/docs/operations.md#serving-profiles)

With standard speculative sampling, a proposed token with draft probability **q** and target probability **p** is kept with probability **min(1, p/q)**, followed by a correction draw when rejected. Hibana's positive soft margin increases the acceptance threshold in Fast. Exact refers to verification against the model as served with its configured weights and numerics. Fast also changes weights and expert selection, so a profile comparison measures the combined effect of those choices. In our small text probe, Exact matched five of five Chinese answers and Fast matched three of five. See the coding results for a separate task-level comparison. [Speculative decoding paper](https://arxiv.org/abs/2211.17192) · [Hibana acceptance rules](https://github.com/sharar-project/hibana/blob/d474b74d5a892b2cd1cc8fc50cb17946e2a539bb/docs/operations.md#serving-profiles) · [Quality probes](../README.md#traditional-chinese-and-tool-calls) · [Coding benchmark](../README.md#multi-file-coding-benchmark)

### Memory, scheduling and observed performance

Prefill processes input in chunks and writes both the attention KV and recurrent state. Decode and target verification reuse those structures. Hibana allocates a shared one-token-page KV pool and a pool of state slots, including reusable prefix snapshots. The measured runtime reported FlashInfer attention, CuTe GDN, and prefill chunks of 4,096 tokens with a solo limit of 8,192. The launcher sets five active requests, a 262,144-token prompt-plus-output limit per request, and a 12 GiB host memory floor. The floor is host headroom used in planning, while the KV and state allocations depend on memory available at startup. [Hibana cache design](https://github.com/sharar-project/hibana/blob/d474b74d5a892b2cd1cc8fc50cb17946e2a539bb/docs/cache_design.md) · [Memory planner](https://github.com/sharar-project/hibana/blob/d474b74d5a892b2cd1cc8fc50cb17946e2a539bb/docs/memory_plan.md) · [FlashInfer paper](https://arxiv.org/abs/2501.01005)

| Startup ledger on our Spark | Exact | Fast |
|---|---:|---:|
| Weights | 68.89 GiB | 53.03 GiB |
| Attention KV pool | 9.36 GiB | 22.54 GiB |
| KV pages | 545,364 | 1,853,962 |
| Recurrent state pool | 9.26 GiB | 12.06 GiB |

As a capacity check, five requests at the full configured limit would ask for 1,310,720 token positions before any scheduler and verification overhead. The Exact run's recorded pool is smaller than that total. The Fast run's recorded pool is larger, and admission still depends on live memory and state availability. Our five-request measurements used short prompts. The larger Fast KV pool and extra buffers help explain why lower weight memory coexisted with higher total CUDA allocation in that session. The **101.2 GiB** figure is a sampled process peak from a separate Fast run, with startup, decode and prefill included. [Exact memory ledger](../results/exact.server.json) · [Fast memory ledger](../results/fast.server.json) · [Fast memory samples](../results/fast.memory.json)

In the recorded thinking-on prose workload, Exact reached **50.8 tok/s** and Fast reached **101.7 tok/s** at one stream. At five streams the corresponding aggregate rates were **116.6** and **213.8 tok/s**. Draft acceptance, fewer weight bytes and fewer distinct expert reads can all improve decode. This comparison changes those settings together, so it measures the whole profile. The fresh-prompt prefill tests report a different rate using input tokens divided by time to the first token. [Decode methodology](benchmarking.md#workload) · [Exact decode](../results/exact.decode.json) · [Fast decode](../results/fast.decode.json) · [Fast prefill](../results/fast.prefill-32k-128k.json)
