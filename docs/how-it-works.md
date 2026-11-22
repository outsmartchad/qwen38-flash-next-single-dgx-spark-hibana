# How Qwen3.8 Flash Next and Hibana work on one DGX Spark

[Setup](setup.md) · [Troubleshooting](troubleshooting.md) · [Measurement method and raw results](benchmarking.md)

## How the model works

How can a model with a 51-billion-parameter lookup table run on a Spark with 128 GB of shared physical memory? The answer starts with what the model reads for one token. It looks up a handful of rows in that table, chooses a small fraction of its feed-forward experts and carries two different kinds of context through its layers. Hibana exploits all three properties. Qwen counts **125B parameters in the language backbone, about 6B active per token, plus 51B in n-gram embeddings and 4B in a multi-token prediction module**. The release also contains a vision encoder. Hibana's pinned Flash Next implementation loads the text path. [Qwen model card][card] · [Architecture report][report] · [Hibana model as served][engine]

### From chat text to four streams of numbers

Suppose you send “The audit code on shelf seven is LIME. What is the audit code?” A chat template adds role markers and an assistant-turn boundary. The tokenizer converts that sequence into integer IDs. A **token** can be a word fragment, punctuation, whitespace or a control marker. The number of tokens used for “LIME” depends on the tokenizer and surrounding text. Every ID selects a row from a padded vocabulary of 248,320 rows. That row is an **embedding**, a vector of 2,560 numbers that provides the starting representation for its position. The conversation's **context** is the ordered sequence available to the model, including template markers and the tokens it has already generated. [Model dimensions][card] · [Pinned checkpoint config][cfg]

The embedding table and projection matrices contain **weights** learned during training. The sequence of numbers computed for this shelf and this question are **activations**. Training can change a weight when the model predicts a continuation poorly. Serving this request leaves the weights fixed and creates activations and temporary memory for its individual context. An identical token starts with the same embedding at both occurrences. Its later representation may describe the code on the shelf in the first sentence and a request for that code in the second. [Qwen training and architecture][report]

```text
IDs → 2,560-wide embeddings → copy into four residual streams
       group 1   GDN + MoE → GDN + MoE → GDN + MoE → QSA + MoE
                             ↑ n-gram features enter before layer 2
       groups 2–12 repeat the four-layer sequence
       all layers read from and write to the four streams
       final gated mixer → output head → vocabulary scores → next ID
       last hidden streams + recent token → one-layer MTP proposer
```

A conventional residual connection adds a layer's update to its input so earlier features can travel through a deep network. Flash Next widens this route to four branches of width 2,560. At the entrance they contain copies of the same embedding. Before a sublayer, the model normalizes each branch. A small network with a 320-dimensional bottleneck produces a gate for each feature on each branch. The gated vectors combine into one input for the expensive sublayer. Separate scalar gates control how strongly its output is added back to each branch. Now the branches can differ. The model makes these read and write decisions again for the next sublayer. The expert runs once on the combined input, and the model retains four channels of features for the next layer. [Qwen's Gated Residual design](https://arxiv.org/html/2608.30320v1#S2.SS2) · [Pinned Hibana mixer][mixer]

### What the large table remembers

A bigram is a pair of consecutive token IDs, and a trigram is a run of three. At the second decoder layer, the prefix-lookup embedding module hashes the current pair and triple. Eight lookup heads for each order select **16 rows per token** from the table. The rows hold learned feature vectors. A projection, a gate informed by the current hidden representation, and a causal convolution inject features into the residual streams. This gives a recurring short sequence a learned representation beyond the ordinary embedding of its final token. Only earlier and current IDs participate. The model can therefore compute the lookup addresses from prompt IDs before all the intervening layers have finished. [Qwen's n-gram design](https://arxiv.org/html/2608.30320v1#S2.SS3) · [Pinned lookup and convolution][ple]

The table is shared across conversations and stays fixed during a chat. The invented fact “shelf seven is LIME” has to travel through the request's sequence memory. A frequently encountered fragment within that sentence might pick up a useful pretrained feature through a lookup. The numerical associations learned during training remain in the table when the conversation ends. These different lifetimes explain why the checkpoint table can live in mapped files while each conversation needs its own recurrent state and attention history. [Lookup implementation][ple] · [Cache design][cache]

### Two ways to connect the question to its answer

**Gated DeltaNet (GDN)** occupies 36 of the 48 layers. Think of a head's recurrent state as a matrix that associates keys with values. Learned projections and a short causal convolution give a new position query, key and value features. A decay gate controls how much earlier state survives. The layer compares what the decayed state predicts for this key with its incoming value and writes a gated fraction of the difference. A query reads from the updated state, and a sigmoid output gate modulates what moves onward. The model uses 16 query/key heads and 48 value heads of width 128. Many positions contribute to a fixed-size state. Decode advances that state incrementally, while chunkwise prefill methods process blocks efficiently. [Gated DeltaNet paper](https://arxiv.org/abs/2412.06464) · [Qwen's GDN design](https://arxiv.org/html/2608.30320v1#S2.SS1.SSS1) · [Engine geometry][engine]

**Qwen Sparse Attention (QSA)** occupies every fourth layer. In ordinary attention a token's query scores visible keys, the scores become weights through softmax, and a weighted sum of the values brings information back. Flash Next uses 24 query heads sharing two KV heads of width 256. Rotary position embeddings act on 64 dimensions to give the match information about relative position. A causal mask prevents a position from reading future text. When the whole prompt fits within the 2,048-token budget, Hibana can run ordinary dense causal attention. [Checkpoint dimensions][cfg] · [Hibana attention path][qsa]

Farther into a document, QSA's cheaper indexer groups history into four-token micro-blocks. It compares queries to compressed block keys, chooses up to 512 completed blocks, and includes the current visible partial block. The main attention then reads the selected token positions. At position 32,000 a block containing the original shelf entry near position 100 can still be selected. The budget limits how much value information the expensive stage reads on this pass. It still needs the retained history available for different selections on the next pass. During continued pretraining Qwen replaced periodic full-attention layers with QSA and trained the indexer. A running Spark performs its learned selections without updating that indexer. [Qwen's QSA design and training](https://arxiv.org/html/2608.30320v1#S2.SS1.SSS2) · [Hibana's sparse path][qsa]

### Which experts run and how the model chooses a token

Each of the 48 layers follows its sequence mixer with a **mixture of experts** (MoE). A router scores 512 candidate feed-forward networks from the current vector. The normal model selects ten, weights their results, and adds a sigmoid-gated shared expert that runs for every token. An expert projects the incoming vector into two branches of width 640, applies a nonlinear gate to one and multiplies it by the other, then projects the result back to width 2,560. These operations transform features at one position. The sequence mixer has already brought in context from other positions. A single token makes 480 routed selections over the 48 layers. The ten selected in layer one refer to a different weight bank from the ten in layer two. [Model specifications][card] · [Hibana's routed and shared experts][moe]

A global mixer reads the final four branches and an output matrix computes one **logit**, or unnormalized next-token score, for every vocabulary row. Greedy generation chooses the largest score. In the measured prose requests, temperature 0.7 sharpened the distribution, top-k 20 kept up to twenty candidates, and top-p 1 imposed no extra probability-mass cutoff. A sampled token joins the context and the tokenizer renders the accumulating IDs as text. The one-layer **multi-token prediction** module (MTP) can use recent hidden features plus a new token embedding to propose the next ID cheaply, then iterate. A target pass verifies a block together. Keeping two acceptable guesses can advance the sequence by several positions per expensive pass. A failed proposal requires a correction and a return to the committed history. [Model forward and head][forward] · [Training with multi-token prediction](https://arxiv.org/abs/2404.19737) · [Hibana's MTP implementation][mtp]

## How this recipe runs it on one DGX Spark

### Build the engine and serve the checkpoint

```bash
./bin/recipe.sh build
./bin/recipe.sh fetch fast
./bin/recipe.sh serve fast
# In another terminal on the Spark
./bin/recipe.sh smoke fast
```

The Dockerfile builds Hibana 0.1.0 at `d474b74` for ARM64 with CUDA and locked Python dependencies. The RadixArk base checkpoint holds NVFP4 routed experts and an FP8 n-gram table. Fast additionally stages a calibrated three-bit (W3) overlay. Disk usage is about 126 GiB for the base plus 47 GiB for W3, apart from the build and kernel caches. For Exact, use `fetch exact`, `serve exact` and `smoke exact`. Both profiles mount the weights read-only and use offline Hub mode. The smoke check confirms the active profile, lists the model and checks a response from the local API at `127.0.0.1:18904`. Chat requests use the served ID `RadixArk/Qwen3.8-Flash-Next-NVFP4`. [Setup](setup.md) · [Dockerfile](../Dockerfile) · [Launcher](../bin/recipe.sh) · [Base snapshot][target] · [W3 snapshot][w3]

```text
localhost chat API → template and tokenizer → scheduler
     prefill → read n-gram rows → run target chunks → KV pages + GDN state
     decode  → MTP proposals → target verifies → commit accepted prefix + correction/bonus
                            ↖ next round begins at committed state
```

The Grace CPU and Blackwell GPU share 128 GB of physical memory. On one Spark, the n-gram table is memory mapped from checkpoint shards. Hibana gathers the requested rows and can prefetch later prompt rows while the first chunk runs. The OS file cache can retain frequently used pages. This avoids a permanent GPU allocation for the whole table while making storage access and cache warmth relevant to latency. The recipe disables Hibana's SSD tier for *saved inference-cache blocks*. The mapped weights still reside on storage. [DGX Spark specifications][spark] · [PLE implementation][ple] · [Hibana operations][ops]

**Prefill** processes the known input positions, creates KV entries and advances the GDN states. The recorded engine used FlashInfer attention, CuTe GDN, 4,096-token prefill chunks and a solo chunk limit of 8,192. **Decode** and target verification reuse that history. Captured CUDA graphs replay supported batches of operations, avoiding repeated CPU dispatch for each GPU kernel. For long contexts the graph scores and assembles selected attention pages on the GPU. KV pages also retain the QSA index data. Resuming a reused prefix needs the matching recurrent snapshot as well as those pages. [Engine architecture][arch] · [QSA graphs][qsa] · [Prefix cache][cache] · [FlashInfer paper](https://arxiv.org/abs/2501.01005)

### Exact and Fast change different parts of a forward pass

| Setting | Exact | Fast |
|---|---|---|
| Routed expert weights | Base NVFP4 | Calibrated W3 |
| Trunk projections | FP8 at load | FP8 plus W3 copies for eligible small decode and verify calls |
| Attention KV pool | FP8 | NVFP4 |
| MTP depth | Cap of 4 | Up to 8 with batch caps and recent-acceptance adaptation |
| Greedy verification | Target top-1 | Target top-3 within a 2-logit gap |
| Sampled verification | `accept_soft=0` | `accept_soft=0.5` |
| Expert selection during decode and verify | Normal routing | Budget on distinct expert weights read per forward pass |
| QSA prefill | Original backend | Optimized backend where eligible |

Quantization stores approximations to the original numbers. NVFP4 packs four-bit values with block scales. The W3 overlay uses calibrated three-bit copies to reduce bytes read when running experts and eligible projections. The attention KV format determines how many bytes record the history of a conversation. Both profiles keep FP32 GDN state. In this release the raw server information's top-level KV dtype stays FP8 even for Fast. Read `config.kv_format` to see the effective cache format. [Exact settings](../results/exact.server.json) · [Fast settings](../results/fast.server.json) · [Pinned profiles][profiles] · [W3 calibration][cal] · [NVIDIA's NVFP4 format][nvfp4]

In standard speculative sampling, a token that has probability *q* under the draft and *p* under the target is kept with probability `min(1, p/q)`. At the first rejection the engine uses a correction draw. If every candidate passes, it can draw a bonus token. Exact applies this test against the target as actually served with quantized weights. Fast's positive soft margin also accepts proposals under a relaxed test, and Fast changes weight precision and expert selection. These differences can affect the answer. In five Traditional Chinese probes Exact matched 5/5 and Fast 3/5, including a wording variant and a clock error. Both passed 3/4 tool checks. On one coding task Exact passed 10/10 independent feature checks and Fast 7/10. [Speculative sampling paper](https://arxiv.org/abs/2211.17192) · [Engine acceptance rules][ops] · [Chinese and tool checks](../README.md#traditional-chinese-and-tool-calls) · [Coding task](../README.md#multi-file-coding-benchmark)

### Read the memory ledger and the benchmark

| Recorded startup allocation | Exact | Fast |
|---|---:|---:|
| Weights | 68.89 GiB | 53.03 GiB |
| Attention KV pool | 9.36 GiB | 22.54 GiB |
| One-token KV pages | 545,364 | 1,853,962 |
| Recurrent state pool | 9.26 GiB | 12.06 GiB |
| Proposer and verification buffers | 4.10 GiB | 6.65 GiB |
| CUDA graphs | 0.77 GiB | 2.80 GiB |

The launcher allows five active requests, a limit of 262,144 prompt-plus-output tokens per request and a 12 GiB host memory floor. This floor preserves host headroom when the planner sizes its pools. Five full-length requests would require 1,310,720 token positions before scheduler and verification overhead. The Exact ledger has fewer pages than that. Fast has more, but admission still depends on live memory and state slots. The five-stream tests used short prompts. Saving weight memory allowed Fast to allocate more history and verification space. Its **101.2 GiB sampled process peak** came from a separate run covering startup, one through five decode streams and 32K/200K prefill. [Memory planner][planner] · [Exact ledger](../results/exact.server.json) · [Fast ledger](../results/fast.server.json) · [Memory series](../results/fast.memory.json)

The November 21 prose test requested 400 generated tokens per stream with thinking on. At one stream the best of three waves reached **50.8 output tok/s for Exact and 101.7 for Fast**. At five streams the aggregate rates were **116.6 and 213.8 tok/s**, respectively. These figures compare complete serving profiles, with multiple simultaneous changes. The separate Fast prefill experiment used fresh prompts, temperature zero, thinking off and verified zero cached tokens. Across three samples per size, median time to first token was **8.64 s at 32K** and **37.07 s at 128K**, or **3,796 and 3,536 input tok/s**. All six returned the expected code. The earlier single 199,981-token prompt also returned its key. A retrieval success tells us what happened to that prompt, while the memory ledger explains which additional requests could compete for the same pools. [Method](benchmarking.md#workload) · [Exact decode](../results/exact.decode.json) · [Fast decode](../results/fast.decode.json) · [32K/128K samples](../results/fast.prefill-32k-128k.json) · [Earlier 200K sample](../results/fast.prefill.json)

[card]: https://huggingface.co/Qwen/Qwen3.8-Flash-Next#model-overview
[report]: https://arxiv.org/html/2608.30320v1#S2
[cfg]: https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4/blob/7b719225242aacd3dbd3f9407468c2ee9a9d2594/config.json
[target]: https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4/tree/7b719225242aacd3dbd3f9407468c2ee9a9d2594
[w3]: https://huggingface.co/yemara/Qwen3.8-Flash-Next-NVFP4-sharar-w3-qad/tree/5c4c62a62853caebea8cedda7fbed952efa1effe
[spark]: https://www.nvidia.com/en-us/products/workstations/dgx-spark/#specifications
[nvfp4]: https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/
[engine]: https://github.com/sharar-project/hibana/blob/d474b74d5a892b2cd1cc8fc50cb17946e2a539bb/docs/flashnext.md#the-model-as-served
[mixer]: https://github.com/sharar-project/hibana/blob/d474b74d5a892b2cd1cc8fc50cb17946e2a539bb/hibana/model/flashnext/hyper.py
[ple]: https://github.com/sharar-project/hibana/blob/d474b74d5a892b2cd1cc8fc50cb17946e2a539bb/hibana/model/flashnext/ple.py
[qsa]: https://github.com/sharar-project/hibana/blob/d474b74d5a892b2cd1cc8fc50cb17946e2a539bb/docs/flashnext.md#sparse-attention-past-2048-tokens
[moe]: https://github.com/sharar-project/hibana/blob/d474b74d5a892b2cd1cc8fc50cb17946e2a539bb/hibana/model/flashnext/moe.py
[forward]: https://github.com/sharar-project/hibana/blob/d474b74d5a892b2cd1cc8fc50cb17946e2a539bb/hibana/model/flashnext/model.py
[mtp]: https://github.com/sharar-project/hibana/blob/d474b74d5a892b2cd1cc8fc50cb17946e2a539bb/docs/flashnext.md#the-mtp-draft
[ops]: https://github.com/sharar-project/hibana/blob/d474b74d5a892b2cd1cc8fc50cb17946e2a539bb/docs/operations.md#serving-profiles
[arch]: https://github.com/sharar-project/hibana/blob/d474b74d5a892b2cd1cc8fc50cb17946e2a539bb/docs/architecture.md
[cache]: https://github.com/sharar-project/hibana/blob/d474b74d5a892b2cd1cc8fc50cb17946e2a539bb/docs/cache_design.md
[cal]: https://github.com/sharar-project/hibana/blob/d474b74d5a892b2cd1cc8fc50cb17946e2a539bb/docs/w3-calibration.md
[profiles]: https://github.com/sharar-project/hibana/blob/d474b74d5a892b2cd1cc8fc50cb17946e2a539bb/hibana/launch/settings.py
[planner]: https://github.com/sharar-project/hibana/blob/d474b74d5a892b2cd1cc8fc50cb17946e2a539bb/docs/memory_plan.md
