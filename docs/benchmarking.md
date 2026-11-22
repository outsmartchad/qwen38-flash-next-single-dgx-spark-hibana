# Benchmark methodology

## Run the benchmarks

Follow the [quick start](../README.md#quick-start) for the profile you want to measure. In a second Spark terminal from the checkout, run the following after the smoke check has passed. Set `profile=fast` for the Fast run.

```bash
export SPARK_BENCHMARK_URL=http://127.0.0.1:18904
export SPARK_BENCHMARK_MODEL=RadixArk/Qwen3.8-Flash-Next-NVFP4
profile=exact
mkdir -p runs
curl --noproxy '*' -fsS "$SPARK_BENCHMARK_URL/get_server_info" >"runs/$profile.server.json" &&
python3 bin/benchmark.py >"runs/$profile.decode.json" &&
python3 bin/checks.py >"runs/$profile.checks.json"
```

## Workload

The direct `/v1/chat/completions` client uses the [SparkDash hash-map prompt](https://github.com/MiaAI-Lab/sparkDash/blob/91a3e46d7cbadf6daf4d53b5e7ae34e519f353ca/src/shared/llmPrompts.js#L53-L56). At concurrency two and above it prefixes each request with its own `Request ID: decode-N`. The client runs a short chat smoke check, then a 32-token warm-up on the prose prompt. It sends three synchronized waves at each concurrency from one through five.

Each measured stream requests 400 tokens with `ignore_eos: true` and verifies that streamed usage reports 400 completion tokens. The run confirmed streamed reasoning in all 135 requests across the three configurations. Repeated prompts can warm the engines' prefix caches. Raw results preserve the reported cached-token count when the API supplies it.

- **Per-request decode** divides output tokens minus one by the interval between the first and last nonempty reasoning or answer chunk.
- **Aggregate decode** divides total output tokens minus the number of streams by the interval between the earliest first chunk and the latest last chunk.
- **TTFT** runs from request start to the first nonempty reasoning or answer chunk.
- **Best wave** means the highest aggregate decode of the three attempts at a given concurrency. The per-request rate and TTFT use the medians within that wave.

## Best-wave per-request throughput

| Streams | TensorFold tok/s | Hibana exact tok/s | Hibana fast tok/s |
|---:|---:|---:|---:|
| 1 | 44.5 | 50.8 | 101.7 |
| 2 | 38.7 | 37.9 | 81.8 |
| 3 | 33.4 | 32.3 | 65.1 |
| 4 | 27.9 | 28.5 | 58.7 |
| 5 | 25.2 | 24.8 | 49.7 |

## Reproducibility

Hibana v0.1.0 ran from unmodified upstream commit `d474b74d5a892b2cd1cc8fc50cb17946e2a539bb` in an ARM64 image built with the pinned [Dockerfile](../Dockerfile) and upstream lockfile. Both profiles used `--max-batch 5`, `--context-length 262144`, a 12 GiB host memory floor, and settings with tracing and the SSD cache tier disabled. The original TensorFold 1.2.7 container was stopped before Hibana and restored after the run. The benchmark API and engine ran on the same host.

For KV format, read `config.kv_format` in the raw Hibana runtime record. The top-level `kv_cache_dtype` field in this release reports `float8_e4m3fn` even when the resolved cache format is NVFP4.

## Cold prefill and process memory

With only the dedicated Hibana benchmark server running and the URL and model variables from the commands above exported, run the prefill client.

```bash
python3 bin/prefill.py runs/fast.prefill-32k-128k.json
```

The script targets 32,768 and 131,072 input tokens using synthetic inventory prompts. It warms up once at each size, then alternates which size runs first over three rounds. Each sample uses fresh prompt content, flushes the server's prefix cache, and validates zero cached tokens from streamed usage. Rate is the actual prompt tokens divided by client time to first token. TTFT includes request handling, prefill, and first-token generation. The table reports median tok/s and median TTFT across the three samples at each size.

For the earlier 32K/200K run, the Spark ran `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits -lms 200` from before launch through decode and prefill. We filtered the samples using the container's host PID. The published peak is the maximum per-process GPU memory sample, converted from MiB to GiB. The recorded series includes startup and workload samples.

### Raw artifacts

- TensorFold control · [Decode](../results/tensorfold.decode.json) · [Chinese and tool calls](../results/tensorfold.checks.json)
- Hibana exact · [Decode](../results/exact.decode.json) · [Chinese and tool calls](../results/exact.checks.json) · [Runtime settings](../results/exact.server.json)
- Hibana fast · [Decode](../results/fast.decode.json) · [Chinese and tool calls](../results/fast.checks.json) · [Runtime settings](../results/fast.server.json)
- [Base revision](../results/target.json) · [W3 revision](../results/w3.json) · [Image digest](../results/hibana.image.txt)
- [32K/128K cold prefill and warm-ups](../results/fast.prefill-32k-128k.json)
- [Earlier 32K/200K run](../results/fast.prefill.json) · [Per-process memory samples](../results/fast.memory.json) · [Decode sweep used for memory sampling](../results/fast.decode-memory-workload.json)
