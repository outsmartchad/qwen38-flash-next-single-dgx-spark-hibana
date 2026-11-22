# Set up Hibana on one DGX Spark

[Quick start](../README.md#quick-start) · [Troubleshooting](troubleshooting.md) · [Benchmark method and raw data](benchmarking.md)

## Before you start

Run the commands on a GB10 DGX Spark with Docker Engine and NVIDIA GPU container support. The reference machine uses NVIDIA driver 580.178.04. Confirm that your current user can access the local Docker daemon and that the host sees the GPU. [NVIDIA's Docker guide for DGX Spark](https://docs.nvidia.com/dgx/dgx-spark/nvidia-container-runtime-for-docker.html)

```bash
uname -m
nvidia-smi -L
docker context show
docker info --format '{{.ServerVersion}}'
free -h
df -h "$HOME"
df -h "$(docker info --format '{{.DockerRootDir}}')"
```

Expect `aarch64` and one NVIDIA GB10. The Docker storage check may require admin access. Check both the Docker image store and the model-cache filesystem if they are on different mounts.

- **Exact** downloads about 126 GiB of base weights. **Fast** also downloads about 47 GiB of calibrated W3 weights.
- Leave extra storage for the CUDA image, uv dependencies, and compiled kernels. `build` fetches the pinned Hibana commit and uses the ARM64 [Dockerfile](../Dockerfile) and upstream lockfile.
- Build and download use the network. Serving uses the staged checkpoints in offline Hub mode. The pinned checkpoint downloads are public.
- Before either profile starts, stop other inference workloads you control and check `free -h`. Both profiles load the base checkpoint and size their KV cache on the Spark's unified memory.

## Choose a cache location

The recipe defaults to `.cache/` inside the checkout. Set the same absolute directory in every terminal to reuse weights and compiled kernels across checkouts or move them to another volume.

```bash
export SPARK_HIBANA_CACHE="$HOME/.cache/hibana-flash-next"
./bin/recipe.sh build &&
./bin/recipe.sh fetch fast &&
./bin/recipe.sh serve exact
```

Use a dedicated cache for these public weights. `fetch` verifies the pinned snapshot files against Hugging Face metadata. On an interrupted download, run the same fetch command again with the same cache path so completed files can be reused.

The image build checks that a non-root user can import and start the Hibana CLI. At runtime the container uses the caller's UID and GID, which gives it access to the recipe's writable state and kernel cache. The pinned weight mount stays read-only. [Launcher](../bin/recipe.sh)

## Send a chat request

In a second terminal on the Spark, verify the profile you started with `./bin/recipe.sh smoke exact` or `./bin/recipe.sh smoke fast`. Then send an OpenAI-compatible request.

```bash
curl --noproxy '*' -fsS --max-time 90 http://127.0.0.1:18904/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"RadixArk/Qwen3.8-Flash-Next-NVFP4","messages":[{"role":"user","content":"Explain how a hash map works in one sentence."}],"temperature":0,"max_tokens":128,"chat_template_kwargs":{"enable_thinking":false}}'
```

The answer is in `choices[0].message.content`. For a client on your workstation, use an SSH local forward. Run it on your workstation with your own Spark SSH alias and leave it open.

```bash
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:18904:127.0.0.1:18904 your-spark
```

Point the workstation client at `http://127.0.0.1:18904/v1`. Both ends of the forward are on loopback. [Reproduce the published measurements](benchmarking.md#run-the-benchmarks)

## Clean-install check

On November 22, 2026, we cloned the recipe on a single DGX Spark, built the pinned ARM64 image, and downloaded and verified both checkpoints into an empty cache. A private checkout reproduced an image import-permissions failure, which the current Dockerfile fixes. The rebuilt image passed the CLI check under the serving user.

We started Exact and Fast in that order. Both served on `127.0.0.1:18904`, returned `READY` in the profile-aware smoke check, and answered the hash-map example. Exact reached the listener in 163 seconds and Fast in 102 seconds in this session, including weight loading and graph capture. The profiles shared the same working cache, so the timings describe this startup order. The test stopped Hibana, restored the original TensorFold container, and verified a reply from it on loopback.
