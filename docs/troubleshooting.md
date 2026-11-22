# Troubleshoot Hibana on DGX Spark

Run these checks from the Spark checkout. Keep the serving terminal open for startup output. Once the container exists, `./bin/recipe.sh logs` shows its last 100 lines. If startup exits and Docker removes the container, use the output from the serving terminal.

## Build or download fails

- **Docker socket permission denied or GPU device unavailable** · Check `docker context show`, `docker info`, and `nvidia-smi -L`. Use the Spark's local Docker daemon. Follow NVIDIA's toolkit guide and test a GPU container with `--gpus all`. [NVIDIA Spark Docker guide](https://docs.nvidia.com/dgx/dgx-spark/nvidia-container-runtime-for-docker.html)
- **Cannot access the Hibana image** · Complete `./bin/recipe.sh build` on the same Docker daemon before running `fetch` or `serve`.
- **Python reports permission denied under `/opt/hibana`** · Pull the current recipe and rebuild the image with `./bin/recipe.sh build`. A clean checkout created with a restrictive umask can produce source files readable only by the image owner. The Dockerfile normalizes the image's source permissions and tests the CLI as a non-root user. Keep the downloaded checkpoints and rerun `serve` after the rebuild.
- **Build dependency or checkpoint download interrupted** · Look for the first failing URL and check network access to GitHub, the Docker image registries, the lockfile's package sources, or Hugging Face as appropriate. Retry the same command and preserve the cache to reuse completed work.
- **No space left on device** · Check free space on the model-cache and Docker image filesystems. Staging both profiles needs about 173 GiB just for the checkpoints. [Storage checks](setup.md#before-you-start)
- **Missing or incomplete checkpoint** · Run `./bin/recipe.sh fetch exact` or `./bin/recipe.sh fetch fast` with the same cache path as `serve`. Fast requires the base checkpoint and the W3 overlay. If a cached upstream source has local edits, choose a clean cache for the pinned build.

## Startup takes a while or the API refuses connections

The first start loads weights and may compile CUDA or Triton kernels. Follow the logs and wait for the API listener. Run the smoke command after startup.

```bash
./bin/recipe.sh logs
ss -ltn 'sport = :18904'
curl --noproxy '*' -fsS --max-time 5 http://127.0.0.1:18904/v1/models
```

- **Port or container name already in use** · Identify the owner before stopping anything. For a running instance launched by this recipe, use `./bin/recipe.sh stop` before switching profiles.
- **GPU or host allocation error** · Check `free -h`, `nvidia-smi`, and the first allocation error in the startup log. Stop other inference workloads you control before retrying on the same host. Start each Hibana profile in its own run.
- **Laptop cannot reach the endpoint** · Try the smoke check on the Spark first. Use the documented SSH local forward for a workstation client. [SSH example](setup.md#send-a-chat-request)

## Chat uses the wrong model or profile

Use the exact model ID from `/v1/models`. Both profiles serve the same model ID. `smoke exact` and `smoke fast` check the active profile name, resolved KV format, draft acceptance setting, and W3 overlay in `/get_server_info`. The runtime's top-level KV dtype can be misleading on this release. [Runtime settings and interpretation](benchmarking.md#reproducibility)

For a bug report, share the recipe commit, Spark driver and Docker versions, the command that failed, and the first error from the logs. Review the excerpt for private paths and credentials before publishing it.
