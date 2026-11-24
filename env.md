# SDXS-512 text-to-image API (FastSD CPU) — environment

Paste into Railway's variables panel. Everything else lives in the image, because a template variable holding a literal is dropped at generate time and republishes as a blank required field.

```
API_KEY="${{secret(32)}}"
```

## Optional

| Variable | Default | Values |
|---|---|---|
| `TINY_VAE` | `1` | `1` decodes with FastSD CPU's tiny autoencoder (what its SDXS-512 mode uses); `0` uses the full SD VAE, slower and about 1 GB larger in memory |
| `MAX_IMAGES` | `4` | Most images per request (`n`) |
| `MAX_PROMPT_CHARS` | `1000` | Longest `prompt` accepted (413 above it) |
| `MAX_SECONDS` | `240` | Wall-clock budget from request arrival; past it the request answers 504 (Railway's edge cuts a connection at about 5 minutes) |
| `MAX_PENDING` | `3` | Requests running or waiting; another answers 429 with `Retry-After: 30` |

## Set by the entrypoint at boot

`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `THREADS` (OpenVINO's `INFERENCE_NUM_THREADS`) come from `/sys/fs/cgroup/cpu.max`, because the runtimes size their thread pools from the host's cores and oversubscribe the container.

`PORT` is left to Railway: the healthcheck dials the injected port, so the app honours it.
