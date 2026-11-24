#!/usr/bin/env python3
"""Text-to-image API for Railway, CPU only: SDXS-512 on OpenVINO, one step, in the way FastSD CPU runs it.

rupeshs/fastsdcpu (MIT) runs SDXS-512 through optimum-intel's OVStableDiffusionPipeline with
rupeshs/sdxs-512-0.9-openvino, 512x512, one inference step, guidance 1.0. This file serves exactly that over HTTP,
because upstream ships a desktop GUI, a Gradio page and a local MCP server, not a deployable, authenticated API.

What this file adds around the model:

  1. Bearer API_KEY on every route except GET /health. An empty API_KEY refuses to start.
  2. GET /health answers 503 until the model is loaded and warmed, then 200, unauthenticated.
  3. POST /v1/images/generations in the shape of the OpenAI images endpoint (prompt, n, size, response_format,
     seed). Weights are baked into the image; nothing downloads at boot.
  4. One generation at a time (CPU bound), a short queue, 429 when it is full, and a wall-clock budget between
     images so a request answers 504 before Railway's roughly 5 minute edge timeout.
  5. Prompts and images are held in memory for the request only and are never logged or written to disk.
"""

import asyncio
import base64
import io
import os
import socket
import sys
import threading
import time

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

API_KEY = os.environ.get("API_KEY") or ""
THREADS = int(os.environ.get("THREADS") or os.cpu_count() or 1)
MODEL_ROOT = os.environ.get("MODEL_ROOT") or "/models"
MODEL_DIR = os.path.join(MODEL_ROOT, "sdxs-512-0.9-openvino")
TAESD_DIR = os.path.join(MODEL_ROOT, "taesd-ov")
MODEL_ID = "sdxs-512"
# FastSD CPU's own SDXS-512 mode (realtime_ui.py) sets use_tiny_auto_encoder = True: the 2.4 MB TAESD decoder
# replaces the 198 MB SD VAE decoder. TINY_VAE=0 keeps the full VAE (about 4x slower per image on 5 vCPU).
TINY_VAE = (os.environ.get("TINY_VAE") or "1") not in ("0", "false", "no")
MAX_PROMPT_CHARS = int(os.environ.get("MAX_PROMPT_CHARS") or 1000)
MAX_IMAGES = max(1, int(os.environ.get("MAX_IMAGES") or 4))
MAX_SECONDS = float(os.environ.get("MAX_SECONDS") or 240)
MAX_PENDING = int(os.environ.get("MAX_PENDING") or 3)
PUBLIC_PATHS = {"/health"}
SIZE = (512, 512)

_state = {"ready": False, "error": None, "loaded_in": None}
_pipe = None
_run_lock = threading.Lock()  # one generation at a time
_pending = 0
_pending_lock = threading.Lock()

app = FastAPI(title="SDXS-512 text-to-image API", docs_url=None, redoc_url=None, openapi_url=None)


class _Deadline(Exception):
    pass


def _status_kb(field: str) -> int:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(field + ":"):
                return int(line.split()[1])
    return 0


def _boot() -> None:
    global _pipe
    t0 = time.time()
    try:
        torch.set_num_threads(THREADS)
        from optimum.intel.openvino.modeling_diffusion import OVStableDiffusionPipeline
        # Same call FastSD CPU makes (src/backend/openvino/pipelines.py), with the thread count taken from the
        # container's cgroup quota: OpenVINO would otherwise size its pool from the host's cores.
        _pipe = OVStableDiffusionPipeline.from_pretrained(
            MODEL_DIR, local_files_only=True, device="CPU",
            ov_config={"CACHE_DIR": "", "INFERENCE_NUM_THREADS": THREADS})
        if TINY_VAE:
            # What fastsdcpu's ov_load_tiny_autoencoder() does, with local files.
            from optimum.intel.openvino.modeling_diffusion import OVBaseModel, OVModelVae, OVModelVaeDecoder, OVModelVaeEncoder
            decoder = OVModelVaeDecoder(model=OVBaseModel.load_model(f"{TAESD_DIR}/vae_decoder/openvino_model.xml"),
                                        parent_pipeline=_pipe, model_name="vae_decoder")
            encoder = OVModelVaeEncoder(model=OVBaseModel.load_model(f"{TAESD_DIR}/vae_encoder/openvino_model.xml"),
                                        parent_pipeline=_pipe, model_name="vae_encoder")
            _pipe.vae = OVModelVae(decoder=decoder, encoder=encoder)
            _pipe.vae.config.scaling_factor = 1.0
        t_load = time.time() - t0
        _pipe(prompt="a cat", width=SIZE[0], height=SIZE[1], num_inference_steps=1, guidance_scale=1.0)
        _state["ready"] = True
        _state["loaded_in"] = round(time.time() - t0, 1)
        print(f"[sdxs] ready threads={THREADS} tiny_vae={TINY_VAE} load={t_load:.1f}s warm_total={_state['loaded_in']}s", flush=True)
    except Exception as exc:  # noqa: BLE001 - reported over /health, never fatal
        _state["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[sdxs] LOAD FAILED: {_state['error']}", flush=True)


def _generate(prompt: str, n: int, steps: int, seed: int | None, deadline: float):
    images = []
    for i in range(n):
        if time.time() > deadline:
            raise _Deadline()
        generator = None
        if seed is not None:
            generator = torch.Generator().manual_seed(seed + i)
        out = _pipe(prompt=prompt, width=SIZE[0], height=SIZE[1], num_inference_steps=steps,
                    guidance_scale=1.0, generator=generator)
        images.append(out.images[0])
    return images


def _png(image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


@app.middleware("http")
async def guards(request: Request, call_next):
    path = request.url.path.rstrip("/") or "/"
    if path in PUBLIC_PATHS:
        return await call_next(request)
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    if not (scheme.lower() == "bearer" and API_KEY and token == API_KEY):
        return JSONResponse({"error": "Missing or invalid Authorization: Bearer <API_KEY>."},
                            status_code=401, headers={"WWW-Authenticate": "Bearer"})
    length = request.headers.get("content-length")
    if length and length.isdigit() and int(length) > 65536:
        return JSONResponse({"error": "request body is too large"}, status_code=413)
    return await call_next(request)


@app.get("/health", include_in_schema=False)
def health():
    if not _state["ready"]:
        return JSONResponse({"status": "loading" if not _state["error"] else "failed",
                             "error": _state["error"]}, status_code=503)
    return {"status": "ok", "loaded_in_s": _state["loaded_in"], "threads": THREADS, "model": MODEL_ID, "tiny_vae": TINY_VAE,
            "rss_mb": _status_kb("VmRSS") // 1024, "peak_rss_mb": _status_kb("VmHWM") // 1024}


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [
        {"id": MODEL_ID, "object": "model", "default": True, "license": "openrail++",
         "description": "SDXS-512-0.9, one-step text-to-image, OpenVINO on CPU, 512x512"}],
        "limits": {"max_prompt_chars": MAX_PROMPT_CHARS, "max_images": MAX_IMAGES, "sizes": ["512x512"],
                   "max_seconds": MAX_SECONDS, "max_pending": MAX_PENDING,
                   "response_formats": ["b64_json", "png"], "steps": "1-4, default 1"}}


class ImageRequest(BaseModel):
    prompt: str
    model: str | None = None
    n: int = Field(default=1, ge=1)
    size: str | None = "512x512"
    response_format: str | None = "b64_json"
    seed: int | None = Field(default=None, ge=0, le=2**31 - 1)
    steps: int = Field(default=1, ge=1, le=4)
    quality: str | None = None
    style: str | None = None
    user: str | None = None


@app.post("/v1/images/generations")
async def generations(req: ImageRequest):
    global _pending
    prompt = (req.prompt or "").strip()
    if not prompt:
        raise HTTPException(422, "`prompt` is empty")
    if len(prompt) > MAX_PROMPT_CHARS:
        raise HTTPException(413, f"`prompt` is {len(prompt)} characters; the limit is {MAX_PROMPT_CHARS}")
    if req.model not in (None, "", MODEL_ID, "sdxs", "dall-e-2", "dall-e-3"):
        raise HTTPException(422, f"unknown model '{req.model}'; use '{MODEL_ID}'")
    if (req.size or "512x512") != "512x512":
        raise HTTPException(422, "only size '512x512' is supported")
    fmt = (req.response_format or "b64_json").lower()
    if fmt not in ("b64_json", "png"):
        raise HTTPException(422, "`response_format` is 'b64_json' (JSON) or 'png' (the image itself); "
                                 "'url' is not supported because nothing is stored")
    if req.n > MAX_IMAGES:
        raise HTTPException(422, f"`n` is at most {MAX_IMAGES}")
    if fmt == "png" and req.n != 1:
        raise HTTPException(422, "response_format 'png' returns one image; use n=1 or b64_json")
    if not _state["ready"]:
        raise HTTPException(503, "model is still loading")
    with _pending_lock:
        if _pending >= MAX_PENDING:
            return JSONResponse({"error": "busy: too many requests are running or waiting"}, status_code=429,
                                headers={"Retry-After": "30"})
        _pending += 1
    deadline = time.time() + MAX_SECONDS

    def job():
        if not _run_lock.acquire(timeout=max(1.0, deadline - time.time())):
            raise _Deadline()
        try:
            t0 = time.time()
            images = _generate(prompt, req.n, req.steps, req.seed, deadline)
            return images, time.time() - t0
        finally:
            _run_lock.release()

    try:
        try:
            images, seconds = await asyncio.to_thread(job)
        except _Deadline:
            return JSONResponse({"error": f"could not finish within {MAX_SECONDS:.0f} s; ask for fewer images"},
                                status_code=504)
        headers = {"X-Processing-Seconds": f"{seconds:.2f}", "X-Seconds-Per-Image": f"{seconds / len(images):.2f}",
                   "X-Model": MODEL_ID, "X-Steps": str(req.steps)}
        if fmt == "png":
            return Response(_png(images[0]), media_type="image/png", headers=headers)
        data = [{"b64_json": base64.b64encode(_png(im)).decode("ascii")} for im in images]
        return JSONResponse({"created": int(time.time()), "data": data, "model": MODEL_ID}, headers=headers)
    finally:
        with _pending_lock:
            _pending -= 1


def main() -> int:
    if not API_KEY:
        print("[sdxs] FATAL: API_KEY is empty; refusing to publish the API to anyone who finds the URL.",
              file=sys.stderr, flush=True)
        return 1
    threading.Thread(target=_boot, name="load", daemon=True).start()
    # Railway injects PORT and its healthcheck dials that injected port. asyncio sets
    # IPV6_V6ONLY on a "::" listener, so one socket per family: IPv4 for the edge and
    # healthcheck, IPv6 for the private network (<service>.railway.internal).
    port = int(os.environ.get("PORT") or 8080)
    socks = []
    for family, addr in ((socket.AF_INET, "0.0.0.0"), (socket.AF_INET6, "::")):
        try:
            sock = socket.socket(family, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if family == socket.AF_INET6:
                sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            sock.bind((addr, port))
            sock.set_inheritable(True)
            socks.append(sock)
        except OSError as exc:
            print(f"[sdxs] cannot bind {addr}:{port}: {exc}", flush=True)
    if not socks:
        return 1
    uvicorn.Server(uvicorn.Config(app, log_level=os.environ.get("LOG_LEVEL", "warning"))).run(sockets=socks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
