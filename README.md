# fastsd-cpu-railway

Text-to-image API for [Railway](https://railway.com), CPU only. Serves SDXS-512 on OpenVINO in the way [FastSD CPU](https://github.com/rupeshs/fastsdcpu) (MIT) runs it (optimum-intel, one step, guidance 1.0, tiny autoencoder) behind one authenticated, OpenAI-style HTTP API.

- `POST /v1/images/generations` (JSON): `prompt`, `n` (1 to 4), `size` (`512x512`), `response_format` (`b64_json` default, or `png` for the image itself), `seed`, `steps` (1 to 4, default 1).
- `GET /v1/models`: the model and the limits in force.
- `GET /health` (503 until the model is loaded and warm, then 200).
- Auth: `Authorization: Bearer $API_KEY` on every route except `/health`. An empty `API_KEY` refuses to start.
- Env: see `env.md`; `API_KEY` is the only required one.

Weights are baked into the image from two ungated Hugging Face repositories pinned to commit SHAs (`rupeshs/sdxs-512-0.9-openvino`, OpenRAIL++; `rupeshs/taesd-ov`, MIT). Prompts and images are held in memory for one request and never logged or stored. The model has no safety checker. Wrapper code is Apache-2.0 (`LICENSE`); see `NOTICE`.

Image: `ghcr.io/bon5co/fastsd-cpu-railway`.
