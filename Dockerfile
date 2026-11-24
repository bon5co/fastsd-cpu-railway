# FastSD CPU-style text-to-image API (SDXS-512, OpenVINO, one step), Bearer API_KEY, OpenAI-style /v1/images/generations.
# Source: https://github.com/bon5co/fastsd-cpu-railway
# CPU only (OpenVINO), weights baked in (about 3 GB), stateless (no volume needed).
# upstream: rupeshs/fastsdcpu (MIT) is the project whose SDXS-512 OpenVINO path this serves; it is not installed (its GUI,
# Qt and MCP server are not needed): the same optimum-intel 2.0.0 / openvino 2026.2.0 / diffusers 0.37.0 / transformers 5.0.0
# pins as fastsdcpu commit 10ed947f16cd35479a1d202be4e1309e243fd767 (main, checked 2026-11-24) are installed from PyPI, and
# the weights are rupeshs/sdxs-512-0.9-openvino at commit f6cacf7a521e8801904f821bafa9dec7e4d10594.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

COPY requirements.txt /tmp/requirements.txt
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r /tmp/requirements.txt \
    && pip list 2>/dev/null | grep -iE "^(torch|numpy|transformers|diffusers|openvino|optimum|optimum-intel|fastapi) "

# Weights: one ungated Hugging Face repository (openrail++), pinned to a commit SHA (see fetch_weights.py).
ENV MODEL_ROOT=/models
COPY fetch_weights.py /tmp/fetch_weights.py
RUN python /tmp/fetch_weights.py && find /models -name .cache -type d -prune -exec rm -rf {} + && rm -rf /root/.cache && du -sh /models

# MIT licence of rupeshs/fastsdcpu at the pinned commit; OpenRAIL++ text is quoted in NOTICE and the readme.
COPY LICENSE-fastsdcpu /licenses/fastsdcpu-LICENSE
COPY NOTICE /licenses/NOTICE

RUN useradd --uid 1000 --create-home --shell /usr/sbin/nologin img \
    && chmod -R a+rX /models /licenses
WORKDIR /app
COPY app.py entrypoint.sh NOTICE LICENSE /app/
RUN chmod +x /app/entrypoint.sh
# Every model file is local; any attempt to reach Hugging Face at run time is a bug, not a download.
ENV MODEL_ROOT=/models HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/tmp/hf
USER 1000
EXPOSE 8080
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python", "/app/app.py"]
