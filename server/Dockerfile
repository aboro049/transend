# seed-vc streaming voice conversion server.
#
# Everything that currently gets wiped when a Runpod pod stops -- the pip
# environment, the repo, the model weights -- is baked in here instead.
# Starting a pod becomes: boot, load weights into VRAM, serve. No installs.
#
# Base image is the exact one validated on the A40: torch 2.8.0+cu128.
# Do NOT let anything reinstall torch; seed-vc's requirements.txt pins
# torch==2.4.0 AND asks for a cu126 nightly in the same file, which would
# downgrade CUDA support and break the GPU path.
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

# HF_HOME points the Hugging Face cache into the image rather than ~/.cache,
# so the 1.3 GB wav2vec2 download happens at build time, not on cold start.
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    OMP_NUM_THREADS=4 \
    HF_HOME=/opt/hf \
    PYTHONPATH=/opt/seedvc

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/seedvc

RUN git clone --depth 1 https://github.com/Plachtaa/seed-vc.git .

# Deliberately NOT seed-vc's requirements.txt -- see note above.
COPY requirements-server.txt /tmp/requirements-server.txt
RUN pip install --no-cache-dir -r /tmp/requirements-server.txt

# Pull every checkpoint at build time. Without this, first connection after
# a cold start waits ~2 minutes on downloads.
COPY prefetch.py /tmp/prefetch.py
RUN python /tmp/prefetch.py && rm /tmp/prefetch.py

COPY app.py /opt/seedvc/server/app.py

# No reference audio is baked in: it is personal data and it changes.
# Clients send it in the handshake, or mount one at /opt/seedvc/reference.wav.
EXPOSE 8000

CMD ["python", "/opt/seedvc/server/app.py", \
     "--seed-vc-dir", "/opt/seedvc", \
     "--reference", "/opt/seedvc/reference.wav", \
     "--port", "8000", \
     "--idle-timeout", "900"]
