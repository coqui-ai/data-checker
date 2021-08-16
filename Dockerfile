FROM ghcr.io/coqui-ai/stt-train

RUN python -m pip install pandarallel

WORKDIR /home/ubuntu
COPY . /home/ubuntu/
