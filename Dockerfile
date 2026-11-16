FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/data/huggingface GRADIO_SERVER_NAME=0.0.0.0 GRADIO_SERVER_PORT=7860
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir '.[tts,asr,studio,train]'
RUN mkdir -p /data && chmod 777 /data
EXPOSE 7860
CMD ["ohamlab-voice", "studio", "--model", "rahul7star/tts-nano-o-hi", "--device", "cpu"]
