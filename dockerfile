FROM python:3.9

RUN apt-get update && \
    apt-get install ffmpeg libsm6 libxext6 -y && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir onnxruntime rtsp numpy paho-mqtt requests

COPY minimal_ai.py ./

CMD ["python","./raspi_monitor.py"]