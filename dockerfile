FROM python:3.9

COPY minimal_ai.py ./

RUN apt-get update && \
    apt-get install ffmpeg libsm6 libxext6 -y && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir onnxruntime rtsp numpy paho-mqtt requests

CMD python -u minimal_ai.py $RTSP_STREAM $MQTT_BROKER $MQTT_TOPIC
# CMD ["python","./raspi_monitor.py"]