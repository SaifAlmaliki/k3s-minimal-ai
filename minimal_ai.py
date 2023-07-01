import sys
import rtsp
import onnxruntime as ort
import numpy as np
import paho.mqtt.client as mqtt
import requests
from preprocess import preprocess
import os



if __name__ == '__main__':
    # if len(sys.argv) != 4:
    #    raise ValueError("This demo app expects 3 arguments and has %d" % (len(sys.argv) - 1))
    
    # Load in the command line arguments
    # rtsp_stream, mqtt_broker, mqtt_topic = sys.argv[1], sys.argv[2], sys.argv[3]
    
    # hostname = os.environ["NODE_NAME"]
    
    mqtt_broker = 'mosquitto-1687368643'
    mqtt_port: 1883
    # rtsp_stream = 'https://user-images.githubusercontent.com/711/208944092-94e352fa-4405-42ec-8e60-4a178733248d.gif' # freight car, amphibian, amphibious vehicle
    rtsp_stream = 'http://158.58.130.148/mjpg/video.mjpg'
    # rtsp_stream = 'http://pendelcam.kip.uni-heidelberg.de/mjpg/video.mjpg'  # Wall Clock
    
    mqtt_topic = 'raspberry/onnx/pred'
    
    # Download the model
    model= requests.get('https://github.com/onnx/models/raw/main/vision/classification/inception_and_googlenet/googlenet/model/googlenet-12.onnx')
    open('model.onnx', 'wb').write(model.content)
    session = ort.InferenceSession('model.onnx')
    inname = [input.name for input in session.get_inputs()]
    
    # Download the class names
    labels = requests.get('https://raw.githubusercontent.com/onnx/models/main/vision/classification/synset.txt')
    open('synset.txt', 'wb').write(labels.content)
    with open('synset.txt', 'r') as f:
        labels = [l.rstrip() for l in f]
    
    # Connect to the MQTT Broker
    # mqtt_client = mqtt.Client('minimal_ai')
    # mqtt_client.connect(mqtt_broker, mqtt_port, 60)   
    # mqtt_client.connect(mqtt_broker)
    # mqtt_client.loop_start()
    
    # Connect to the RTSP Stream
    rtsp_client = rtsp.Client(rtsp_server_uri= rtsp_stream)
    while rtsp_client.isOpened():
        
        # read a frame from the RTSP stream
        img = rtsp_client.read()
        if img != None:
            
            # preprocess the image
            img = preprocess(img)
            
            # run the model inference, extract most likely class
            predictions = session.run(None, {inname[0]: img})
            pred = np.squeeze(predictions)
            a = np.argsort(pred)[::-1]
            
            # print output and publish to MQTT broker
            print(labels[a[0]])
            # mqtt_client.publish(mqtt_topic, str(labels[a[0]]))
    
    rtsp_client.close()
    # mqtt_client.disconnect()