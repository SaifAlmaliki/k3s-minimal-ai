import sys
import rtsp
import onnxruntime as ort
import numpy as np
import paho.mqtt.client as paho
from paho import mqtt
import requests
from preprocess import preprocess
import os

# Callback for successfully connection to HiveMQ MQTT cluster
def on_connect(client, userdata, flags, rc, properties=None):
    print("Connection Acknowledgment received with code %s." % rc)
    
# with this callback you can see if your publish was successful
def on_publish(client, userdata, mid, properties=None):
    print("mid: " + str(mid))

# print which topic was subscribed to
def on_subscribe(client, userdata, mid, granted_qos, properties=None):
    print("Subscribed: " + str(mid) + " " + str(granted_qos))

# print message, useful for checking if it was successful
def on_message(client, userdata, msg):
    print(msg.topic + " " + str(msg.qos) + " " + str(msg.payload))

if __name__ == '__main__':
    if len(sys.argv) != 4:
        raise ValueError("This demo app expects 3 arguments and has %d" % (len(sys.argv) - 1))
    
    # Load in the command line arguments
    rtsp_stream, mqtt_broker, mqtt_topic = sys.argv[1], sys.argv[2], sys.argv[3]
    

    
    # hostname = os.environ["NODE_NAME"]
    
    # mqtt_broker = 'mosquitto-1687368643'
    # mqtt_port: 1883
    # rtsp_stream = 'https://user-images.githubusercontent.com/711/208944092-94e352fa-4405-42ec-8e60-4a178733248d.gif' # freight car, amphibian, amphibious vehicle
    # rtsp_stream = 'http://158.58.130.148/mjpg/video.mjpg'
    # rtsp_stream = 'http://pendelcam.kip.uni-heidelberg.de/mjpg/video.mjpg'  # Wall Clock
    # mqtt_topic = 'raspberry/onnx/pred'
    
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
    
    # using MQTT version 5 here, for 3.1.1: MQTTv311, 3.1: MQTTv31
    # userdata is user defined data of any type, updated by user_data_set()
    # client_id is the given name of the client
    mqtt_client = paho.Client(client_id="rpi-k3s-cluster", userdata=None, protocol=paho.MQTTv5)
    mqtt_client.on_connect = on_connect
    
    # enable TLS for secure connection
    mqtt_client.tls_set(tls_version=mqtt.client.ssl.PROTOCOL_TLS)
    # set username and password
    mqtt_client.username_pw_set("saifwsm", "P@ssw0rd")
    
    # Connect to the HiveMQ cloud MQTT Broker
    mqtt_client.connect(mqtt_broker, 8883)
    
    # setting callbacks, use separate functions like above for better visibility
    mqtt_client.on_subscribe = on_subscribe
    mqtt_client.on_message = on_message
    mqtt_client.on_publish = on_publish
    
    # subscribe to all topics of raspberry by using the wildcard "#"
    mqtt_client.subscribe("raspberry/#", qos=1)
    
    mqtt_client.loop_start()
    
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
            mqtt_client.publish(mqtt_topic, str(labels[a[0]]), qos=1)
    
    rtsp_client.close()
    mqtt_client.disconnect()