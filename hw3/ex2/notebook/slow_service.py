import time
import json
import numpy as np
import tensorflow as tf
import base64 
import paho.mqtt.client as PahoMQTT
from utils import SignalGenerator_slow
from utils import predict


seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)


class Server:
    def __init__(self, serverID):
        self.serverID = serverID
        self.count_messages = 0
        
        # create an instance of paho.mqtt.client
        self._paho_mqtt = PahoMQTT.Client(self.serverID, False) 
        
        # register the callback
        self._paho_mqtt.on_connect = self.myOnConnect
        self._paho_mqtt.on_message = self.myOnMessageReceived
        self.topic = '/not_confident'

        self.messageBroker = 'test.mosquitto.org' #'broker.emqx.io' 


    def start (self):
        #manage connection to broker
        self._paho_mqtt.connect(self.messageBroker, 1883)
        self._paho_mqtt.loop_start()
        self._paho_mqtt.subscribe('/not_confident', 2)

    def stop (self):
        self._paho_mqtt.unsubscribe('/prediction')
        self._paho_mqtt.loop_stop()
        self._paho_mqtt.disconnect()

    def myPublish(self, topic, message):
        # publish a message with a certain topic
        self._paho_mqtt.publish(topic, message, 2) 

    def myOnConnect(self, paho_mqtt, userdata, flags, rc):
        # print ("Connected to %s with result code: %d" % (self.messageBroker, rc)

        pass

    def myOnMessageReceived(self, paho_mqtt , userdata, msg):
        # parameter used for the slow preprocessing
        options = {'frame_length': 640, 'frame_step': 320,
            'lower_frequency': 20, 'upper_frequency': 4000, 'num_mel_bins': 40,
            'num_coefficients': 10, 'sampling_rate' : 16000}
        
        input_json = json.loads(msg.payload)
        
        audio_bytes = base64.b64decode(input_json['e'][0]['vd'])
        
        generator = SignalGenerator_slow(audio_bytes, **options)
        mfcc = generator.preprocess_with_mfcc()
        output_prediction = predict('kws_dscnn_True.tflite', mfcc)
        
        message = {'n': 'audio_'+ str(self.count_messages), 'vd': str(output_prediction)} 
        message_string = json.dumps(message)

        self.myPublish("/prediction", message_string) 
        

        
if __name__ == "__main__":
    server = Server("Web server")
    server.start()

    while True:
        time.sleep(5)