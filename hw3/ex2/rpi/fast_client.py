import time
import json
from datetime import datetime
import numpy as np
import os
import tensorflow as tf
import base64 
import wave
import paho.mqtt.client as PahoMQTT
from utils import SignalGenerator, load_and_evaluation

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)


class Client():
    
    def __init__(self, clientID, confident_outputs, labels, count_non_confident):
        self.clientID = clientID
        self.confident_outputs = confident_outputs
        self.count_non_confident = count_non_confident
        self.received = 0
        self.labels = labels

        # create an instance of paho.mqtt.client
        self._paho_mqtt = PahoMQTT.Client(self.clientID, False) 
        # register the callback
        self._paho_mqtt.on_connect = self.myOnConnect
        self._paho_mqtt.on_message = self.myOnMessageReceived
        self.messageBroker = 'test.mosquitto.org' #'broker.emqx.io' 

    def start (self):
        #manage connection to broker
        self._paho_mqtt.connect(self.messageBroker, 1883)
        self._paho_mqtt.loop_start()
        self._paho_mqtt.subscribe('/prediction', 2)

    def stop (self):
        self._paho_mqtt.unsubscribe('/prediction')
        self._paho_mqtt.loop_stop()
        self._paho_mqtt.disconnect()

    def myPublish(self, topic, msg):
        # publish a message with a certain topic  
        self._paho_mqtt.publish(topic, msg, 2)

    def myOnConnect (self, paho_mqtt, userdata, flags, rc):
        # print ("Connected to %s with result code: %d" % (self.messageBroker, rc))
        pass

    def collaborative_accuracy(self):
        outputs = np.squeeze(self.confident_outputs)
        labels = np.squeeze(np.array(self.labels))
        acc = sum(np.equal(labels, np.argmax(outputs, axis = 1)))/len(outputs)
        print('Accuracy: {:.3f}%'.format(acc*100))

    def myOnMessageReceived (self, paho_mqtt , userdata, msg):
        # A new message is received
        self.received = self.received + 1 
        input_json = json.loads(msg.payload)
        output = json.loads(input_json['vd'])
        
        self.confident_outputs.append(np.array(output))
        if self.received == self.count_non_confident:
            self.collaborative_accuracy()
        

        
if __name__ == "__main__":

    # download the mini speech commands dataset
    zip_path = tf.keras.utils.get_file(
    origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
    fname='mini_speech_commands.zip',
    extract=True,
    cache_dir='.', cache_subdir='data')

    # generation of the files used to generate the provided
    kws_test_split = open("kws_test_split.txt", "r")
    test_lines = kws_test_split.read().splitlines()
    test_files = tf.convert_to_tensor(test_lines)
    num_samples_test = test_files.shape
    
    # Label mapping
    labels = open("labels.txt", "r")
    labels = str(labels.read())
    characters_to_remove = "[]''""  "
    for character in characters_to_remove: 
        labels = labels.replace(character, "")
    LABELS = labels.split(",")
    
    # parameter used for the fast preprocessing
    options = {'frame_length': 320, 'frame_step': 160, 'mfcc': True,
                    'lower_frequency': 20, 'upper_frequency': 4000, 'num_mel_bins': 16,
                    'num_coefficients': 10, 'sampling_rate' : 8000}
    
    # Create test dataset through SignalGenerator class
    generator = SignalGenerator(LABELS, **options)
    test_ds = generator.make_dataset(test_files, False)
    model_dir = os.path.join('.', 'kws_dscnn_True.tflite')

    # Evaluation of the tflite model
    confident_outputs, labels, not_confident_idx = load_and_evaluation(model_dir, test_ds)
    count_non_confident = len(not_confident_idx)
    comunication_cost = 0
    
    test = Client("Fast client", confident_outputs, labels, count_non_confident)
    test.start()
    

    # Creation of the SENML + JSON message to be published with audio bites whose prediction 
    # were not confident 
    for idx in not_confident_idx:
        with open(test_lines[idx], 'rb') as fd:
            audio = fd.read()
        audio_b64bytes = base64.b64encode(audio)
        audio_string = audio_b64bytes.decode()

        now = datetime.now()
        timestamp = int(now.timestamp())

        # pack data into SENML+JSON STRING
        message = {
            'bn': 'fast_client', #unique identifier of the board/sensor who's sending the data
            'bt': timestamp, # timestamp of different events (basetimestamp + offset)
            'e': [
                # for every event we need to define name, unit, offset, value
                {'n': 'audio_'+ str(idx), 'u': '/', 't': 0, 'vd': audio_string}
            ]
        }
        
        message_string = json.dumps(message)
        comunication_cost = comunication_cost + len(message_string)
        time.sleep(0.1)
        test.myPublish("/not_confident", message_string)

    print('Communication cost: {:.3f} MB'.format(comunication_cost/1000000))
    
    while True:
        time.sleep(5)
    # test.stop()