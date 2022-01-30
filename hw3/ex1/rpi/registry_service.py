import cherrypy
import adafruit_dht
from board import D4
from datetime import datetime
import time
import json
import os
import base64
import tensorflow as tf
import numpy as np
from alertPublisher import AlertPublisher

class AddService(object):
    exposed=True
        
    def GET(self, *path, **query):
        pass
    
    def POST(self, *path, **query):

        if len(path) > 0:
            raise cherrypy.HTTPError(400, 'Wrong path')

        if len(query) > 0:
            raise cherrypy.HTTPError(400, 'Wrong query')

        subfolder='./models/'
        body = cherrypy.request.body.read()
        body = json.loads(body)
        model = body.get('model')
        name = body.get('name')

        if not os.path.exists(subfolder):
            os.mkdir(subfolder)     
        if model is None:
            raise cherrypy.HTTPError(400, 'model missing')
        if name is None:
            raise cherrypy.HTTPError(400, 'name missing')

        model_64bytes = base64.b64decode(model)

        path = subfolder+name
        with open(path, 'wb') as f:
            f.write(model_64bytes)

    
    def PUT(self, *path, **query):
        pass

    def DELETE(self, *path, **query):
        pass


class ListService(object):
    exposed = True

    def GET(self, *path, **query):

        if len(path) > 0:
            raise cherrypy.HTTPError(400, 'Wrong path')

        if len(query) > 0:
            raise cherrypy.HTTPError(400, 'Wrong query')
        
        subfolder = "./models"
            
        if not os.path.exists(subfolder):
            raise cherrypy.HTTPError(400, 'Directory {} missing'.format(subfolder))

        files = os.listdir(subfolder)
        models = []
        for file in files:
            name, ext = file.split(".")
            if ext == "tflite":   
                models.append(name)
            
        body = {'models': models}
        body_json = json.dumps(body)

        return body_json

    def POST(self, *path, **query):
        pass

    def PUT(self, *path, **query):
        pass

    def DELETE(self, *path, **query):
        pass


class PredictService(object):

    exposed=True

    def __init__(self):
        self.dht_device = adafruit_dht.DHT11(D4)
        self.test = AlertPublisher("AlertPublisher")
        self.test.start()
        self.i = 0 # index used to fill the window (for prediction)

    def GET(self, *path, **query):

        if len(path) > 0:
            raise cherrypy.HTTPError(400, 'Wrong path')

        if len(query) != 3:
            raise cherrypy.HTTPError(400, 'Wrong query')

        model = query.get('model')
        tthres = np.float32(query.get('tthres'))
        hthres = np.float32(query.get('hthres'))

        # load model and define window
        interpreter = tf.lite.Interpreter(model_path='./models/{}.tflite'.format(model))
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        window = np.zeros([1, 6, 2], dtype=np.float32)
        expected = np.zeros(2, dtype=np.float32)
        MEAN = np.array([9.107597, 75.904076], dtype=np.float32)
        STD = np.array([ 8.654227, 16.557089], dtype=np.float32)

        self.i = 0
        # Start simulation
        while True:
            temperature = self.dht_device.temperature
            humidity = self.dht_device.humidity
            timestamp = int((datetime.now()).timestamp())

            if self.i <= 5:
                window[0, self.i , 0] = np.float32(temperature)
                window[0, self.i, 1] = np.float32(humidity)
                self.i += 1

            else:
                expected[0] = np.float32(temperature)
                expected[1] = np.float32(humidity)

                temp_window = (window - MEAN) / STD
                interpreter.set_tensor(input_details[0]['index'], temp_window)
                interpreter.invoke()
                predicted = interpreter.get_tensor(output_details[0]['index'])

                # Compute absolute errors for temperature and humidity predicition
                abs_err_t = abs(expected[0]-predicted[0,0])
                abs_err_h = abs(expected[1]-predicted[0,1])
                
                # check if the thresholds are exceeded
                if abs_err_t > tthres:
                    # pack message into SENML+JSON STRING
                    message = {
                        'bn': 'raspberrypi.local',
                        'bt': timestamp,
                        'e':[
                            {'n': 'temperature', 
                             'u': '°C', 
                             't': 0, 
                             'v': float(predicted[0,0])},
                             {'n': 'temperature', 
                             'u': '°C', 
                             't': 0, 
                             'v': float(expected[0])}  
                        ]
                    }
                    message = json.dumps(message)
                    self.test.publish('/temperature_alert', message)

                if abs_err_h > hthres:
                    # pack message into SENML+JSON STRING
                    message = {
                        'bn': 'raspberrypi.local',
                        'bt': timestamp,
                        'e':[
                            {'n': 'humidity', 
                             'u': '%', 
                             't': 0, 
                             'v': float(predicted[0,1])},
                             {'n': 'humidity', 
                             'u': '%', 
                             't': 0, 
                             'v': float(expected[1])}   
                        ]
                    }
                    message = json.dumps(message)
                    self.test.publish('/humidity_alert', message)

                # Update the window (slide all values to the left of 1 position)
                # We are keeping only the last 5 values (1st element of the window is now in the last position of the window)
                window[0, :, 0] = np.roll(window[0, :, 0], -1)
                window[0, :, 1] = np.roll(window[0, :, 1], -1)

                # The last value of the window (which is the 1st element slided) is now overwritten with the new measurement
                window[0, -1, 0] = np.float32(temperature)
                window[0, -1, 1] = np.float32(humidity)

            time.sleep(1)
        
 
    
    def POST(self, *path, **query):
        pass

    def PUT(self, *path, **query):
        pass

    def DELETE(self, *path, **query):
        pass

if __name__ == '__main__':
    conf = {'/': {'request.dispatch': cherrypy.dispatch.MethodDispatcher()}}

    cherrypy.tree.mount(AddService(), '/add', conf)
    cherrypy.tree.mount(ListService(), '/list', conf)
    cherrypy.tree.mount(PredictService(), '/predict', conf)

    cherrypy.config.update({'server.socket_host': '0.0.0.0'})
    cherrypy.config.update({'server.socket_port': 8080})
    cherrypy.engine.start()
    cherrypy.engine.block()