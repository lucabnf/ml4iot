import datetime
import json
import paho.mqtt.client as PahoMQTT
from datetime import datetime

class AlertSubscriber:
	def __init__(self, clientID, topics):
		self.clientID = clientID
		# create an instance of paho.mqtt.client
		self._paho_mqtt = PahoMQTT.Client(clientID, False) 
		# register the callback
		self._paho_mqtt.on_connect = self.myOnConnect 
		self._paho_mqtt.on_message = self.myOnMessageReceived 
		
		self.topics = topics
		self.messageBroker = 'test.mosquitto.org'

	def start (self):
		#manage connection to broker
		self._paho_mqtt.connect(self.messageBroker, 1883)
		self._paho_mqtt.loop_start()
		# subscribe for all the topics
		for topic in self.topics:
			self._paho_mqtt.subscribe(topic, 2)
	
	def stop (self):
		for topic in self.topics:
			self._paho_mqtt.unsubscribe(topic)
		self._paho_mqtt.loop_stop()
		self._paho_mqtt.disconnect()

	def myOnConnect (self, paho_mqtt, userdata, flags, rc):
		print ("Connected to %s with result code: %d" % (self.messageBroker, rc))
		
	def myOnMessageReceived (self, paho_mqtt , userdata, msg): # define the preprocessing on the message received
		# A new message is received
		payload = json.loads(msg.payload)
		t = payload["bt"]
		n = payload["e"][0]["n"]
		u = payload["e"][0]["u"]
		v_predicted = payload["e"][0]["v"]
		v_actual = payload["e"][1]["v"]

		n = n.capitalize() + " alert"
		dt = datetime.fromtimestamp(t)

		print ("("+str(dt)+") " + n +": "+ "Predicted={:.3f}".format(v_predicted)+u+" Actual={}".format(v_actual)+u)


# ---------------- predict request ----------------
test = AlertSubscriber("AlertSubscriber", ["/temperature_alert", "/humidity_alert"])
test.start()

while True:
	pass