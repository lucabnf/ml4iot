import base64
import requests

# ---------------- add request ----------------
print("--- Add request ---")
print("1) Load mlp model")
mlp_name = './2_mlp.tflite'

# load and convert cnn model (as string)
with open(mlp_name, 'rb') as f:
    mlp_model = f.read()
mlp_64bytes = base64.b64encode(mlp_model)

# url for the add request model
add_url = 'http://raspberrypi.local:8080/add'

# define body of the request
body = {'model': str(mlp_64bytes.decode('utf-8')),
        'name': mlp_name}

r = requests.post(add_url , json=body)
if r.status_code == 200:
    print("Code: ", r.status_code)
else:
    print('Error:', r.status_code)


print("2) Load cnn model")
cnn_name = './2_cnn.tflite'

# load and convert model 1 (as string)
with open(cnn_name, 'rb') as f:
    cnn_model = f.read()
cnn_64bytes = base64.b64encode(cnn_model)

# define body of the request
body = {'model': str(cnn_64bytes.decode('utf-8')),
        'name': cnn_name}

r = requests.post(add_url , json=body)
if r.status_code == 200:
    print("Code:", r.status_code)
else:	
    print('Error:', r.status_code)


# ---------------- list request ----------------
print("--- List request ---")
list_url = 'http://raspberrypi.local:8080/list'

r = requests.get(list_url)

if r.status_code == 200:
    body = r.json()
    print("Code:", r.status_code)
    models = body['models']
    if len(models) == 2:
        print("Two models correctly uploaded. Models: ", models)
    else:
        print("Something went wrong in the upload of the two models. Models: ", models)

else:
    print('Error:', r.status_code)


# ---------------- inference request ----------------
predict_url = 'http://raspberrypi.local:8080/predict/?model=2_cnn&tthres=0.1&hthres=0.2'

print("Inference started... Run 'monitoring_client.py' to show the alerts")

r = requests.get(predict_url)

if r.status_code == 200:
    print("Code: ", r.status_code)
else:
    print("Error: ", r.status_code)