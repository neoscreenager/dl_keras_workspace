import warnings
warnings.filterwarnings('ignore',category=FutureWarning) # to ignore the numpy warnings
import json
import numpy as np
import requests
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

def load_image(filename):
	# load the image
	img = load_img(filename, color_mode="grayscale", target_size=(28, 28))
	# convert to array
	img = img_to_array(img)
	# reshape into a single sample with 1 channel
	img = img.reshape(28, 28)
	# prepare pixel data
	img = img.astype('float32')
	img = img / 255.0
	return img

# Preprocessing our input image
img = load_image("./digit_7.png")
# print(img.tolist())
# headers = {"content-type": "application/json"}
# data = json.dumps({"signature_name": "serving_default", "instances": [{'input_image': img.tolist()}]})
#data = json.dumps({"instances": [{'input_image': img.tolist()}]})
#data = json.dumps({"instances": img.tolist()})
# data = {'instances': [{'input_image': img.tolist()}]}
payload = {
    "instances": [{'input_image': img.tolist()}]
}

# sending post request to TensorFlow Serving server
json_response = requests.post('http://localhost:9000/v1/models/MnistClassifier:predict', json=payload) #, headers=headers)
print(json_response)
predictions = json.loads(json_response.text)['predictions']
print(predictions)
#pred = json.loads(r.content.decode('utf-8'))
#print(pred)
# Decoding the response
# decode_predictions(preds, top=5) by default gives top 5 results
# You can pass "top=10" to get top 10 predicitons
# print(json.dumps(inception_v3.decode_predictions(np.array(pred['predictions']))[0]))
#print(json.dumps(np.array(pred['predictions'])[0]))