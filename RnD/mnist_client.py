'''
    Client code to test the mnist model served via tenorflow serving.
    It takes the test image from mnist dataset and call REST api for predictions.
    TO DO: TO MAKE A PREDICTION USING A FILE FROM THE DISK INSTEAD OF PICKING FROM DATASET.
'''
import warnings
warnings.filterwarnings('ignore',category=FutureWarning) # to ignore the numpy warnings
import json
import numpy as np
import requests
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.utils import to_categorical

# loading test images from mnist dataset and randomly choosing one and call serving
# to get predictions
# loading traing data and test data from mnist dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# preprocessing the image data
train_images = train_images.reshape((60000,28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000,28 * 28))
test_images = test_images.astype('float32') / 255

# preprocessing labels by catagorically encoding them
train_labels = to_categorical(train_labels)
test_labels  = to_categorical(test_labels)

def show(idx, title):
    plt.figure()
    plt.imshow(test_images[idx].reshape(28,28))
    plt.axis('off')
    plt.title('\n\n{}'.format(title), fontdict={'size': 16})
    plt.show()

# sending post request to TensorFlow Serving server
data = json.dumps({"signature_name": "serving_default", "instances": test_images[0:3].tolist()})
headers = {"content-type": "application/json"}
json_response = requests.post('http://localhost:9000/v1/models/MnistClassifier:predict', data=data, headers=headers)
predictions = json.loads(json_response.text)['predictions']
print(predictions)
for i in range(0,3):
    show(i, 'The model thought this was a {} (class {}), and it was actually a {} (class {})'.format(
  np.argmax(predictions[i]), test_labels[i], np.argmax(predictions[i]), test_labels[i]))
