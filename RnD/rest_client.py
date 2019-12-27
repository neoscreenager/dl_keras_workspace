'''
    Client to make predictions using a file from the disk and trained MNIST model
    served via tensorflow model serving.
    Images are pre-processed first and then used for recognition.

'''

import warnings
warnings.filterwarnings('ignore',category=FutureWarning) # to ignore the numpy warnings
import json
import numpy as np
import requests
from keras.preprocessing import image
import cv2
import math
from scipy import ndimage


#image_path = 'digit_7.png'
image_path = '5.jpg'
# Preprocessing our input image
## img = image.img_to_array(image.load_img(image_path, target_size=(224, 224))) / 255.

# this line is added because of a bug in tf_serving(1.10.0-dev)
# img = img.astype('float32')

# pre-processing the image(s) using open cv
images = np.zeros((1,784)) # array of image(s), one dimensional vector 28 x 28
# correct value array, for example, for 9, the correct value array should be [0,0,0,0,0,0,0,0,0,1]
correct_vals = np.zeros((1,10))
# read the image
gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
# resize the images and invert it (black background)
gray = cv2.resize(255-gray, (28, 28))
# makeing the background black and foreground white to make it look like the test images
(thresh, gray) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
'''
    All images are size normalized to fit in a 20x20 pixel box and there are centered in a 28x28 
    image using the center of mass. These are important information for our pre-processing.
'''
'''
    First we want to fit the images into this 20x20 pixel box. Therefore we need 
    to remove every row and column at the sides of the image which are completely black.
'''
while np.sum(gray[0]) == 0:
    gray = gray[1:]
while np.sum(gray[:,0]) == 0:
    gray = np.delete(gray,0,1)
while np.sum(gray[-1]) == 0:
    gray = gray[:-1]
while np.sum(gray[:,-1]) == 0:
    gray = np.delete(gray,-1,1)
rows,cols = gray.shape
print(rows, cols)
'''
    Now we want to resize our outer box to fit it into a 20x20 box. We need a resize factor for this.
'''
if rows > cols:
    factor = 20.0/rows
    rows = 20
    cols = int(round(cols*factor))
    gray = cv2.resize(gray, (cols,rows))
else:
    factor = 20.0/cols
    cols = 20
    rows = int(round(rows*factor))
    gray = cv2.resize(gray, (cols, rows))
'''
    At the end we need a 28x28 pixel image so we add the missing black 
    rows and columns using the np.lib.pad function which adds 0s to the sides.
'''
colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))
rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))
gray = np.lib.pad(gray,(rowsPadding,colsPadding),'constant')
'''
    We need two functions for this last step. The first one will get 
    the center_of_mass mass which is a function in the library ndimage from scipy
'''
def getBestShift(img):
    cy,cx = ndimage.measurements.center_of_mass(img)

    rows,cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)

    return shiftx,shifty
'''
    The second functions shifts the image in the given directions
'''
def shift(img,sx,sy):
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted
gray = np.lib.pad(gray,(rowsPadding,colsPadding),'constant')
shiftx,shifty = getBestShift(gray)
shifted = shift(gray,shiftx,shifty)
gray = shifted
gray = cv2.resize(gray,(28,28))
# save the processed images
cv2.imwrite("PREPROCESSED_"+image_path, gray)
"""
    all images in the training set have an range from 0-1
    and not from 0-255 so we divide our flatten images
    (a one dimensional vector with our 784 pixels)
    to use the same 0-1 based range
"""
flatten = gray.flatten() / 255.0
"""
    we need to store the flatten image and generate
    the correct_vals array
    correct_val for the first digit (9) would be
    [0,0,0,0,0,0,0,0,0,1]
"""
images[0] = flatten
correct_val = np.zeros((10))
correct_val[7] = 1 # since our image is digit 7, it could be made dynamic
correct_vals = correct_val
payload = {
    "instances": [{'input_image': images}]
}

# sending post request to TensorFlow Serving server
data = json.dumps({"signature_name": "serving_default", "instances": images.tolist()})
headers = {"content-type": "application/json"}
json_response = requests.post('http://localhost:9000/v1/models/MnistClassifier:predict', data=data, headers=headers)
predictions = json.loads(json_response.text)['predictions']
print(predictions)
print(np.argmax(predictions))

#r = requests.post('http://localhost:9000/v1/models/MnistClassifier:predict', json=payload)
#pred = json.loads(r.content.decode('utf-8'))

#print(pred)
# Decoding the response
# decode_predictions(preds, top=5) by default gives top 5 results
# You can pass "top=10" to get top 10 predicitons
#print(json.dumps(inception_v3.decode_predictions(np.array(pred['predictions']))[0]))
