from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical
from keras.models import model_from_json

# loading traing data and test data from mnist dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# create the network
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

# network compilation
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

# preprocessing the image data
train_images = train_images.reshape((60000,28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000,28 * 28))
test_images = test_images.astype('float32') / 255

# preprocessing labels by catagorically encoding them
train_labels = to_categorical(train_labels)
test_labels  = to_categorical(test_labels)

# train the network
network.fit(train_images, train_labels, epochs=5, batch_size=128)

# serialize network model to JSON
model_json = network.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
network.save_weights("model.h5")
print("Saved model to disk")

# test the network with test data
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test acc: ', test_acc)

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data, note that we have to compile the
# loaded model first before evaluating it
loaded_model.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
test_loss, test_acc = loaded_model.evaluate(test_images, test_labels)
print('test acc: ', test_acc)