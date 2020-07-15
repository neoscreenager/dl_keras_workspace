'''
    The train_mnist function do following things:
    1. It creates the network and trains it with MNIST dataset available in Keras.
    2. Save the model to the disk in .h5 format.
    3. Load the saved model and test its accuracy.
    4. Export this tested keras model to tensorflow model so that it could be served
       using tensorflow serving.
    
'''

import warnings
warnings.filterwarnings('ignore',category=FutureWarning) # to ignore the numpy warnings
from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical
from keras.models import model_from_json
import tensorflow as tf

def train_mnist(version):
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

    # saving model as whole so that it could be exported in tensorflow
    # for tensorflow serving

    network.save("model_for_serving.h5")

    # test the network with test data
    test_loss, test_acc = network.evaluate(test_images, test_labels)
    print('test acc: ', test_acc)
    print('test_loss: ', test_loss)

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

    # export this tested model to tensorflow model so that it could be served using
    # tensorflow serving

    # The export path contains the name and the version of the model
    tf.keras.backend.set_learning_phase(0)  # Ignore dropout at inference
    keras_model = tf.keras.models.load_model('model_for_serving.h5')
    export_path = 'mnist_image_classifier/'+version
    print(export_path)
    
    # Fetch the Keras session and save the model
    # The signature definition is defined by the input and output tensors
    # And stored with the default serving key
    '''
        simple_save is deprecated, it will be available in
        v1, changed it to tf.compat.v1.saved_model
        Also, tf.keras.backend.get_session is deprecated
        changed it to tf.compat.v1.keras.backend.get_session
    '''
    # with tf.keras.backend.get_session() as sess:
    with tf.compat.v1.keras.backend.get_session() as sess:
        # tf.saved_model.simple_save(
        tf.compat.v1.saved_model.simple_save(
            sess,
            export_path,
            inputs={'input_image': keras_model.input},
            outputs={t.name: t for t in keras_model.outputs})

    '''
        Use following command to look at the MetaGraphDefs (the models)
        and SignatureDefs (the methods you can call) in our SavedModel:
        saved_model_cli show --dir mnist_image_classifier/1 --all
    '''

if __name__ == "__main__":
    train_mnist('6')
    