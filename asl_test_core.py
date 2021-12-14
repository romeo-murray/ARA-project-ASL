import os
import pickle
import matplotlib.pyplot as plt

os.environ["TF-CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers
print("TensorFlow version: ", tf.__version__)

"""
@authors: Romeo Garcia, Ashwin Deodhar, Angela You

References: [Youtube Users - Aladdin Persson, sentdex]
TensorFlow documentation, Keras documentation, StackOverFlow
Dataset provided from: Akash Nagaraj and Anas Khan

This is an updated version of the code that allowed for ASL to be translated from images
using a CNN. The code was made and last updated in September of 2020, and tensorflow
has since drastically updated.

The convolutional neural network that is built here is not the most efficient, but it was utilized
to train a model that had an accuracy rate of 98% based on tensorflow's tests, and from our own
personal testing, has a real accuracy rate of a flat 90%. This could most likely be improved
by simply training a dataset with higher pixel amounts, but due to time constraints, we
withheld from training at the original 200x200, and compressed down to 50x50.

Anothing that that could be improved would be the CPU use, as we were not able to properly
set up GPU use as the ones we had (NVIDIA) had updated beyond what tensorflow accepts. This
means that all the code must rely solely on the CPU for training, which isn't great.
"""

# Load in the files from both the test set and training set
X = pickle.load(open("X.pickle", "rb"))
Y = pickle.load(open("Y.pickle", "rb"))

X_test = pickle.load(open("X_test.pickle", "rb"))
Y_test = pickle.load(open("Y_test.pickle", "rb"))

with tf.device("cpu:0"):
    # create the model
    model = models.Sequential()
    model.add(layers.Conv2D(32,(3,3), activation='relu', input_shape=(50,50,3)))
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Conv2D(64, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Conv2D(64, (3,3), activation='relu'))
    
    # adding dense layers
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(29))
    
    # a summary of the model constructed
    model.summary()
    
    # compile the model
    model.compile(
        optimizer='adam', #Optimizer
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), # loss function to minimize
        metrics = ['accuracy'] # list of metrics to monitor
    )
    
    # train the model
    history = model.fit(
        X, # the input data, our trained data
        Y,
        epochs=10,
        batch_size=32,
        validation_data=(X_test, Y_test) # actually testing the data to validate
    )
    
    # an evaluation of the model
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    
    test_loss, test_acc = model.evaluate(X_test,  Y_test, verbose=2)
    
    # print the results
    print(test_acc)
    
    # then we save the model
    model.save('C:/your/chosen/directory/path/here/')
