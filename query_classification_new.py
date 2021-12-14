import sys # For time
import os # For reading files
import numpy as np # for creating arrays
from gtts import gTTS   # Import Google Text to Speech

# Disable tensorflow compilation errors
os.environ["TF-CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image

"""
@author: Romeo, Ashwin, Angela
References: Youtube Users - Aladdin Persson, sentdex, TensorFlow documentation, Keras documentation, StackOverFlow
Dataset provided from: Akash Nagaraj and Anas Khan

This is a testing file that will run the model we trained against test images.

"""

# Language used by Google Text to Speech
language = 'en'

# Image to be classified
image_path = sys.argv[1]

# load up the newest model
model = keras.models.load_model("C:\your\directory\here\\") # put where you store the saved-model here

# convert the image to a tensorflow readable format
image_data = image.load_img(image_path, target_size=(50, 50))
image_data = np.expand_dims(image_data, axis=0)

# Load training labels file
label_lines = [line.rstrip() for line in tf.io.gfile.GFile("training_set_labels.txt")]

# Get prediction by decoding the jpg image
predictions = model.predict(image_data)

# Sort the predictions in descending order based on score
sorted_predictions = predictions[0].argsort()[-len(predictions[0]):][::-1]

# Print the predicted letter and the score
print("\n\nPredicted Letter: ", str(label_lines[sorted_predictions[0]]).upper(), "\tScore: ", predictions[0][sorted_predictions[0]], "\n\n")

# Create the text to be spoken
prediction_text = "The predicted letter is " + str(label_lines[sorted_predictions[0]])

# Create a speech object from text to be spoken
speech_object = gTTS(text=prediction_text, lang=language, slow=False)

# Save the speech object in a file called 'prediction.mp3'
speech_object.save("prediction.mp3")
 
# Playing the speech using mpg321
os.system("mpg321 prediction.mp3")

# Display the letters and the score for each prediction
'''
for letter_prediction in sorted_predictions:
    letter = label_lines[letter_prediction]
    score = predictions[0][letter_prediction]
    print('%s (score = %.5f)' % (letter, score))'''
