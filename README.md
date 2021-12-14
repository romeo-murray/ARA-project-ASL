# ARA-project-ASL
A program that translates ASL to English text, utilizing TensorFlow's built-in convolutional neural network. It is considerably a remaster of an older model made by @grassknoted and @anaskhan96.

## The Idea
Really, all we are attempting here is to get a CNN working, and use an older dataset from a different ASL project. The reason that there is any need for this is that TensorFlow has updated since the original projects debut, meaning those are no longer operable because of the changes TensorFlow had undergone. So, we are attempting to recreate the model and use a similar format with Google Text to Speech for the prediction process.

### Notes
Firstly, this project will not have a live demo as there wasn't enough time to complete it to that extent. Secondly, because TensorFlow has not updated to the current version of CUDA, there are GPU restrictions and cannot be operated using a GPU. So, this version forces TensorFlow to operate using the CPU, consider this fair warning if attempting to use this on your own device.

Thirdly, this is a school oriented project. This means that *no, this is not professional level coding*, and there are most likely mistakes or possible resource leaks, or issues due to there being a massive lack of time. If there is something that is easy to fix, feel free to correct us, or take the code and improve it yourselves!

# How to Run
# Firstly, all the installs you will need to run the files:

Python (this is redundant considering the project, but still)

PIP (the shall we say, installer)

Tensorflow (this will include Keras)

{{everything under here can be installed via 'pip install'}}
pillow (for graphics/images)

PIL (subsidary of pillow)

GTTS (google text to speech)

playsound==1.2.2 (for playing audio)

pexpect (for more audio playing)

If there are still import issues, we apologize as it was quite the effort to get this all working, and most of the beginning was finding what was updated and would work still and what wouldn't.

# Secondly, the order to run the files:

Now, if all the imports are correct, we can move onward. First, you should run the asl_create_testset file. It should be as simple as opening the file, changing DATADIR to the path you have downloaded the asl_alphabet_test, and then running the file. Two files, *X_test.pickle* and *Y_test.pickle* should be created. You can uncomment the code at the bottom to ensure the data is able to be loaded back in.

If it all ran smoothly, then prepare the second longest part, loading in asl_create_dataset. It's the same process, first changing DATADIR to the path you downloaded asl_training_images. Then, you should be able to run it uninterrupted. This will take some time, as there is roughly 80,000 files for it to handle, compress, and churn into a dataset. It will produce two files as well, *X.pickle*, and *Y.pickle*.

Now, to create the model: Run the asl_test_core file. Here, you need to again change the directory path at the very bottom this time, on the line model.save("C:/your/path/here/"). Once you have done this, you should be able to run the file, and it will create a file called *saved_model.pb*. This is the final step, note that your prompt should also show you the EPOCHS and general summary of the model that is created upon running this file. If errors occur, then trace them back and see if it is the data, or the model. If its the model, then you can go to TensorFlow's documentation of a CNN and compare it. If its the data, then you should check out sentdex on YouTube, as he was our reference for a good chunk of loading in datasets.

### How to get a prediction

Hooray! The model must be saved if you're here. This means the fun part, there will be a bunch of new files in your directory, but the most important at this point is the *saved_model.pb* and *training_set_labels.txt*. Go to the file query_classification_new and once again edit the directory, this time only leaving the directory path, *not* all the way to the saved_model file. It would look like ("C:\your\directory\\"), then change the path given to the playsound function at the bottom of the file, that one you should include the prediction.mp3 (that the file generates) in the path.

Then you should be able to run it, and suddenly google's speech will speak the predicted letter or phrase! It will also print to the console with a score.

### How to use your Own Testing Images
Create a new image of PNG or JPEG (or whatever images that TensorFlow accepts) and convert it to the size of 200x200 pixels. Then port it to the test images folder in the category of whatever letter your hand is gesturing. Lastly, go into the query_classification_new file.py and change the configuration settings on your IDE. In spyder for instance, you would click on the file, click on run>configuration per file> go down to general settings, and on the command line options you would put in the path to the testing image you created.

## Results
![image](https://user-images.githubusercontent.com/92128432/145925092-bfda24df-41fb-4341-8b20-912f443d27a7.png)

The accuracy rate in the end was able to increase to about 98%, but this was based on TensorFlow testing through evaluation and validation processes. After performing our own tests, we had found a few notable issues that remain unresolved. 

### Issues
We had found that out of all the options available (29 of them) 3 were inaccurate for reasons we don't know. This would mean that we had reached a roughly 90% accuracy rate generally speaking. Those three are M, N, and nothing. We theorized that the issue could stem from the quality of the images, as compressing them down to 50x50 did decrease the quality substantially. The other theory potentially may lie within how we organized the testing labels, as upon our first tests it appeared inaccurate only because of organizational issues. 

## Conclusion
The process was a bit painstaking, as there were many learning curves coming from introductory courses, but in the end, our understanind of Neural Networks had improved. The accuracy rate was much higher than expected, and we also did have theories on how to improve them moving forward.
