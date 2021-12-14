# ARA-project-ASL
A program that translates ASL to English text, utilizing TensorFlow's built-in convolutional neural network. It is considerably a remaster of an older model made by @grassknoted and @anaskhan96.

## The Idea
Really, all we are attempting here is to get a CNN working, and use an older dataset from a different ASL project. The reason that there is any need for this is that TensorFlow has updated since the original projects debut, meaning those are no longer operable because of the changes TensorFlow had undergone. So, we are attempting to recreate the model and use a similar format with Google Text to Speech for the prediction process.

### Notes
Firstly, this project will not have a live demo as there wasn't enough time to complete it to that extent. Secondly, because TensorFlow has not updated to the current version of CUDA, there are GPU restrictions and cannot be operated using a GPU. So, this version forces TensorFlow to operate using the CPU, consider this fair warning if attempting to use this on your own device.

Thirdly, this is a school oriented project. This means that *no, this is not professional level coding*, and there are most likely mistakes or possible resource leaks, or issues due to there being a massive lack of time. If there is something that is easy to fix, feel free to correct us, or take the code and improve it yourselves!

# How to Run
TODO: This hasn't been flushed out yet, there will be detailed instructions once the program is ready.

### How to use your Own Testing Images
Create a new image of PNG or JPEG (or whatever images that TensorFlow accepts) and convert it to the size of 200x200 pixels. Then port it to the test images folder in the category of whatever letter your hand is gesturing. Lastly, go into the query_classification_new file.py and change the configuration settings on your IDE. In spyder for instance, you would click on the file, click on run>configuration per file> go down to general settings, and on the command line options you would put in the path to the testing image you created.

## Results
![image](https://user-images.githubusercontent.com/92128432/145925092-bfda24df-41fb-4341-8b20-912f443d27a7.png)

The accuracy rate in the end was able to increase to about 98%, but this was based on TensorFlow testing through evaluation and validation processes. After performing our own tests, we had found a few notable issues that remain unresolved. 

### Issues
We had found that out of all the options available (29 of them) 3 were inaccurate for reasons we don't know. This would mean that we had reached a roughly 90% accuracy rate generally speaking. Those three are M, N, and nothing. We theorized that the issue could stem from the quality of the images, as compressing them down to 50x50 did decrease the quality substantially. The other theory potentially may lie within how we organized the testing labels, as upon our first tests it appeared inaccurate only because of organizational issues. 

## Conclusion
The process was a bit painstaking, as there were many learning curves coming from introductory courses, but in the end, our understanind of Neural Networks had improved. The accuracy rate was much higher than expected, and we also did have theories on how to improve them moving forward.
