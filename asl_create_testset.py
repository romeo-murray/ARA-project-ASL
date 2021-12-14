import numpy as np
import os
import cv2
import random
import pickle

IMG_SIZE = 50 # size of the images (for compression)
DATADIR = 'C:/your/directory/asl_alphabet_test/' # place the directory path for the asl_test_images here
CATEGORIES = ['A', 'B', 'C', 'D', 'del', 'E', 'F', 'G', 'H', 'I', 
             'J', 'K', 'L', 'M', 'N', 'nothing', 'O', 'P', 'Q', 'R', 
             'S', 'space', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

training_data = [] # the array that will hold our data

def create_train_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category) # the path to the directory of images
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                print(e)

create_train_data() # this calls the method and creates our dataset
print(len(training_data)) # check

# shuffle the data
random.shuffle(training_data)

# create the two outputs for this dataset
X = []
Y = []

# we reshape the dataset
for features, label in training_data:
    X.append(features)
    Y.append(label)
    print(label)
    
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
Y = np.array(Y)

# Now we need to save the data in preparation for the model in a seperate file
pickle_out = open("X_test.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("Y_test.pickle", "wb")
pickle.dump(Y, pickle_out)
pickle_out.close()

# testing loading the dataset to ensure we got it right
# pickle_in = open("X_test.pickle", "rb")
# X = pickle.load(pickle_in)
# print(X[1])
