# This file is where we will train the program to identify 48 2018 SPIS students, 12 mentors and 3 instructors.
# We will need to create a file and save it, so that we will be able to use it in "face-recognition.py".
import os
from PIL import Image
import cv2
import numpy as np
import pickle

#Main/Root directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__)) #__file__ current file
data_dir = os.path.join(PROJECT_ROOT, "images")
#print(data_dir)

#Face detector for training images
faces_cascade = cv2.CascadeClassifier('cascades\data\haarcascade_frontalface_default.xml')

#Data Station setup
userNumber = 0  #Assigning numbers instead of names
userInfo =  {}
labels = []
train_photos = []

#Looping through our src\images folder
for dirName, subdirList, fileList in os.walk(data_dir):
    #print(fileList)
    #print(dirName)
    #print(subdirList)
    for file in fileList:
        #print(file)
        if file.endswith(".png") or file.endswith(".jpg"):
            path = os.path.join(dirName,file)
            names = os.path.basename(dirName)   #Getting the names of the users
            #print(path)
            #print(names)
            if names not in userInfo:
                userInfo[names] = userNumber   
                userNumber += 1
            myId = userInfo[names] #Recognizing people by their user number
            #print(myId)

            filt_photos = Image.open(path).convert('L') #Converts the data images to grayscale
            #filt_photos.show()
            npArray = np.array(filt_photos, "uint8") #Taking every pixel value and turning it into a numpy array
            #print(npArray) #Now we can see the pictures in numbers
            #Apply cascade, now that the images are in grayscale
            faceId = faces_cascade.detectMultiScale(npArray) #, scaleFactor=1.3, minNeighbors = 3
            #print(faceId)
            
            for (x,y,w,h) in faceId:
                roi_grayscale = npArray[y:y+h, x:x+w]
                #print(roi_grayscale)
                train_photos.append(roi_grayscale)
                labels.append(myId)

#print(userInfo)
#print(labels)
#print(train_photos)


#Using pickle to convert a python object into string.
pickle_out = open("training/ids.pickle","wb")
pickle.dump(userInfo,pickle_out)
pickle_out.close()

#Train
#Creating Recognizer
recognition = cv2.face.LBPHFaceRecognizer_create()
'''We are working with LBPH, because we won't need to worry about the brightness. It works with the idividual pixels '''
#print(len(train_photos), len(np.array(labels)))
recognition.train(train_photos, np.array(labels))
recognition.save("training/trained.yml")