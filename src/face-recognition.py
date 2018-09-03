##########################################################################################################
##                                                                                                      ##
##            Welcome to (*INSERT GOOD NAME HERE*)! Created by Dare Hunt and Matias Lin                 ##
##                                                                                                      ##
##########################################################################################################
##                                                                                                      ##
##                          The purpose of this program is to be able to identify                       ##
##                      every single 48 students, 12 mentors and 3 instructors of SPIS 2018.            ##
##                                                                                                      ##
##########################################################################################################
##                                                                                                      ##
##              We will be using a "yml" file which contains all the training needed to run this        ##
##                                                 program                                              ##
##                                                                                                      ##
##########################################################################################################

import cv2
import numpy as np
import pickle
from PIL import Image
import os
import shutil
import os.path
from os import path

webCam = cv2.VideoCapture(0)
faces_cascade = cv2.CascadeClassifier('cascades\data\haarcascade_frontalface_default.xml')

#Creating recognizer and reading data
recognition = cv2.face.LBPHFaceRecognizer_create()
recognition.read("training/trained.yml")

#Inverting the key-value pair in user
def dictinvers(user):
    userInfo = {}
    for k, v in user.items():
        keys = userInfo.setdefault(v, [])
        keys.append(k)
    return userInfo

#Getting the labels
user = {}
pickle_in = open("training/ids.pickle", "rb")
user = pickle.load(pickle_in)
userInfo = dictinvers(user)
#print(dictinvers(user))

while(True):
    #Activate webCam
    ret, scope = webCam.read()

    if ret != True:
        break

    #Converting the frames from BGR to Gray Scale
    gray_scale = cv2.cvtColor(scope, cv2.COLOR_BGR2GRAY)
    
    #Face Identifier
    faceId = faces_cascade.detectMultiScale(gray_scale, scaleFactor=1.5, minNeighbors = 6)

    #Square and Label around the face
    for (x, y, w, h) in faceId:
        #print(x,y,h,w) #Checking if haarcascade is working

        save_image = "last_user.jpg"
        cv2.imwrite(save_image,scope)

        #Set up for Rectangle
        color = (255, 255, 0) #BGR
        thickness = 2
        endx = x + w
        endy = y + h
        cv2.rectangle(scope, (x,y), (endx, endy), color, thickness)

        #roi = region of interest
        roi_gray = gray_scale[y:y+h, x:x+w]
        #roi_scope = scope[y:y+h, x:x+w]

        #Setting up the recognizer 
        myId, conf = recognition.predict(roi_gray)
        if conf >=4 and conf <= 85:
            #print(myId)
            myName = str(userInfo[myId])
            print(myName)
            cv2.putText(scope, myName, (x,y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255,255,255), 3)
            display = "last_user_display.jpg"
            cv2.imwrite(display,scope)

    cv2.imshow('scope', scope)
    #Escape Key
    if cv2.waitKey(20) & 0xFF == ord('`'):
        last_user = Image.open("last_user_display.jpg")
        last_user.show()
        print("\n\n\n\n#########################################################################################################\n\n")
        print("Thank you for using ... ! Please take a moment to complete the following survey.\n\n")
        right_wrong = input("Was the prediction correct? ").lower()
        if right_wrong == ("yes") or right_wrong == ('y'):
            print("\n\n\n\n\n\nGreat! Please come back!\n\n")
            print("\n\n\n\n#########################################################################################################\n\n")
            break
        elif right_wrong == ('no') or right_wrong  == ("n"):
            print("I'm sorry to hear that. It will be of great help if you answer the next question!")
            name = input("What's your name? ")
            destination = ("images/%s" %name)
            #newImage = os.rename(save_image, name + ".jpg") #FIND HOW TO RENAME BUT MAKE IT A JPG
            #print(path.isdir(destination))
            if path.isdir(destination) == True:
                shutil.move(save_image,destination)
            else:
                os.mkdir("images/%s"%name)
                shutil.move(save_image, destination)
            print("\n\n\n\nThank you very much!\n\n")
            print("\n\n\n\n#########################################################################################################\n\n")
            break
        else:
            print("No comprendo")
            break

#This will free the Camera for future2 uses
webCam.release()
cv2.destroyAllWindows()