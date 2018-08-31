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

webCam = cv2.VideoCapture(0)

while(True):
    #Activate webCam
    ret, scope = webCam.read()

    if ret != True:
        break

    #Converting the frames from BGR to Gray Scale
    gray_scale = cv2.cvtColor(scope, cv2.COLOR_BGR2GRAY)
    
    #Face Identifier
    faces_cascade = cv2.CascadeClassifier('cascades\data\haarcascade_frontalface_default.xml')
    faceId = faces_cascade.detectMultiScale(gray_scale, scaleFactor=1.5, minNeighbors = 5)

    #Square and Label around the face
    for (x, y, h, w) in faceId:
        #print(x,y,h,w) #Checking if haarcascade is working

        #Set up for Rectangle
        color = (255, 255, 0) #BGR
        stroke = 2
        endx = x + w
        endy = y + h
        cv2.rectangle(scope, (x,y), (endx, endy), color, stroke)

        #roi = region of interest
        soi_gray = gray_scale[y:y+h, x:x+w]
        soi_scope = scope[y:y+h, x:x+w]

    cv2.imshow('scope', scope)
    #Escape Key
    if cv2.waitKey(1) & 0xFF == ord('`'):
      break

#This will free the Camera for future uses
webCam.release()
cv2.destroyAllWindows()