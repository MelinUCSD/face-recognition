# This file is where we will train the program to identify 48 2018 SPIS students, 12 mentors and 3 instructors.
# We will need to create a file and save it, so that we will be able to use it in "face-recognition.py".

import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(PROJECT_ROOT, "images")
#print(data_dir)


images = []

for dirName, subdirList, fileList in os.walk(data_dir):
    #print(fileList)
    #print(dirName)
    #print(subdirList)
    for file in fileList:
        #print(file)
        if file.endswith(".png") or file.endswith(".jpg"):
            if  
            images.append(file)
print(images)