# Importing Libraries
import os
import cv2
import random

import openface
import openface.helper
from openface.data import iterImgs

openface.helper.mkdirP('./aligned-images/') #'/aligned-images/ --> Out Directory for storing aligned images

# Model used to predict the 68 face landmarks
dlibFacePredictor = './models/shape_predictor_68_face_landmarks.dat'

#Defining an object of AlignDlib class
align = openface.AlignDlib(dlibFacePredictor)

#Iterating among all the images
imgs = list(iterImgs('./training-images'))

#Shuffling the Images
random.shuffle(imgs)

nFallbacks = 0

landmarkMap = {
        'outerEyesAndNose' : openface.AlignDlib.OUTER_EYES_AND_NOSE,
        'innerEyesAndBottomLip' : openface.AlignDlib.INNER_EYES_AND_BOTTOM_LIP
        }
landmarks = 'outerEyesAndNose' # Default, can also take 'innerEyesAndBottomLip'
landmarkIndices = landmarkMap[landmarks]

#Iterating through images one-by-one
for imgObject in imgs :
    #Creating Output Directory 
    outDir = os.path.join('./aligned-images/',imgObject.cls)
    openface.helper.mkdirP(outDir)
    outputPrefix = os.path.join(outDir, imgObject.name)
    imgName = outputPrefix + '.png'
    
    #Checking if the file exist already or not
    if os.path.isfile(imgName) :
        print("Already Found %s Skipping..."%imgName)
    else:
        rgb = imgObject.getRGB()
        if rgb is None :
            print("Unable to load {}".format(imgName))
            outRgb = None
        else :
            size = 96 # Taken as default size
            
            #aligning the input images
            outRgb = align.align(size, rgb, landmarkIndices = landmarkIndices)
            
            if outRgb is None :
                print("Unable to align {}".format(imgName))
        if outRgb is None :
            nFallbacks += 1
            #print('Skipping...')
        if outRgb is not None :
            #Converting images in BGR format and storing in the disk
            print("Writing {} to disk...".format(imgName))
            outBgr = cv2.cvtColor(outRgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(imgName, outBgr)
            