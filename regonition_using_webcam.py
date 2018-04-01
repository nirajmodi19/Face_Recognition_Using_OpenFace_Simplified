#importing the libraries
import cv2
import os

#import sys
import pickle
import openface

from operator import itemgetter

import numpy as np
import pandas as pd

modelDir = './models'
dlibFacePredictor = './models/shape_predictor_68_face_landmarks.dat'#os.path.join(modelDir, '/shape_predictor_68_face_landmarks.dat')

align = openface.AlignDlib(dlibFacePredictor)

def getRep(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    bb = align.getAllFaceBoundingBoxes(rgb)
    
    alignedFaces = []
    for box in bb:
        alignedFaces.append(align.align(96, 
                                        rgb,
                                        box,
                                        landmarkIndices = openface.AlignDlib.OUTER_EYES_AND_NOSE
                                        ))
    reps = []
    for alignedFace in alignedFaces :
        reps.append(net.forward(alignedFace))
    return reps, bb
        

def recog(frame) :
    with open('./generated-embeddings/classifier.pkl', 'rb') as f :
        (le, clf) = pickle.load(f)
    reps, bbs = getRep(frame)
    persons = []
    confidences = []
    for rep in reps :
        rep = rep.reshape(1, -1)
        predictions = clf.predict_proba(rep).ravel()
        maxI = np.argmax(predictions)
        persons.append(le.inverse_transform(maxI))
        confidences.append(predictions[maxI])
        
    return (persons, confidences, bbs)
    

    
networkModel = './models/nn4.small2.v1.t7'
net = openface.TorchNeuralNet(networkModel, imgDim = 96)

cam = cv2.VideoCapture(0)
cam.set(3, 320)
cam.set(4, 240)

confidenceList = []
while True :
    ret, frame = cam.read()
    persons, confidences, bbs = recog(frame)
    
    for i, c in enumerate(confidences):
        if c<= 0.5:
            persons[i] = 'unknown'
    for ids, person in enumerate(persons) :
        cv2.rectangle(frame, (bbs[ids].left(), bbs[ids].top()), (bbs[ids].right(), bbs[ids].bottom()), (0, 255, 0), 2 )
        cv2.putText(frame, "{} -> {:.2f}". format(person, confidences[ids]), (bbs[ids].left(), bbs[ids].bottom()+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imshow('Camera', frame)
    if cv2.waitKey(1) & 0xFF == ord('q') :
        break

cam.release()
cv2.destroyAllWindows()
        
        
        
        
        
        
        
        
        
        
        