#importing the libraries
import cv2
import os
#import sys
import pickle

from operator import itemgetter

import numpy as np
import pandas as pd

#import openface

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

modelDir = './models/'

#dlibFacePredictor = os.path.join(modelDir, './shape_predictor_68_face_landmarks.dat')
#networkModel = 'nn4.small2.v1.t7'

#align = openface.AlignDlib(dlibFacePredictor)
#net = openface.TorchNeuralNet(networkModel, imgDim = 96)

def train() :
    print("Loading Embeddings")
    fname = './generated-embeddings/labels.csv'
    labels = pd.read_csv(fname, header=None).as_matrix()[:, 1]
    labels = map(itemgetter(1), map(os.path.split, map(os.path.dirname, labels)))
    labels = list(labels)
    fname = './generated-embeddings/reps.csv'
    embeddings = pd.read_csv(fname, header=None).as_matrix()
    le = LabelEncoder().fit(labels)
    labelsNum = le.transform(labels)
    #print(labelsNum)
    classifier = SVC(C=1, kernel = 'linear', probability=True)
    classifier.fit(embeddings, labelsNum)
    fname = './generated-embeddings/classifier.pkl'
    with open(fname, 'wb') as f:
        print('Saving Classifier to \"{}\" '.format(fname))
        pickle.dump((le, classifier), f)      

train()


    