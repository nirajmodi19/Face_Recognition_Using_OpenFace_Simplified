# Face_Recognition_Using_OpenFace_Simplified
This is a simplified version of using openface to recognize faces.
***

Libraries Requirements :-

   1. OpenFace
   2. Torch
   3. Open CV3
   4. Dlib
	 
Models Used :-

	 1. shape_predictor_68_face_landmarls.dat (OpenFace model for predicting 68 landmarks)
	 2. nn4.small2.v1.t7 (Dlib model)
	 3. LinearSVM (Classification Model)
	
Addressing Files and Directories :-

	 1. './aligned-images'             ---> Contains images after aligned.
	 2. './batch-represent'            ---> Contains lua script to generate 128 embeddings.
	 3. './generated-embeddings'       ---> Contains 'classifier.pkl' and CSV files of 128-dimensions and labels.
	 4. './models'                     --->	Contains Openface and Dlib models.
	 5. './training-images'            ---> Contains training images.
	 6. 'align-images.py'              ---> Python file used to align the training images.
	 7. 'classifier_train.py'          ---> Python file to train a classifier and dump a 'classifier.pkl' file.
	 8. 'recognition_using_webcam.py'  ---> Python file to recognize images using webcam.
 
Steps :-

   1. Clone or Download the repository.
   2. Download and extract 'shape_predictor_68_face_landmarks.dat' from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 and save it on './models' directory.
   3. Download 'nn4.small2.v1.t7' from  https://storage.cmusatyalab.org/openface-models/nn4.small2.v1.t7 in the './models' directory.
   4. Put your training images in the './training-images' directory.
	 5. First align images using the command  'python align-images.py', this will create sub-directories and will store aligned images on the directory.
	 6. Now generate embeddings using './batch-represent/main.lua -outDir ./generated-embeddings/ -data ./aligned-images/' .
	 		Here, '-outDir' takes the directory in which the embeddings will be stored and '-data' takes the aligned-images which is our training data.
	 7. We will train our classifier using 'python classifier_train.py' this will store the trained classifier in the './generated_embeddings' diretory.
	 8. Lastly, to detect the images using webcam run 'python recognition_using_webcam.py'. 
