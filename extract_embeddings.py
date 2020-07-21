from imutils import paths
import numpy as np
import cv2
import pickle
import os
import imutils

print("Loading Face Detector...")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

protoPath = os.path.join(BASE_DIR,"face_detection_model/deploy.prototxt")
modelPath = os.path.join(BASE_DIR,"face_detection_model/res10_300x300_ssd_iter_140000.caffemodel")
embedderPath = os.path.join(BASE_DIR,"openface_nn4.small2.v1.t7")
datasetPath = os.path.join(BASE_DIR,"dataset")
detector = cv2.dnn.readNetFromCaffe(protoPath,modelPath)
confi = 0.5
print("Loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(embedderPath)

print("Quantifying faces...")
imagePaths = list(paths.list_images(datasetPath))

knownEmbeddings = []
knownNames = []

total = 0


#Loop for the image path
for (i,imagePath) in enumerate(imagePaths):
    #Label name form path name
    print("procecing image {}/{}".format(i+1,len(imagePaths)))
    name = imagePath.split(os.path.sep)[-2]

    #Image resizing
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]

    #Construct blob for Face Detection from the image
    imageBlob = cv2.dnn.blobFromImage(cv2.resize(image,(300,300)),1.0,(300,300),(104.0,177.0,123.0),swapRB=False,crop=False)

    #Applying Deep Learning model 128dnn
    detector.setInput(imageBlob)
    detections = detector.forward()

    #To Ensure at Least one face was found
    if len(detections) > 0 :
        #Find bounding box with the largest probability
        i = np.argmax(detections[0,0,:,2])
        confidence = detections[0,0,i,2]

        #Filter out weak detections
        if confidence > confi:
            #make a box
            box = detections[0,0,i,3:7]*np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            #Extrac face ROI
            face = image[startY:endY,startX:endX]
            (fH, fW) = face.shape[:2]

            #make sure face is in good size
            if fW < 20 or fH < 20:
                continue
            #construct a blob for the face ROI and pass in embedding model to obtain 128-d
            faceBlob = cv2.dnn.blobFromImage(face, 1.0/255,(96,96), (0,0,0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            #Labeling Detected face and embedding to their respective lists
            knownNames.append(name)
            knownEmbeddings.append(vec.flatten())
            total +=1

#Store facial embeddings and name to folder
print("Serializing {} encoddings...".format(total))
data = {"embeddings":knownEmbeddings,"names":knownNames}
f = open('output/embeddings.pickle',"wb")
f.write(pickle.dumps(data))
f.close()
