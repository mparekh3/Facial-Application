from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import cv2
import imutils
import pickle
import os
import timeq


confi = 0.5
print("LOAD FACE DETECTOR...")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

protoPath = os.path.join(BASE_DIR,"face_detection_model/deploy.prototxt")
modelPath = os.path.join(BASE_DIR,"face_detection_model/res10_300x300_ssd_iter_140000.caffemodel")
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

print("LOAD FACE RECOGNIZER...")
embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")


#Recognizer with encoder
recognizer = pickle.loads(open("output/recognizer.pickle","rb").read())
le = pickle.loads(open("output/le.pickle","rb").read())

print("START FACE IDENTIFICATION...")

vs = VideoStream(src=0).start()

#frame per second counter
fps = FPS().start()

while True:
    frame = vs.read()
    frame = imutils.resize(frame,width=600)
    (h, w) = frame.shape[:2]

    #construct a blob from the image
    imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300),(104.0, 177.0, 123.0), swapRB=False, crop=False)

    #DeepLearning to detect face
    detector.setInput(imageBlob)
    detections = detector.forward()

    for i in range(0,detections.shape[2]):
        confidence = detections[0,0,i,2]
        if confidence > confi:
            box = detections[0,0,i,3:7]*np.array([w,h,w,h])
            (startX, startY, endX, endY) = box.astype("int")

            #extract face ROI
            face = frame[startY:endY,startX:endX]
            (fH, fW) = face.shape[:2]

            if fW < 20 or fH <20:
                continue

            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,(96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]

            text = "{}: {:.2f}%".format(name,proba*100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame,(startX,startY),(endX,endY),(0,0,200),2)
            cv2.putText(frame,text,(startX,y),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,0,255),2)
    fps.update()
    cv2.imshow("Face_Identifier",frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
       break

fps.stop()
print("Elasped time: {:.2f}".format(fps.elapsed()))
print("Aprrox. FPS {:.2}".format(fps.fps()))

cv2.destroyAllWindows()
vs.stop()
