from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2
import os

#Construct Arguments parser and parse the arguments

ap = argparse.ArgumentParser()

ap.add_argument("-c","--cascade",required=True,help = "path to where the face cascade resides")
ap.add_argument("-o","--output",required=True,help="path to output directory")

args = vars(ap.parse_args())

#Load Haar Cascade for face detection from folder
detector = cv2.CascadeClassifier(args["cascade"])

#Intialize the video camera streaming, allow camera sensor to warm-up and initialize the total
#number of examples faces written.
def load_effect():
    load=[".","..","..."]
    i=0
    while len(load):
        print(load[i])
	time.sleep(0.5)
        i+=1
	if i == 3:
		break


print("Video Streaming start")
load_effect()

vs = VideoStream(src=0).start() #src=0 is for webcam.
time.sleep(2.0)
total = 0 #Initial number of face stored

#Loop over the frames from the vs
while True:
    frame = vs.read()
    orig = frame.copy()
    frame = imutils.resize(frame,width=400)
    #detect faces in the grayscale frame
    rects = detector.detectMultiScale(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),
                                      scaleFactor=1.1, minNeighbors=5,minSize=(30,30))
    #Loop for face detection
    for (x, y, w, h) in rects:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0), 2)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # Here"c" key use to capture and store faces.
    if key == ord("c"):
        p = os.path.sep.join([args["output"],"{}.png".format(str(total).zfill(5))])
        cv2.imwrite(p,orig)
        total += 1
    #Here "q" key use to quit the loop
    elif key == ord("q"):
        break
#print the total number of face stored and do a bit of cleanup
print("{} face image stored".format(total))
print("cleaning up")
load_effect()
cv2.destroyAllWindows()
vs.stop()

