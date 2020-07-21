from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle

print("Loading Face embeddings...")
data = pickle.loads(open("output/embeddings.pickle","rb").read())

print("Encode labels...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])

#Train the model used to accept the 128-d embeddings of the face and then produce the actual face recognition
print("Start training the model...")
recoginizer = SVC(C=1.0,kernel="linear",probability=True)
recoginizer.fit(data["embeddings"],labels)

#Write the actual face recognition model
f = open("output/recognizer.pickle","wb")
f.write(pickle.dumps(recoginizer))
f.close()
#Write the label encoder
f = open("output/le.pickle","wb")
f.write(pickle.dumps(le))
f.close()
