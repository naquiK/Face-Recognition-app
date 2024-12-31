import cv2
import numpy as np
import os

# Data
dataSet_path = "./dataset/"
faceData = []
labels = []
nameMap = {}
class_id = 0

offset = 20
skip = 0

face_data = []

# Load the dataset
for f in os.listdir(dataSet_path):
    if f.endswith(".npy"):
        nameMap[class_id] = f[:-4]
        dataitem = np.load(dataSet_path + f)  # Load data only once
        faceData.append(dataitem)

        # Create labels
        target = class_id * np.ones((dataitem.shape[0],))
        class_id += 1
        labels.append(target)

X = np.concatenate(faceData, axis=0)
Yt = np.concatenate(labels, axis=0).reshape((-1, 1))

# Algorithm
def dist(p, q):
    return np.sqrt(np.sum((p - q) ** 2))

def knn(X, y, xt, k=7):
    m = X.shape[0]
    dlist = []

    # Calculate distance from the input image to each training sample
    for i in range(m):
        d = dist(X[i], xt)
        dlist.append((d, y[i]))  # Append distance and label as a tuple
    
    # Sort the list by distance (the first element of each tuple)
    dlist = sorted(dlist, key=lambda x: x[0])

    # Take the first k elements (smallest distances)
    dlist = dlist[:k]
    
    # Extract labels from the sorted list
    labels = [label for _, label in dlist]

    # Find the most frequent label
    labels, cnts = np.unique(labels, return_counts=True)
    idx = cnts.argmax()
    pred = labels[idx]
    return int(pred)


# Prediction
cam = cv2.VideoCapture(0)

model = cv2.CascadeClassifier('./haarcascade_frontalface_alt.xml')

while True:
    success, img = cam.read()
    
    if not success:
        print("Failed to read image from camera")
        break
    
    faces = model.detectMultiScale(img, 1.2, 2)
    

    # Pick the largest face
    for f in faces:
        x, y, w, h = f
        
        # Crop and save the largest face from the image
        cropped_face = img[y - offset : y + h + offset, x - offset : x + w + offset]
        cropped_face = cv2.resize(cropped_face, (100, 100))  # Resize to match dataset size
        
        # Flatten the cropped face for prediction
        classPred = knn(X, Yt, cropped_face.flatten() )
        
        # Get the name of the predicted person
        namePred = nameMap[classPred]
        
        # Display the name on the image
        cv2.putText(img, namePred, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        
    
    # Show the image
    cv2.imshow("Prediction window", img)
    
    # Exit on pressing 'c'
    key = cv2.waitKey(50)
    if key == ord('c'):
        break

# Release the camera and close the window
cam.release()
cv2.destroyAllWindows()
