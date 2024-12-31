import cv2
import numpy as np

#create a camera object
cam = cv2.VideoCapture(0)

#Asking the name of the person
preson_name = input("Enter the name of the person: ")
dataset = "./dataset/"
offset = 20
skip=0
#Model
model = cv2.CascadeClassifier('./haarcascade_frontalface_alt.xml')

# Create a list to save face data
face_data = []

#read image from camera object
while True:
   success, img = cam.read()
   
   if not success:
     print("Failed to read image from camera")
   
   #convert the image to grayscale
   grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   
   #pick the face from the image with largest bonding box
   faces = model.detectMultiScale(img , 1.2 , 2)
   
   #sort the faces based on the area of the face
   faces = sorted(faces, key = lambda f: f[2]*f[3])
   
   #Pick the largest face
   if len(faces)>0:
     f = faces[-1]
     x, y, w, h = f
     cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
   
     #crop and save the largest face from the image
     cropped_face = img[y -offset : y+h + offset, x - offset : x+w + offset]
     
     cropped_face = cv2.resize(cropped_face, (100, 100))
     
     skip+=1
     if skip%10==0:
         face_data.append(cropped_face)
         print("Saved so far " + str(len(face_data)))
     
   
   
    
   cv2.imshow("Image window", img)
   key = cv2.waitKey(50)
   if key == ord('c'):
     break
 
 #Write the face data to the disk
 
 
face_data = np.asarray(face_data)
m =face_data.shape[0]
face_data = face_data.reshape((m, -1))

print(face_data.shape)
#Save the face data as np array

file = dataset + preson_name + ".npy"
np.save(file, face_data)
print("Data successfully saved at " + file)
 
 #Realse the camera object
 
cam.release()
cv2.destroyAllWindows()