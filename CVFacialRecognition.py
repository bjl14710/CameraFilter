import numpy as np
import cv2, time


## get the camera running
cap = cv2.VideoCapture(0)
#overlay image
#foreground = np.ones((100,100,3),dtype = 'uint8')*255 # making the connection

image = cv2.imread('test_images/heart.jpg')
#get image dimensions
img_height, img_width, _ = image.shape


# for this cascade to work, needs to be in the same file as the code. So fixed by copy and pasting this in practice code folder
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

eyes_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

## facial recognition part of the code
while(True):
    ret, frame = cap.read() # ret is true or false, frame is the matrix.



    # convert to gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #cv2.imshow('frame',gray) # to test the grayscale


    faces = face_cascade.detectMultiScale(gray,1.1,4)

    eyes = eyes_cascade.detectMultiScale(gray,1.1,4)
    


    for(x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w, y+h),(255,0,0),2)

    for(x,y,w,h) in eyes:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        # might try to manipulate eyes here
        img = cv2.resize(image, (w,h))
        hit,wid, _ = img.shape
        if (wid == w) & (hit == h):
            frame[y:y+w , x:x+h] = img
    
    if cv2.waitKey(20) & 0xFF == ord('q'): #sets q as the button to stop recording
        break
    
    cv2.imshow('frame',frame) # shows the frame
    





cap.release()
cv2.destroyAllWindows()

## finding the face



