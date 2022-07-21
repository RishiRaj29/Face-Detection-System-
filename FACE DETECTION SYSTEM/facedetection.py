import cv2

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#The above classifier has already been trained and can be found in the opencv repository on GITHUB

# To capture video from webcam. 
cap = cv2.VideoCapture(0)

# To use a video file as input 
# cap = cv2.VideoCapture('filename.mp4')

while True:
    # Read the frame
    _, img = cap.read() # this function returns 2 arguments 1)A flag which tells whether the video was read properly or not and 2) The frame itself
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # we need to convert our RGB images to gray scale
    # cv2 processes images in BGR format(Blue,Green,Red) instead of the usual RGB format(Red,Blue,Green)
    
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)# this function takes 3 arguments- 1)The Gray Scale Image 2)Scale Factor 3)Minimum Neighbours
    # Minimum Neighbours- it specifies how many neighbors each candidate rectangle should have to retain it.
    
    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)#(255,0,0) specifies the colour of the rectangle and 2 specifies its thickness
        
    # Display
    cv2.imshow('img', img)
    
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
    
# Release the VideoCapture object
cap.release()
