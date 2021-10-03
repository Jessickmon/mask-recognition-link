import cv2
import os

# call in our file with the training images
cascPath=os.path.dirname(cv2.__file__)+"/data/haarcascade_frontalface_default.xml"

# use these training images as classifiers to the machine
faceCascade = cv2.CascadeClassifier(cascPath)

# captures the video from our camera
video = cv2.VideoCapture(0)

minArea = 500
while True:
    # read the image from our video constantly
    ret, pic = video.read()

    # turn that image into grayscale for classifying
    gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)

    # built-in algorithm to detect faces
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # set boundaries of where face is
    for (a, b, c, d) in faces:
        cv2.rectangle(pic, (a, b), (a+c, b+d), (0, 255, 0), 2)
        area = c * d

        # if face detected, notify on-screen that there is no mask
        if area > minArea:
            cv2.putText(pic, "No mask!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('Video of Good Looking People', pic)

    # allow user to exit video by inputting 's'
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

video.release()

cv2.destroyAllWindows()
