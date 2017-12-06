from __future__ import division
import cv2
import numpy as np
import os

#loads in haarcascades
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

template = cv2.imread('bhuard.jpg', 0)
w, h = template.shape[::-1]


#loads in webcam, zero brings in camera 1st is in pos o
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        #flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
       # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        #region of image where the face is [location of face] roi is y then x
        #roi_gray = gray[y:y+h, x:x+w]
        #region of image color, imposes rectanges back on color cam stream
        #roi_color = frame[y:y+h, x:x+w]

        #capturing images from webcam and storing them
        face_cap = frame[y:y + h, x:x + w]
        FaceFileName = "facePics\\fp_capture_" + str(y) + ".jpg"
        cv2.imwrite(FaceFileName, face_cap)

    # outer loop graps all files in specified directory
    for root, dirs, files in os.walk('D:\\CIS465-ComputerVision\\Project\\FacialDetection\\facePics'):
        for x in files:
           # builds path to read image in for template matching, convert to gray, matchTemplate
            path = "D:\\CIS465-ComputerVision\\Project\\FacialDetection\\facePics\\" + x
            fp_img = cv2.imread(path)
            fp_img_gray = cv2.cvtColor(fp_img, cv2.COLOR_BGR2GRAY)
            res = cv2.matchTemplate(fp_img_gray, template, cv2.TM_CCOEFF_NORMED)
            threshold = 0.8
            loc = np.where(res >= threshold)

            #
            for pt in zip(*loc[::-1]):

                # draw rectangle on image
                cv2.rectangle(fp_img, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 2)
                cv2.imshow('MATCH FOUND', fp_img)
                cv2.waitKey(0)


    # Display the resulting frame
    #cv2.imshow('Video', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
