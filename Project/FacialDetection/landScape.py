from __future__ import division
import dlib
import cv2
import numpy as np


def resize(img, width=None, height=None, interpolation=cv2.INTER_AREA):
    global ratio
    w, h = img.shape

    if width is None and height is None:
        return img
    elif width is None:
        ratio = height / h
        width = int(w * ratio)
        resized = cv2.resize(img, (height, width), interpolation)
        return resized
    else:
        ratio = width / w
        height = int(h * ratio)
        resized = cv2.resize(img, (height, width), interpolation)
        return resized


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


def landscape_fet_grab():
    # purpose: function purpose is to extract facial features into a vector
    # inputs: none
    # outputs: none
    # 1. function may need to populate lists with file names using os.walk()
    # 2. iterate through lists extracting facial landscape profile and storing it in vector
    # 3. then call a comparison function, "landscape_fet_comp()", to look for matches
    dum = []


def landscape_fet_comp():
    # purpose: functions purpose is to compare two vectors and return a boolean
    # inputs: vectors from template and pic capture
    # output: boolean
    dum = []


def match_display():
    # purpose: process match and display results
    # inputs: string "file_name"
    # outputs: image with identity
    dum = []


def image_capture():
    video_capture = cv2.VideoCapture(0)
    flag = 0

    # loads in haarcascades
    cascPath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)

    while True:
        ret, frame = video_capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30))

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            print("drew rectangle")

            face_cap = frame[y:y + h, x:x + w]
            FaceFileName = "facePics\\fp_capture_" + str(y) + ".jpg"
            cv2.imwrite(FaceFileName, face_cap)
            print("wrote file")
            cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()


def main():

    image_capture()

    camera = cv2.VideoCapture(0)
    predictor_path = 'D:\CIS465-ComputerVision\Project\\shape_predictor_68_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    while True:

        ret, frame = camera.read()
        if ret == False:
            print('Failed to capture frame from camera. Check camera index in cv2.VideoCapture(0) \n')
            break

        frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_resized = resize(frame_grey, width=120)

        # Ask the detector to find the bounding boxes of each face. The 1 in the
        # second argument indicates that we should upsample the image 1 time. This
        # will make everything bigger and allow us to detect more faces.
        dets = detector(frame_resized, 1)
        if len(dets) > 0:
            for k, d in enumerate(dets):
                # determine the facial landmarks for the face region, then
                # convert the facial landmark (x, y)-coordinates to a NumPy array
                shape = predictor(frame_resized, d)
                shape = shape_to_np(shape)

                # loop over the (x, y)-coordinates for the facial landmarks
                # and draw them on the image
                for (x, y) in shape:
                    cv2.circle(frame, (int(x / ratio), int(y / ratio)), 3, (255, 255, 255), -1)
                    # cv2.rectangle(frame, (int(d.left()/ratio), int(d.top()/ratio)),(int(d.right()/ratio), int(d.bottom()/ratio)), (0, 255, 0), 1)

        cv2.imshow("image", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            camera.release()
            break


if __name__ == "__main__":
    main()
