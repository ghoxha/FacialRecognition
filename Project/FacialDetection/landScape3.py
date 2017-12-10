import numpy as np
import cv2
import dlib


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


def get_landmarks(file_path):
    image_path = file_path
    cascade_path = "haarcascade_frontalface_default.xml"
    predictor_path = "D:\CIS465-ComputerVision\Project\\shape_predictor_68_face_landmarks.dat"

    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier(cascade_path)

    # create the landmark predictor
    predictor = dlib.shape_predictor(predictor_path)

    # Read the image
    image = cv2.imread(image_path)
    # print("image ", type(image))

    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=5,
        minSize=(100, 100),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    print("Found {0} faces!".format(len(faces)))

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Converting the OpenCV rectangle coordinates to Dlib rectangle
        dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        #print(dlib_rect)

        detected_landmarks = predictor(image, dlib_rect).parts()

        # use the face detect region to detect the face landmarks,
        # and extract out the coordinates of the landmarks so OpenCV can use them
        landmarks = np.matrix([[p.x, p.y] for p in detected_landmarks])

    #     #  CODE BELOW THIS STATEMENT IS USELESS IN THE PROGRAM.
    #     #  LEFT IN FOR INSTRUCTIONAL PURPOSES.
    #     # copying the image so we can see side-by-side
    #     image_copy = image.copy()
    #
    #     for idx, point in enumerate(landmarks):
    #         pos = (point[0, 0], point[0, 1])
    #
    #         # print("pos: ", type(pos), " ", pos)
    #
    #         # annotate the positions
    #         # cv2.putText(image_copy, str(idx), pos,
    #         # fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    #         # fontScale=0.4,
    #         # color=(0, 0, 255))
    #
    #         # draw points on the landmark positions
    #         cv2.circle(image_copy, pos, 3, color=(0, 255, 255))
    #
    # cv2.imshow("Faces found", image)
    # cv2.imshow("Landmarks found", image_copy)
    return landmarks


def main():
    #image_capture()
    landmark1 = get_landmarks("bhuard.jpg")
    landmark2 = get_landmarks("bhuard667.jpg")
    print("landmark1 ", type(landmark1))
    print("landmark2 ", type(landmark2))

    #this doesnt work, landscapes matrices are always going to be different!!!!!!
    # no idea on how to compare these two matrices
    if np.array_equal(landmark1, landmark2):
        print('these motherfuckers match through a simple equals condition')

    cv2.waitKey(0)


if __name__ == "__main__":
    main()
