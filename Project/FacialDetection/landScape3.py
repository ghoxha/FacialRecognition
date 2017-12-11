import numpy as np
import cv2
import dlib
import os


def image_capture():
    video_capture = cv2.VideoCapture(0)

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

        # Draws rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            #print("drew rectangle")

            #grabs the region of interest and imposes that on the color frame and writes that frame to pic lib
            face_cap = frame[y:y + h, x:x + w]
            FaceFileName = "ls_pic_repos\\fp_capture_" + str(y) + ".jpg"
            cv2.imwrite(FaceFileName, face_cap)
            print("wrote file")
            cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()


def load_images(path):
    print('getting templates from ', path)
    img_list = []

    #gets all pictures from specified path and jam them in a list
    for root, dirs, files in os.walk(path):
        # print(root)
        # print(dirs)
        # print(files)

        #builds the image's dir path and jams them in a img_list
        for x in files:
            img_list.append(os.path.join(root, x))

    return img_list


def get_landmarks(processing_list):
    #processing list is list full of the associated image paths

    #landmarks list holds all the individual LMs found in the pList
    landmarks_list = []
    cascade_path = "haarcascade_frontalface_default.xml"
    predictor_path = "D:\CIS465-ComputerVision\Project\\shape_predictor_68_face_landmarks.dat"
    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier(cascade_path)

    # create the landmark predictor
    predictor = dlib.shape_predictor(predictor_path)

    #this loop gets the landmarks for every image in the pList
    for b in processing_list:
        image_path = b

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

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Converting the OpenCV rectangle coordinates to Dlib rectangle
            dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
            detected_landmarks = predictor(image, dlib_rect).parts()

            # use the face detect region to detect the face landmarks,
            # and extract out the coordinates of the landmarks so OpenCV can use them
            landmarks_list.append(np.matrix([[p.x, p.y] for p in detected_landmarks]))

    return landmarks_list


def match(template_landmarks_list, image_landmark_list, templates_list):
    print('Looking for a match')
    ink = 0

    for i in template_landmarks_list:
        #inner loop leverages every image in the pic lib from the webcam against
        #the template lib of identities looking for a match
        for h in image_landmark_list:
            if np.array_equal(i, h):
                print('MATCH FOUND')

                #gets the identity pic from the templates list
                matcher = cv2.imread(templates_list[ink])
                convictName = templates_list[ink]
                #print(convictName)

                #extract filename/identity
                convictName = convictName.split(".")
                convictName = convictName[0]
                convictName = convictName[-7:-1]

                #display the identity with label
                matcher_copy = matcher.copy()
                cv2.putText(matcher_copy, convictName, (10, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(0, 0, 255))
                cv2.imshow("MATCH", matcher_copy)
                break
            else:
                print('NO MATCH!!')
        ink = ink + 1


def main():
    template_path = 'D:\\CIS465-ComputerVision\\Project\\FacialDetection\\templates'
    image_lib_path = 'D:\CIS465-ComputerVision\Project\FacialDetection\ls_pic_repos'

    # image_capture()

    templates_list = load_images(template_path)
    template_landmarks_list = get_landmarks(templates_list)
    image_lib_list = load_images(image_lib_path)
    image_landmark_list = get_landmarks(image_lib_list)

    match(template_landmarks_list, image_landmark_list, templates_list)

    cv2.waitKey(0)
    exit(0)


if __name__ == "__main__":
    main()
