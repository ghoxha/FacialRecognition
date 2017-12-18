import numpy as np
import cv2
import dlib
import os
from skimage import feature
from skimage.util.shape import view_as_windows
import scipy.io as sio
from sklearn.decomposition import PCA
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from time import time
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier


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
            # print("drew rectangle")

            # grabs the region of interest and imposes that on the color frame and writes that frame to pic lib
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


class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        # store the number of points and radius
        self.numPoints = numPoints
        self.radius = radius

    def describe(self, image, eps=1e-7):
        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns
        lbp = feature.local_binary_pattern(image, self.numPoints,
                                           self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, self.numPoints + 3),
                                 range=(0, self.numPoints + 2))
        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)

        # return the histogram of Local Binary Patterns
        return hist


def main():

    H = []
    H2 = []
    LBPListDB = []
    LBPListDS = []
    count = 0

    #directory paths for the identities "templates" and the image_library "webcam frame captures"
    template_path = 'D:\\CIS465-ComputerVision\\Project\\FacialDetection\\templates'
    image_lib_path = 'D:\\CIS465-ComputerVision\\Project\\FacialDetection\\ls_pic_repos'

    #captures frames from webcam
    #image_capture()

    templates_list = load_images(template_path)
    image_lib_list = load_images(image_lib_path)

    #get LBP for the database
    for x in templates_list:

        # read the image
        img = cv2.imread(x, 0)

        # create the LBP feature instance
        desc = LocalBinaryPatterns(8, 1)

        # resize the image to 100*100
        face = cv2.resize(img, (100, 100))

        # create the local patches of an image
        patches = view_as_windows(face, (5, 5), step=4)

        for pp in patches:
            for p in pp:
                hist = desc.describe(p)
                H = np.concatenate((H, hist), axis=0)

        print('H: ', type(H))
        #LBPListDB[count, :] = H
        #count += 1
        LBPListDB.append(H)


    # H will be LBP feature for the face
    print("FINISHED THE LBP EXTRACTION for database")
    np.save('LBP_DBase_File.npy', LBPListDB)
    print("Saved LBPList as a .npy file type")

    #reset count
    #count = 0

    #get LBP for the dataset
    for g in image_lib_list:

        # read the image
        img = cv2.imread(g, 0)

        # create the LBP feature instance
        desc = LocalBinaryPatterns(8, 1)

        # resize the image to 100*100
        face = cv2.resize(img, (100, 100))

        # create the local patches of an image
        patches = view_as_windows(face, (5, 5), step=5)

        for pp in patches:
            for p in pp:
                hist = desc.describe(p)
                H2 = np.concatenate((H2, hist), axis=0)

        print('H: ', type(H2))
        #LBPListDS[count, :] = H
        #count += 1
        LBPListDS.append(H2)

    # H will be LBP feature for the face
    print("FINISHED THE LBP EXTRACTION for dataset")
    np.save('LBP_DSet_File.npy', LBPListDS)
    print("Saved LBPList as a .npy file type")


    # load our LBP features
    # n-by-d matrix, n-number of samples, d-dimensionality

    # load ground truth labels
    #FaceMat = sio.loadmat('LBP_DBase_File.npy')
    FaceMat = np.load('LBP_DBase_File.npy')
    print("LOADED DATABASE FILE")
    # n by 1 label vector
    gnd = FaceMat[..., np.newaxis]

    fea = np.load('LBP_DSet_File.npy')
    print('LOADED LBP dataset FILE')
    # number of dimension after PCA
    new_dim = 50
    # number of different identities
    n_classes = 40
    print('SET NEW DIMENSIONS')

    # We first run PCA to reduce the dimensionality
    print('ENTERING THE PCA')
    pca = PCA(n_components=new_dim)
    print('CREATED pca OBJECT!!')
    pca.fit(fea)
    print('FEA SENT TO THE pca.FIT')
    feaPCA = pca.transform(fea)
    print('feaPCA OBJECT CREATED AND TRANSFORMED pcaFEA')

    # use SVM for classification
    print("Fitting the classifier to the training set")
    t0 = time()
    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                  'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
    clf = clf.fit(feaPCA, gnd.ravel())
    print("done in %0.3fs" % (time() - t0))
    print("Best estimator found by grid search:")
    print(clf.best_estimator_)

    print("Predicting on the test set")
    t0 = time()
    # predicted labels
    y_pred = clf.predict(feaPCA)
    # classification accuracy
    acc = clf.score(feaPCA, gnd)
    print("SVM classification accuracy is %0.3f" % acc)
    print("done in %0.3fs" % (time() - t0))

    cv2.waitKey(0)
    exit(0)


if __name__ == "__main__":
    main()
