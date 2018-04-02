# import cv2  # Uncomment if you have OpenCV and want to run the real-time demo
import numpy as np
import math
#from sklearn.linear_model import LogisticRegression
def sigmoid(z):
    return 1.0/(1+np.exp(-z))

def whiten(faces):
    mean = faces.mean(axis = 0)
    faces_1 = faces - mean
    cov = faces_1.T.dot(faces_1)
    cov = cov + 0.01 * np.eye(faces.shape[1])
    eigenvalue,eigenvector = np.linalg.eigh(cov)
    diag = np.diag(eigenvalue)
    diag_inv = np.linalg.inv(diag)
    t = np.sqrt(diag_inv)
    whiten_t = t.dot(eigenvector.T).dot(faces.T)
    return whiten_t

def J (w, faces, labels, alpha = 0.):
    i = labels - np.dot(faces,w)
    result = 0.5 * np.dot(np.transpose(i),i) + 0.5 * alpha * w.dot(w)
    #J = 0.5 * (labels - faces.dot(w)).T.dot(labels - faces.dot(w)) + 0.5 * alpha * w.dot(w)
    return result  # TODO implement this!

def Jce (w, faces, labels,alpha = 0.):
    #print faces.shape
    #print w.shape
    #y = np.dot(faces,w)
    y = sigmoid(faces.dot(w))
    m = labels.shape[0]
    #print m
    y[y==1] = 0.999999
    y[y==0] = 0.000001
    cost = -1.0/m * (labels.dot(np.log(y))+(np.ones(m)-labels).dot(np.log(np.ones(m)-y))) + 0.5 * alpha * w.dot(w)
    return cost

def gradJ (w, faces, labels, alpha = 0.):
    i = labels - np.dot(faces,w)
    result = -np.dot(np.transpose(faces),i) + alpha * w
    #print result.shape
    result_1 = result **2
    sum = result_1.sum(axis = 0).sum()
    sum = math.sqrt(sum)
    result = result / sum
    return result  # TODO implement this!

def gradJce (w,faces,labels,alpha = 0.):
    y = sigmoid(faces.dot(w))
    m = labels.shape[0]
    y[y == 1] = 0.999999
    y[y == 0] = 0.000001
    result = -1.0/m * faces.T.dot(labels - y) + alpha * w
    result_1 = result **2
    sum = result_1.sum(axis=0).sum()
    sum =math.sqrt(sum)
    result = result/sum
    return result

def gradientDescent (trainingFaces, trainingLabels, testingFaces, testingLabels, alpha = 0.):
    w = np.random.randn(trainingFaces.shape[1])
    #print w.shape
    w1 = J(w,trainingFaces,trainingLabels,alpha)+ 0.1
    while (abs(w1- J(w,trainingFaces,trainingLabels,alpha)) > 0.001):
        w1 = J(w,trainingFaces,trainingLabels,alpha)
        w = w - 0.01 *gradJ(w,trainingFaces,trainingLabels,alpha)
    # TODO implement this!
    return w

def gradientDescentce(trainingFaces, trainingLabels, testingFaces, testingLabels, alpha = 0.):
    w = np.random.randn(trainingFaces.shape[1])
    cost = Jce(w, trainingFaces, trainingLabels, alpha) + 0.1
    #print cost
    while (abs(cost - Jce(w, trainingFaces, trainingLabels, alpha)) > 0.001):
        cost = Jce(w, trainingFaces, trainingLabels)
        w = w - 0.01 * gradJce(w, trainingFaces, trainingLabels, alpha)
        print(Jce(w, trainingFaces, trainingLabels))
    return w
def method1 (trainingFaces, trainingLabels, testingFaces, testingLabels):
    # w = ...  TODO implement this
    w = np.linalg.solve(np.dot(trainingFaces.T,trainingFaces),np.dot(trainingFaces.T,trainingLabels))
    #print np.trace(trainingFaces)
    #print trainingLabels.T.dot(trainingLabels)
    #print w.T.dot(w)
    return w

def method2 (trainingFaces, trainingLabels, testingFaces, testingLabels):
    return gradientDescent(trainingFaces, trainingLabels, testingFaces, testingLabels)

def method3 (trainingFaces, trainingLabels, testingFaces, testingLabels):
    alpha = 1e3
    return gradientDescent(trainingFaces, trainingLabels, testingFaces, testingLabels, alpha)

def method4(trainingFaces, trainingLabels, testingFaces, testingLabels):
    #w = np.random.randn(trainingFaces.shape[1])
    #w1 = Jce(w,trainingFaces,trainingLabels, alpha) +0.1
    #while (abs(w1-Jce(w,trainingFaces,trainingLabels, alpha)) >0.01):
     #   w1 = Jce(w,trainingFaces,trainingLabels)
      #  w = w1 - 0.01 * gradJce(w,trainingFaces,trainingLabels,alpha)
       # print (Jce(w,trainingFaces,trainingLabels))
    #return w

    return gradientDescentce(trainingFaces, trainingLabels, testingFaces, testingLabels)
def reportCosts (w, trainingFaces, trainingLabels, testingFaces, testingLabels, alpha = 0.):
    print "Training cost: {}".format(J(w, trainingFaces, trainingLabels, alpha))
    print "Testing cost:  {}".format(J(w, testingFaces, testingLabels, alpha))

def reportCostsce (w, trainingFaces, trainingLabels, testingFaces, testingLabels, alpha = 0.):
    print "Training probability: {}".format(Jce(w, trainingFaces, trainingLabels, alpha))
    print "Testing probability:  {}".format(Jce(w, testingFaces, testingLabels, alpha))

# Accesses the web camera, displays a window showing the face, and classifies smiles in real time
# Requires OpenCV.
def detectSmiles (w):
    # Given the image captured from the web camera, classify the smile
    def classifySmile (im, imGray, faceBox, w):
        # Extract face patch as vector
        face = imGray[faceBox[1]:faceBox[1]+faceBox[3], faceBox[0]:faceBox[0]+faceBox[2]]
        face = cv2.resize(face, (24, 24))
        face = (face - np.mean(face)) / np.std(face)  # Normalize
        face = np.reshape(face, face.shape[0]*face.shape[1])

        # Classify face patch
        yhat = w.dot(face)
        print yhat

        # Draw result as colored rectangle
        THICKNESS = 3
        green = 128 + (yhat - 0.5) * 255
        color = (0, green, 255 - green)
        pt1 = (faceBox[0], faceBox[1])
        pt2 = (faceBox[0]+faceBox[2], faceBox[1]+faceBox[3])
        cv2.rectangle(im, pt1, pt2, color, THICKNESS)

    # Starting video capture
    vc = cv2.VideoCapture()
    vc.open(0)
    faceDetector = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")  # TODO update the path
    while vc.grab():
        (tf,im) = vc.read()
        im = cv2.resize(im, (im.shape[1]/2, im.shape[0]/2))  # Divide resolution by 2 for speed
        imGray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        k = cv2.waitKey(30)
        if k >= 0 and chr(k) == 'q':
            print "quitting"
            break

        # Detect faces
        faceBoxes = faceDetector.detectMultiScale(imGray)
        for faceBox in faceBoxes:
            classifySmile(im, imGray, faceBox, w)
        cv2.imshow("WebCam", im)

    cv2.destroyWindow("WebCam")
    vc.release()

if __name__ == "__main__":
    # Load data
    if ('trainingFaces' not in globals()):  # In ipython, use "run -i homework2_template.py" to avoid re-loading of data
        trainingFaces = np.load("trainingFaces.npy")
        trainingLabels = np.load("trainingLabels.npy")
        testingFaces = np.load("testingFaces.npy")
        testingLabels = np.load("testingLabels.npy")

    w1 = method1(trainingFaces, trainingLabels, testingFaces, testingLabels)
    w2 = method2(trainingFaces, trainingLabels, testingFaces, testingLabels)
    w3 = method3(trainingFaces, trainingLabels, testingFaces, testingLabels)
    w4 = method4(trainingFaces, trainingLabels, testingFaces, testingLabels)
    for w in [ w1, w2, w3 ]:
        reportCosts(w, trainingFaces, trainingLabels, testingFaces, testingLabels)
    reportCostsce(w4,trainingFaces,trainingLabels,testingFaces,testingLabels)
    #detectSmiles(w3)  # Requires OpenCV
