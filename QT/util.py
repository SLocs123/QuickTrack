from filterpy.kalman import KalmanFilter
import numpy as np
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000, delta_e_cmc, delta_e_cie1994, delta_e_cie1976


def calculateTimeDependents(a, b, t):
    return round((a[0] - b[0]) / t, 3)


def averageShape(matrix):
    zeroTwo = []
    oneThree = []
    for row in matrix:
        zeroTwo.append(row[2] - row[0])
        oneThree.append(row[3] - row[1])
    width = average(zeroTwo)
    height = average(oneThree)
    return width/height


def getMiddle(detection):
    bbox = detection[:4]
    middle = [(bbox[2]-bbox[0])/2+bbox[0], (bbox[3]-bbox[1])/2+bbox[1]]
    return middle


def average(list):
    return sum(list) / len(list)


def sortHighest(list):
    return sorted(list, key=lambda x: x[2], reverse=True)


def getColourSimple(self, detection):
    width = detection[2] - detection[0]
    height = detection[3] - detection[1]
    b, g, r = self.img[int((detection[1] + height / 2)), int((detection[0] + width / 2))]
    color = sRGBColor(r, g, b)
    return convert_color(color, LabColor)


def getColourML(self, detection, img):
    # import the git https://github.com/benaloha/car-classifier-yolo3-python
    return colour


def getModel(self, detection, img):
    # import the git https://github.com/benaloha/car-classifier-yolo3-python
    return colour


def createKF(x, y):
    """Initializes a Kalman Filter for a new track based on initial (x, y) position."""
    kf = KalmanFilter(dim_x=4, dim_z=2)
    
    # State transition matrix
    dt = 1  # time step
    kf.F = np.array([[1, 0, dt, 0], 
                     [0, 1, 0, dt], 
                     [0, 0, 1, 0], 
                     [0, 0, 0, 1]])
    
    # Measurement function
    kf.H = np.array([[1, 0, 0, 0], 
                     [0, 1, 0, 0]])
    
    # Initial state estimate
    kf.x = np.array([x, y, 0, 0]).T
    
    # Covariance matrix
    kf.P *= 1000
    
    # Measurement noise
    kf.R = np.array([[5, 0], 
                     [0, 5]])
    
    # Process noise
    kf.Q = np.array([[dt**4/4, 0, dt**3/2, 0],
                  [0, dt**4/4, 0, dt**3/2],
                  [dt**3/2, 0, dt**2, 0],
                  [0, dt**3/2, 0, dt**2]]) * 0.03
    return kf


def KFTrustworthy(track, variance_thresholds):
    p_diag = np.diag(track.kf.P)  # Extract the diagonal (variances) from the covariance matrix
    for variance, threshold in zip(p_diag, variance_thresholds):
        if variance > threshold:
            return False  # Prediction is not trustworthy if any variance exceeds its threshold
    return True  # All variances are within their thresholds


def load_classes(path):
     # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)

# define all setup functions:

# def displacement():

# def shape():

# def boundary():

# def featureEmbedding():
