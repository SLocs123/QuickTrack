from filterpy.kalman import KalmanFilter
import numpy as np

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

def getColourML(self, detection):
# import the git https://github.com/benaloha/car-classifier-yolo3-python
    return convert_color(color, LabColor)


# define all setup functions:

# def displacement():

# def shape():

# def boundary():

# def featureEmbedding():



#    def getDistance(self, detection):
#        refHeight = 194  # this is the height of the model pulled from monodepth2, to ensure the correct standard is used (could also change to percentage of screen but either works)
#        scale = refHeight / self.imgH # scale to shift the measured detection height to the standard -------------------(ensure this is screen hieght)
#        cls = round(detection[5]) # gets the class, so the correst reference equation is used

        # current distance
#        height = detection[3] - detection[1] # measured detection height and then scale
#        height = height * scale

#        equ = referenceValues[cls]  # reference numerator, reference power
#        Dist = round(equ[0] * height ** equ[1], 3) # estimates distance from the above set of equations

#        return Dist

#    def load_classes(self):
        # Loads *.names file at 'path'
#        with open(self.path, 'r') as f:
#            names = f.read().split('\n')
#        return list(filter(None, names))  # filter removes empty strings (such as lastline)

    # def updateTracks(self):

# ...

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
