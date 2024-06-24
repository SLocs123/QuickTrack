from filterpy.kalman import KalmanFilter
import numpy as np
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000, delta_e_cmc, delta_e_cie1994, delta_e_cie1976

def x_to_bbox(x): # conversions originally from SORT code!
    """
    x in form [x,y,s,r]
    Takes a bounding box in the center form [x, y, s, r] and returns it in the form
    [x1, y1, x2, y2] where x1, y1 is the top left and x2, y2 is the bottom right.
    """
    width = np.sqrt(x[2] * x[3])
    height = x[2] / width
    x1= x[0] - width / 2.0
    y1 = x[1] - height / 2.0
    x2 = x[0] + width / 2.0
    y2 = x[1] + height / 2.0
    return np.array([x1, y1, x2, y2]).reshape((1, 4))


def bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1, y1, x2, y2] and returns z in the form
    [x, y, s, r] where x, y is the center of the box, s is the scale/area, and r is
    the aspect ratio.
    """
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    x = bbox[0] + width / 2.0
    y = bbox[1] + height / 2.0
    scale = width * height  # scale is just area
    aspectRatio = width / float(height)
    return np.array([x, y, scale, aspectRatio]).reshape((4, 1))



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


def point_in_polygon(point, polygon):
    """
    Function to check if a point is inside a polygon using the ray-casting algorithm.
    
    Args:
    - point: A tuple or list representing the (x, y) coordinates of the point to check.
    - polygon: A list of tuples or lists, each representing the (x, y) coordinates of a vertex of the polygon.
    
    Returns:
    - True if the point is inside the polygon, False otherwise.
    """
    x, y = point
    n = len(polygon)
    inside = False

    # Ray-casting algorithm
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]

        # Check if point is on the edge of the polygon
        if (y1 == y2 and y == y1 and min(x1, x2) <= x <= max(x1, x2)):
            return True

        # Check if point is above or below the edge
        if min(y1, y2) < y <= max(y1, y2):
            if x1 == x2 or x <= min(x1, x2):
                inside = not inside
            elif x >= max(x1, x2):
                continue
            else:
                # Calculate intersection of ray with edge
                intersection = (y - y1) * (x2 - x1) / (y2 - y1) + x1
                if x <= intersection:
                    inside = not inside

    return inside


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


def createKF(x, y, s, r):
    """Initializes a Kalman Filter for a new track based on initial (x, y) position, scale, and aspect ratio. This is alligned with SORT method"""
    kf = KalmanFilter(dim_x=6, dim_z=4)
    
    # State transition matrix
    dt = 1  # time step
    kf.F = np.array([[1, 0, dt, 0, 0, 0], 
                     [0, 1, 0, dt, 0, 0], 
                     [0, 0, 1, 0, 0, 0],
                     [0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 0, 1]])
    
    # Measurement function
    kf.H = np.array([[1, 0, 0, 0, 0, 0], 
                     [0, 1, 0, 0, 0, 0],
                     [0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 0, 1]])
    
    # Initial state estimate
    kf.x = np.array([x, y, 0, 0, s, r]).T
    
    # Covariance matrix
    kf.P *= 1000
    
    # Measurement noise
    kf.R = np.array([[5, 0, 0, 0], 
                     [0, 5, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    
    # Process noise
    kf.Q = np.array([[dt**4/4, 0, dt**3/2, 0, 0, 0],
                     [0, dt**4/4, 0, dt**3/2, 0, 0],
                     [dt**3/2, 0, dt**2, 0, 0, 0],
                     [0, dt**3/2, 0, dt**2, 0, 0],
                     [0, 0, 0, 0, 0.01, 0],
                     [0, 0, 0, 0, 0, 0.01]]) * 0.03
    
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


#----------------------------------------------------------------------------------------------------------------------------------------------------------
def conf_KF_bbox(track, tracklet, maxDisp): 
    """
    Calculate Intersection over Union (IOU) for two bounding boxes.

    Args:
    - box1: List or tuple representing the coordinates of the first bounding box in the format (x1, y1, x2, y2).
    - box2: List or tuple representing the coordinates of the second bounding box in the format (x1, y1, x2, y2).

    Returns:
    - iou: IOU value between the two bounding boxes.
    - confidence: Confidence score ranging from 1 (perfect match) to 0.
    """

    predictedbbox = track.predictedbbox
    trackletbbox = tracklet.bbox
    # print(predictedbbox, '///', trackletbbox)
    if predictedbbox is not None and np.any(predictedbbox):
        
        predicted = predictedbbox[-1]
        new = trackletbbox
        # Calculate coordinates of intersection rectangle
        x1 = max(predicted[0], new[0])
        y1 = max(predicted[1], new[1])
        x2 = min(predicted[2], new[2])
        y2 = min(predicted[3], new[3])
        # print('pass')
        # Calculate intersection area (IA)
        intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

        # Calculate union area (UA)
        predicted_area = (predicted[2] - predicted[0] + 1) * (predicted[3] - predicted[1] + 1)
        new_area = (new[2] - new[0] + 1) * (new[3] - new[1] + 1)
        union_area = predicted_area + new_area - intersection_area

        # Calculate IOU
        iou = intersection_area / union_area
        # print('|', iou)
        # Transform IOU into confidence score (1 to 0)
        # confidence = max(0, min(1, 1 - iou))
        # print(confidence, '|')

        return iou
    else:
        #Euclidean distance
        pos = track.loc
        new = tracklet.loc

        distance = np.sqrt((pos[0] - new[0]) ** 2 + (pos[1] - new[1]) ** 2)
        sigma = maxDisp[0] / 3
        confidence = np.exp(-0.5 * (distance / sigma) ** 2)
        
        return confidence


def conf_shape(track, tracklet):
    diff = abs(track.shape - tracklet.shape)/((track.shape + tracklet.shape)/2)
    conf = 1- diff
    return conf

def conf_b():
    return 0.65

def conf_c():
    return 0.80

# Vital confidence calculation functions
def confVital_a():
    # Simulated vital confidence calculation
    return 1  # Example return value

def confVital_b():
    return 0.9

# define all setup functions:

# def displacement():

# def shape():

# def boundary():

# def featureEmbedding():

# def conf_KF_POS(predictedbbox, trackletLoc, maxDisp): # functrion to get predict vs actual loc difference
#     if not predictedbbox:
#         return 0
#     else:
#         predicted = predictedbbox
#         new = trackletLoc

#         #Euclidean distance
#         distance = np.sqrt((predicted[0] - new[0]) ** 2 + (predicted[1] - new[1]) ** 2)
#         sigma = maxDisp[0] / 3
#         confidence = np.exp(-0.5 * (distance / sigma) ** 2)
#         # #weighted distance calculation
#         # normalized_weighted_distance = np.sqrt((wx * (predicted[0] - new[0]) ** 2 + wy * (predicted[1] - new[1]) ** 2) / (wx + wy))
#         # sigma = max_distance / 3
#         # confidence = np.exp(-0.5 * (normalized_weighted_distance / sigma) ** 2)
#         # # independant
#         # distance_x = abs(predicted[0] - new[0])
#         # distance_y = abs(predicted[1] - new[1])
#         # sigma_x = max_distance_x / 3
#         # sigma_y = max_distance_y / 3
#         # confidence_x = np.exp(-0.5 * (distance_x / sigma_x) ** 2)
#         # confidence_y = np.exp(-0.5 * (distance_y / sigma_y) ** 2)
#         return confidence
