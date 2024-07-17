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


def load_classes(path):
     # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)


#----------------------------------------------------------------------------------------------------------------------------------------------------------
