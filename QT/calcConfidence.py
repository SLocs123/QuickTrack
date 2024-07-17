import numpy as np


def calculateWeightedConfidence(self, track, tracklet, KF=False, FE=False, Zone=False, Shape=False, weights=None):
    total_conf = 0
    total_weight = sum(weights)
    confs = []
    confs_vital = []
    # call confidence functions here
    # ------------------------------------------------------------------------------------- # 
    if KF: confs.append(conf_KF_bbox(track, tracklet, self.maxDisp))
    if FE: confs.append(0.5)#run feture comparison
    if Shape: confs.append(conf_shape(track, tracklet))
    
    # confs_vital.append(confVital_a(track, tracklet))
    # confs_vital.append(confVital_b(track, tracklet))
    # ------------------------------------------------------------------------------------- #
    if weights is not None and len(weights) != len(confs):
        raise ValueError("The number of inputted weights must match the number of non-vital functions called. Check the _calculateweightedconfidence function in Quicktrack.py and reassess QuickTrack inputs")
    flag = False
    for conf in confs_vital:
        if conf != 1:
            flag = True
            break
    for conf, weight in zip(confs, weights):
        total_conf += conf * weight
    weighted_confidence = total_conf / total_weight
    if flag:
        weighted_confidence = weighted_confidence * self.vitalScale 
    return weighted_confidence


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
