from .Tracks import Tracks
from .Tracklet import Tracklet
from .util import *
from scipy.optimize import linear_sum_assignment
# from colormath.color_objects import sRGBColor, LabColor
# from colormath.color_conversions import convert_color
import cv2
import random
import warnings
from __future__ import annotations
# implementation of SAE - https://github.com/starwit/sae-stage-template


class QuickTrack:
    def __init__(self, confParams: list[str]=[["KF"], ["KF", "FE"]], classPath: str='QT/default.names', threshold: list[float]=[0.7, 0.3], maxDisplacement: list[int]= [150,100], maxColourDif: int=2000, maxShapeDif: float=0.5, weights: list[int]=[[1], [1,2]], maxAge: int=6, colour: str='no', vitalScale: float=0.7, assign: str='LinAssign'):
        """
        img is the current frame being inferenced, this also needs to be passed into the update function
        thres is the confidence threshold that gates a track-tracklet conf. The conf must be higher than this value
        the max variables determine a gradient for confidence calculations, these will be tuned for stationary traffic camera conditions, but changing them could help for varying scenarios - Implement adaptable max disp!!!!!!!!!!!!!
        weights represents the weighting of the confidence fundtions. These are called in the _calculateWeightedConfidence function
        classes must point to class list that the inference uses, defaut is the coco set
        tracks ----------------------------
        trackCount is used to name new tracks, it doesn't represent the number of LIVE tracks ut the total tracks created
        tracklets
        trackletCount, same as above
        maxAge determines the maximum age of a track withoutbeing updated, once this is passed the track will be removed/deactivated
        frame is the number of frames, not the img. 
        colour is the colour method preference, |none is recomended atm|
        vitalScale is the value which the confidence is scaled down by when vital functions aren't fulfilled |subject to change|
        assign is the method used to assign tracks-tracklets
        colours is used to show bbox, unique for each class

        POTENTIAL CHANGES:
        Look at using sets or numpy arrays for the tracks and tracklets, to try and increase loop speeds
        np arrays for colour theory parts
        """
        if colour not in {'no', 'simple', 'ML'}:
            warnings.warn(f"Invalid colour '{colour}'. Defaulting to 'no'.", RuntimeWarning)
            self.colour = 'no'
        else:
            self.colour = colour
        if assign not in {'greedy', 'LinAssign'}:
            warnings.warn(f"Invalid assign method '{assign}'. Defaulting to 'greedy'.", RuntimeWarning)
            self.assign = 'greedy'
        else:
            self.assign = assign  

        self.confParams = confParams
        self.img = None 
        self.thres = threshold
        self.maxDisp = maxDisplacement 
        self.maxColDif = maxColourDif 
        self.maxShapeDif = maxShapeDif 
        self.weights = weights 
        self.classes = load_classes(classPath)
        self.tracks = []
        self.trackCount = 0
        self.tracklets = []
        self.trackletCount = 0
        self.maxAge = maxAge
        self.frame = 0 
        self.vitalScale = vitalScale
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.classes]

    def update(self, detectionList, img):
        """
        This is the main function while running, will process all other operations in the tracker.
        return is alligned to the SAE benchmarking format: https://github.com/starwit/object-tracker-benchmark/tree/main
        """
        self._updateFrame(img)
        if self.frame == 0:
            self._generateInitialTracks(detectionList)
        else:
            self.tracklets = self._generateTracklets(detectionList)
            self._updateTracks()
        self.frame += 1

        out = []
        for track in self.tracks:
            out.append([track.bbox[0],track.bbox[1], track.bbox[2], track.bbox[3], track.Id, track.conf, track.cls, 1])
        return out


    def show(self):
        """
        Show will output a video stream showing all tracks labelled in the inference/input video
        Uses the yolo format/approach
        """
        for track in self.tracks:
            xyxy = track.bbox
            label = '%s | %d' % (self.classes[track.cls], track.Id)
            self.plot_one_box(xyxy, self.img, label=label, color=self.colors[track.cls], line_thickness=3)
        # cv2.imshow('Output', self.img)
        # cv2.waitKey(1)  # 1 millisecond, 0 for a keypress
        return self.img

    def _generateInitialTracks(self, detectionList):
        """
        Called for the first frame only, this function will convert all detections into tracks instead of tracklets
        """
        tracks = []
        for detection in detectionList:
            colour = self._getColour(detection)
            newTrack = Tracks(self.trackCount, detection, 0, colour) # ----------------- this might be an error detection
            tracks.append(newTrack)
            self.trackCount += 1
        self.tracks = tracks        

    def _updateFrame(self, img):
        self.img = img
        self.tracklets = []
        self.trackletCount = 0

    def _generateTracklets(self, detectionList):
        tracklets = []
        for detection in detectionList:
            colour = self._getColour(detection)
            tracklet = Tracklet(self.trackletCount, detection, colour, self.frame) #??????????????? review the detection formatting ----------------------------!
            tracklets.append(tracklet)
            self.trackletCount += 1
        return tracklets

    def _updateTracks(self):
        for track in self.tracks:
            track.assigned = False

        for param_set in self.confParams:
            tracks = []
            tracklets = []
            # Create a dictionary of keyword arguments
            kwargs = {
                'KF': 'KF' in param_set,
                'FE': 'FE' in param_set,
                'Zone': 'Zone' in param_set,
                'Shape': 'Shape' in param_set
            }
            for track in self.tracks:
                if not track.assigned:
                    tracks.append(track)

            confs = self._calculateConfidence(tracks, kwargs) # needs testing

            self._assignTracklets(confs) # needs testing
        

        for tracklet in self.tracklets:
            obj = [tracklet.bbox[0], tracklet.bbox[1], tracklet.bbox[2], tracklet.bbox[3], tracklet.conf, tracklet.cls]
            self.tracks.append(Tracks(self.trackCount, obj, self.frame, tracklet.colour))
            self.trackCount+=1
        self.__removeTracks()
        self.tracklets = []

    def _calculateConfidence(self, tracks, kwargs):
        numTracks = len(tracks)
        numTracklets = len(self.tracklets)

        confMatrix = np.zeros((numTracks,numTracklets))

        for x, track in enumerate(tracks):
            for y, tracklet in enumerate(self.tracklets):
                conf = self._calculateWeightedConfidence(track, tracklet, **kwargs) # calculate confidence for each track tracklet pair  # if conf >= self.thres: #     trackConfidence.append([track.Id, tracklet.Id, conf])
                confMatrix[x, y] = conf # potenitally convert to np array
        return confMatrix

    def _assignTracklets(self, confs, assign=None):
        if assign == None:
            assign = self.assign
        if assign == 'greedy':
            confs = sortHighest(confs) # potenitally convert to np array and np sorting, investigate priority queues instead of sorting
            # assignedTrackIds = set()
            # assignedTrackletIds = set()
            # for conf in confs:
            #     if conf[0] in assignedTrackIds or conf[1] in assignedTrackletIds:
            #         continue  
            #     track = next((track for track in self.tracks if track.Id == conf[0]), None)
            #     if track is not None:
            #         tracklet = next((tracklet for tracklet in self.tracklets if tracklet.Id == conf[1]), None) 
            #         if tracklet is not None:
            #             track.updateTrack(tracklet)
            #             assignedTrackIds.add(track.Id)
            #             assignedTrackletIds.add(tracklet.Id)
            # # --------- potentially add after all assign logic options to avoid repeated code, assignments would have to use same logic though-----------#
            # unassignedTracklets = [tracklet for tracklet in self.tracklets if tracklet.Id not in assignedTrackletIds]
            # for tracklet in unassignedTracklets:
            #     obj = [tracklet.bbox[0], tracklet.bbox[1], tracklet.bbox[2], tracklet.bbox[3], tracklet.conf, tracklet.cls]
            #     newTrack = Tracks(self.trackCount, obj, self.frame, tracklet.colour)
            #     self.tracks.append(newTrack)
            #---------------------------------------------------------------------------------------------------------------------------------------------#
        elif assign == 'LinAssign':
            matches = self.linSumAssign(confs)

            for x, y in matches: 
                if confs[x,y] > self.thres:
                    self.tracks[x].updateTrack(self.tracklets[y])
                    self.tracklets[y] = None

        self.tracklets = [tracklet for tracklet in self.tracklets if tracklet is not None]



    def __removeTracks(self): # ------------------------------------------------------------ needs to be looked at again, use assigned
         self.tracks = [item for item in self.tracks if self.frame - item.frame <= self.maxAge]


    def _calculateWeightedConfidence(self, track, tracklet, KF=False, FE=False, Zone=False, Shape=False):
        total_conf = 0
        total_weight = sum(self.weights)
        confs = []
        confs_vital = []

        # call confidence functions here
        # ------------------------------------------------------------------------------------- # look at passing only required information, instead of whole tracks
        if KF: confs.append(conf_KF_bbox(track, tracklet, self.maxDisp))
        # if FE: #run feture comparison
        if Shape: confs.append(conf_shape(track, tracklet))
        
        # confs_vital.append(confVital_a(track, tracklet))
        # confs_vital.append(confVital_b(track, tracklet))
        # ------------------------------------------------------------------------------------- #
        if len(self.weights) != len(confs):
            raise ValueError("The number of inputted weights must match the number of non-vital functions called. Check the _calculate_weighted_confidence function in Quicktrack.py.")

        flag = False
        for conf in confs_vital:
            if conf != 1:
                flag = True
                break

        for conf, weight in zip(confs, self.weights):
            total_conf += conf * weight
        weighted_confidence = total_conf / total_weight
        if flag:
            weighted_confidence = weighted_confidence * self.vitalScale 
        return weighted_confidence
    

    def linSumAssign(self, confs):
        costs = 1- confs
        x, y = linear_sum_assignment(costs)
        return np.array(list(zip(x, y)))
            
   
    def plot_one_box(self, x, img, color=None, label=None, line_thickness=3):
        # Plots one bounding box on image img
        tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    def _getColour(self, detection):
        if self.colour == 'ML':
            colour = getColourML(detection, self.img)
        if self.colour == 'simple':
            colour = getColourSimple(detection, self.img)
        else:
            colour = None
        return colour