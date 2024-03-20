from .Tracks import Tracks
from .Tracklet import Tracklet
from .util import *
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
import sys
import cv2
import random

# implementation of SAE - https://github.com/starwit/sae-stage-template


class QuickTrack: # Quicktrack class contains all image parameters and the maximum tolerances for tracking properties. It also conatains lists of all tracks and tracklets
    def __init__(self, help=False, classPath='QT/default.names', threshold=0.7, maxDisplacement=[150, 100], maxColourDif=2000, maxShapeDif=0.5, weights=[15, 2, 2], maxAge=6, colour='no', vitalScale=0.7, assign='greedy'):

        if help:
            print('required params.txt .......') # -----------------------------------------------------Finish, or print a txt file---------------------------------------------------#
        self.img = None # the img is the current frame being inferenced, needs to be updated every frame in the update function
        # self.imgH = None # img used for distance estimation, not necassary unless distance is reintroduced
        self.thres = threshold # confidence threshold for tracklet to be assigned as potential to a track
        self.maxDisp = maxDisplacement # maximum [x, y] displacement from predicted position, allows creatiion of a gradient for confidence calc
        self.MaxColDif = maxColourDif # same as previous but for colour, not relevant when using ML method, simple is not reliable
        self.maxShapeDif = maxShapeDif # same as previous, but for object shape, or aspect ratio
        self.weights = weights # Weights is the input to adjust how different confidence factors are weighted in the average, check conf.py for conf functions, then input weights in same order for preference
        self.classes = load_classes(classPath) # Allows the loading of classes, allows translation from inference class integer to wording. This file must allign with the inference type e.g. coco.names
        self.tracks = [] # These are the confirmed tracks, could look at using sets!!!!!!!!!!!!!!!!!
        self.trackCount = 0 # allows unique naming of tracks
        self.tracklets = [] # These are tracklets created from detection list, could look at using sets!!!!!!!!!!!!!!!!!
        self.trackletCount = 0 # allows unique naming of tracklets
        self.maxAge = maxAge # determines how old a track can be without an update before being removed
        self.frame = 0 # used to track how old the sequence is
        self.colour = colour # this is the colour method preference, none is recomended atm
        self.vitalScale = vitalScale
        self.assign = assign
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.classes]

                # Get only functions from util.py that start with 'conf_'
        # self.confMetrics = [func for func in dir(util) if callable(getattr(util, func)) and func.startswith("conf_")]
        # # Identify vital functions that start with 'confVital_'
        # self.vitalFunctions = [func for func in dir(util) if callable(getattr(util, func)) and func.startswith("confVital_")]

    def _generateInitialTracks(self, detectionList):
        tracks = []
        for detection in detectionList:
            # distance = self.getDistance(detection) # -----------------------------# define initial measurements for a track
            # colour = self.getColour(detection)
            if self.colour == 'ML':
                colour = getColourML(detection, self.img)
            if self.colour == 'Simple':
                colour = getColourSimple(detection, self.img)
            else:
                colour = None   
            newTrack = Tracks(self.trackCount, detection, 0, colour)
            tracks.append(newTrack)
            self.trackCount += 1
        self.tracks = tracks    

    def update(self, detectionList, img):
        self._updateFrame(img)
        if self.frame == 0:
            self._generateInitialTracks(detectionList)
        else:
            self.tracklets = self._generateTracklets(detectionList)
            self._updateTracks()
        self.frame += 1

    def show(self):
        for track in self.tracks:
            xyxy = track.bbox
            label = '%s | %d' % (self.classes[track.cls], track.Id)
            self.plot_one_box(xyxy, self.img, label=label, color=self.colors[track.cls], line_thickness=3)
        cv2.imshow('Output', self.img)
        cv2.waitKey(1)  # 1 millisecond

    def _updateFrame(self, img):
        self.img = img
        # self.imgH, _, _ = img.shape
        self.tracklets = []
        self.trackletCount = 0

    def _generateTracklets(self, detectionList):
        tracklets = []
        for detection in detectionList:
            if self.colour == 'ML':
                colour = getColourML(detection, self.img)
            if self.colour == 'Simple':
                colour = getColourSimple(detection, self.img)
            else:
                colour = None
            tracklet = Tracklet(self.trackletCount, detection, colour) #??????????????? review the detection formatting ----------------------------!
            tracklets.append(tracklet)
            self.trackletCount += 1
        return tracklets

    def _updateTracks(self): # reassess inputs
        confs = self._calculateConfidence()
        # confs = self._calculateConfidence(False)
        # print(confs)
        self._assignTracklets(confs)

        # assign tracklets/update position track=tracklet
        # create new tracks, if not assigned
        # remove old tracks, too long without update
        self.tracklets = [] # remove when done

    def _calculateConfidence(self):
        # need identifier for if a tracklet has been assigned to potential and if it gets updated
        trackConfidence = []
        for track in self.tracks:
            # trackConfidence[track.Id] = []
            for tracklet in self.tracklets:
                conf = self._calculateWeightedConfidence(track, tracklet) # calculate confidence for each track tracklet pair
                # if conf >= self.thres:
                #     trackConfidence.append([track.Id, tracklet.Id, conf])
                trackConfidence.append([track.Id, tracklet.Id, conf])
        return trackConfidence

    def _assignTracklets(self, confs): # --------------------------------------------------- Could add assignment options as well, hungarian etc...----------------#
        if self.assign == 'greedy':
            confs = sortHighest(confs)
            assignedTrackIds = set()
            assignedTrackletIds = set()
            for conf in confs:
                if conf[0] in assignedTrackIds or conf[1] in assignedTrackletIds:
                    continue  
                track = next((track for track in self.tracks if track.Id == conf[0]), None)
                if track is not None:
                    # Find the tracklet by ID
                    tracklet = next((tracklet for tracklet in self.tracklets if tracklet.Id == conf[1]), None)
                    if tracklet is not None:
                        track.updateTrack(tracklet)
                        assignedTrackIds.add(track.Id)
                        assignedTrackletIds.add(tracklet.Id)
            # generate new tracks for unassigned
        # elif self.assign == 'hungarian':
        #     continue

    def __removeTracks(self):
        for item in self.tracks:
            if item.age > self.maxAge:
                self.tracks.remove(item) # --------------------------------might not work----------------------# .pop(index) might be better
        # add a way to track track age
                
    def _calculateWeightedConfidence(self, track, tracklet):
        total_conf = 0
        total_weight = sum(self.weights)
        confs = []
        confs_vital = []

        # call confidence functions here
        # ------------------------------------------------------------------------------------- #
        confs.append(conf_a())
        confs.append(conf_b())
        confs.append(conf_c())
        confs_vital.append(confVital_a())
        confs_vital.append(confVital_b())
        # ------------------------------------------------------------------------------------- #
        flag = False
        for conf in confs_vital:
            if conf != 1:
                flag = True
                break


        if len(self.weights) == len(confs):
            for conf, weight in zip(confs, self.weights):
                total_conf += conf * weight
            weighted_confidence = total_conf / total_weight
            if flag:
                weighted_confidence = weighted_confidence * self.vitalScale 
            return weighted_confidence
        else:
            sys.exit('The number of inputted weights must match the number of non vital functions called, check _calculate_weighted_confidence function in Quicktrack.py')
   
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
