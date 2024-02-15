from .Tracks import Tracks
from .Tracklet import Tracklet
from .util import sortHighest, load_classes, getColourML, getColourSimple
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color


class QuickTrack: # Quicktrack class contains all image parameters and the maximum tolerances for tracking properties. It also conatains lists of all tracks and tracklets
    def __init__(self, help=False, classPath='QT/default.names', threshold=0.7, maxDisplacement=[150, 100], maxColourDif=2000, maxShapeDif=0.5, weights=[15, 2, 2, 2], maxAge=6, colour='no'):

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
        self.tracks = [] # These are the confirmed tracks
        self.trackCount = 0 # allows unique naming of tracks
        self.tracklets = [] # These are tracklets created from detection list
        self.trackletCount = 0 # allows unique naming of tracklets
        self.maxAge = maxAge # determines how old a track can be without an update before being removed
        self.frame = 0 # used to track how old the sequence is
        self.colour = colour # this is the colour method preference, none is recomended atm

    def generateInitialTracks(self, detectionList):
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
                
            newTrack = Tracks(self.trackCount, detection[:4], 0, colour)
            tracks.append(newTrack)
            #self.count += 1
        self.tracks = tracks    

    def update(self, detectionList, img):
        self._updateFrame(img)
        if self.frame == 0:
            self.generateInitialTracks(detectionList)
        self.tracklets = self.generateTracklets(detectionList)
        self._updateTracks()
        self.frame += 1

    def _updateFrame(self, img):
        self.img = img
        # self.imgH, _, _ = img.shape
        self.tracklets = []
        self.trackletCount = 0

    def _generateTracklets(self, detectionList):
        for detection in detectionList:
            if self.colour == 'ML':
                colour = getColourML(detection, self.img)
            if self.colour == 'Simple':
                colour = getColourSimple(detection, self.img)
            else:
                colour = None
            tracklet = Tracklet(self.trackletCount, detection[:4], colour) #??????????????? review the detection formatting ----------------------------!
            self.tracklets.append(tracklet)

    def _updateTracks(self): # reassess inputs
        # calculate confidences
        # assign tracklets/update position track=tracklet
        # create new tracks, if not assigned
        # remove old tracks, too long without update
        self.tracklets = []

    def _calculateConfidence(self):
        # need identifier for if a tracklet has been assigned to potential and if it gets updated
        trackConfidence = []
        for track in self.tracks:
            trackConfidence[track.id] = []
            for tracklet in self.tracklets:
                # generate util setup
                # calculate confidence for each track tracklet pair
                conf = track.__compareToTracklet(tracklet)
                trackConfidence.append([tracklet.id, track.id, conf])
        return sortHighest(trackConfidence)

    def _assignTracklets(self): # --------------------------------------------------- Could add assignment options as well, hungarian etc...----------------#
        list = self.calculateConfidence()
        for item in list:
            if self.tracklets[item[0]] is not None:
                self.updateTrack(item[1], self.tracklets[item[0]])
                self.tracklets[item[0]] = None # ---------------------------------------- must do this for all values with the same trackID/Tracklet Id
        # for item in self.tracklets:
            # if item is not None:
                # create new track from tracklet(item)

    def __removeTracks(self):
        for item in self.tracks:
            if item.age > self.maxAge:
                self.tracks.remove(item) # --------------------------------might not work----------------------# .pop(index) might be better
        # add a way to track track age

#gradDisp = [1 / maxDisp[0], 1 / maxDisp[1]]
#gradCol = 1 / maxColDif
#gradShape = 1 / maxShapeDif

