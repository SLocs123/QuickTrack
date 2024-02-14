import Tracks
import Tracklet
from util import sortHighest # -------------------------------#
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color


class QuickTrack: # Quicktrack class contains all image parameters and the maximum tolerances for tracking properties. It also conatains lists of all tracks and tracklets
    def __init__(self, help=False, classPath=Names, threshold=0.7, maxDisplacement=[150, 100], maxColourDif=2000, maxShapeDif=0.5, weights=[15, 2, 2, 2], maxAge=6):

        if help:
            print('required params.txt .......') # -----------------------------------------------------Finish, or print a txt file---------------------------------------------------#
        self.img = None
        self.imgH = None
        self.thres = threshold
        self.maxDisp = maxDisplacement
        self.MaxColDif = maxColourDif
        self.maxShapeDif = maxShapeDif
        self.weights = weights
        self.path = classPath
        self.classes = self.load_classes()
        self.tracks = []
        self.tracklets = []
        self.trackletCount = 0
        sefl.maxAge = maxAge
        #self.count = 1
        self.confData = {}

    def generateInitialTracks(self, detectionList):
        tracks = []
        for detection in detectionList:
            # distance = self.getDistance(detection) # -----------------------------# define initial measurements for a track
            # colour = self.getColour(detection)
            newTrack = Tracks(self.count, detection[:4], 0, colour, distance)
            tracks.append(newTrack)
            #self.count += 1
        self.tracks = tracks    

    def updateFrame(self, img):
        self.img = img
        self.imgH, _, _ = img.shape
        self.tracklets = []
        self.trackletCount = 0

    def generateTracklets(self, detectionList):
        for detection in detectionList:
            #get colour
            colour = self.getColour(detectionList)
            tracklet = Tracklet(detectionList)
            self.tracklets.append(tracklet)    

    def updateTracks(self, trackid, trackletObject): # reassess inputs
        self.tracklets = self.generateTracklets(detectionList)
        # calculate confidences
        # assign tracklets
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

    def _assignTracklets(self): # --------------------------------------------------- Could add assignment options as well----------------#
        list = self.calculateConfidence()
        for item in list:
            if self.tracklets[item[0]] is not None:
                self.updateTrack(item[1], self.tracklets[item[0]])
                self.tracklets[item[0]] = None # ---------------------------------------- must do this for all values with the same trackID/Tracklet Id
        # for item in self.tracklets:
            # if item is not None:
                # create new track from tracklet(item)

    def removeTracks(self):
        for item in self.tracks:
            if item.age > self.maxAge:
                self.tracks.remove(item) # --------------------------------might not work----------------------# .pop(index) might be better
        # add a way to track track age





gradDisp = [1 / maxDisp[0], 1 / maxDisp[1]]
gradCol = 1 / maxColDif
gradShape = 1 / maxShapeDif

