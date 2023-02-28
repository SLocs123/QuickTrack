from Tracks import Tracks
from Tracklet import Tracklet
from util import sortHighest
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color

referenceValues = {
    'person': [60, 20],
    'car': [1367.955976688, -1.0088759860],
    'truck': [120, 20],
    'van': [140, 20],
    'bus': [170, 20]
}


class QuickTrack:
    def __init__(self, classPath, threshold=0.7, maxDisplacement=[150, 100], maxColourDif=2000, maxShapeDif=0.5,  weights=[15, 2, 2, 2]):

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
        self.count = 1

    def updateFrame(self, img):
        self.img = img
        self.imgH, _, _ = img.shape
        self.tracklets = []
        self.trackletCount = 0

    def generateInitialTracks(self, detectionList):
        tracks = []
        for detection in detectionList:
            distance = self.getDistance(detection)
            colour = self.getColour(detection)
            newTrack = Tracks(self.count, detection[:4], 0, colour, distance)
            tracks.append(newTrack)
            self.count += 1
        self.tracks = tracks

    def updateTrack(self, trackid, trackletObject):
        self.tracklets = self.generateTracklets(detectionList)
        # search for the correct track id

    def generateTracklets(self, detectionList):
        for detection in detectionList:
            #get colour
            colour = self.getColour(detectionList)
            tracklet = Tracklet(detectionList)
            self.tracklets.append(tracklet)
        #Make sure to wipe tracklet cound and array after each frame

    def calculateConfidence(self):
        # need identifier for if a tracklet has been assigned to potential and if it gets updated
        trackConfidence = []
        for track in self.tracks:
            trackConfidence[track.id] = []
            for tracklet in self.tracklets:
                # Logic for generating confidence
                conf = track.compare(tracklet)
                trackConfidence.append([tracklet.id, track.id, conf])
        return sortHighest(trackConfidence)

    def assignTracklets(self):
        list = self.calculateConfidence()
        for item in list:
            if self.tracklets[item[0]] is not None:
                self.updateTrack(item[1], self.tracklets[item[0]])
                self.tracklets[item[0]] = None
        for item in self.tracklets:
            if item is not None:
                # create new track from tracklet(item)


    # def removeTrack(self, id):
    #     self.tracks = [track for track in self.tracks if track.getId() != id]

    def getColour(self, detection):
        width = detection[2] - detection[0]
        height = detection[3] - detection[1]
        b, g, r = self.img[int((detection[1] + height / 2)), int((detection[0] + width / 2))]
        color = sRGBColor(r, g, b)
        return convert_color(color, LabColor)

    def getDistance(self, detection):
        refHeight = 194  # this is the height of the model pulled from monodepth2, to ensure the correct standard is used (could also change to percentage of screen but either works)
        scale = refHeight / self.imgH
        cls = round(detection[5])

        # current distance
        height = detection[3] - detection[1]
        height = height * scale

        equ = referenceValues[cls]  # reference numerator, reference power
        Dist = round(equ[0] * height ** equ[1], 3)

        return Dist

    def load_classes(self):
        # Loads *.names file at 'path'
        with open(self.path, 'r') as f:
            names = f.read().split('\n')
        return list(filter(None, names))  # filter removes empty strings (such as lastline)

    # def updateTracks(self):



gradDisp = [1 / maxDisp[0], 1 / maxDisp[1]]
gradCol = 1 / maxColDif
gradShape = 1 / maxShapeDif

