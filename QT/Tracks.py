import time
from .util import calculateTimeDependents, averageShape, createKF, KFTrustworthy, getMiddle
import numpy as np
# or import util ?????????????????????????????


class Tracks:
    def __init__(self, Id, Obj, Frame, Colour, Bounds=[6, 2]):
        self.Id = Id
        self.bbox = Obj[:4]
        self.loc = getMiddle(self.bbox)
        self.cls = round(Obj[5])
        self.conf = Obj[4]
        self.initialFrame = Frame
        self.colour = Colour
        self.shape = [self.__calculateShape([Obj[:4]])]
        self.size = [(self.bbox[2]-self.bbox[0])*(self.bbox[3]-self.bbox[1])]
        # self.distance = [Distance]
        # self.speed = []
        # self.acceleration = []
        # self.upper = Bounds[0] # What are these? can figure it out
        # self.lower = Bounds[1] # What are these? can figure it out
        self.kf = createKF(self.loc[0], self.loc[1])
        self.predictedPOS = None
        #self.age = () # --------------------------------------------------!!


    def updateTrack(self, tracklet):
        self.loc.append(tracklet.loc)
        self._updateKF(tracklet.loc)
        self.shape.append(tracklet.shape)
        self.colour = tracklet.colour
        self.size.append(tracklet.size)
        self.bbox = tracklet.bbox


    def _updateKF(self, newxy):
        # print(np.array(newxy))
        self.kf.update(np.array(newxy))
        predict = self.kf.predict()
        if KFTrustworthy(self, [10, 10, 5, 5]):
            self.predictedPOS = predict
        

    # def _updatePosition(self, position):
    #     self.__setLoc(position)
    #     self.__setShape()


    # def __setShape(self, bbox):
    #     self.shape.append(self.__calculateShape(bbox))


    # def __setLoc(self, coords):
    #     self.loc.append(coords)
        # if len(self.loc) > self.upper:
        #     self.loc = self.loc[-self.upper:]


    # def __compareToTracklet(self, tracklet):
    #     return tracklet


    def getId(self):
        return self.id


    @staticmethod
    def __calculateShape(bbox):
        return averageShape(bbox)



    # --------------------------------------------------------------------------------------------------------------- Distance, Ignore for now ----------------------------------------------- #

#    def updateMovement(self, distance):
#        self.__setDistance(distance)
#
#        if len(self.distance) <= self.lower:
#            return None
#        self.__setSpeed()
#
#        if len(self.speed) <= self.lower:
#            return None
#        self.__setAcceleration()
#
#    def __setDistance(self, distance):
#        self.distance.append([distance, time.time()])
#        if len(self.distance) > self.upper:
#            self.distance = self.distance[-self.upper:]
#    
#    def __setSpeed(self):
#        speed = self.__calculateSpeed()
#        self.speed.append(speed)
#        if len(self.speed) > self.upper:
#            self.speed = self.speed[-self.upper:]
#
#    def __setAcceleration(self):
#        accel = self.__calculateAccel()
#        self.acceleration.append(accel)
#        if len(self.acceleration) > self.upper:
#            self.acceleration = self.acceleration[-self.upper:]
#
#    def __calculateSpeed(self):
#        prev = self.distance[-2]
#        current = self.distance[-1]
#        t = current[1] - prev[1]
#        return calculateTimeDependents(current, prev, t)
#
#    def __calculateAccel(self):
#        prev = [self.speed[-2], self.distance[-2][1]]
#        current = [self.speed[-1], self.distance[-1][1]]
#        t = current[1] - prev[1]
#        return calculateTimeDependents(current, prev, t)
#   referenceValues = {
#       'person': [60, 20],
#       'car': [1367.955976688, -1.0088759860],
#       'truck': [120, 20],
#       'van': [140, 20],
#       'bus': [170, 20]
#   }


    # --------------------------------------------------------------------------------------------------------------- Distance, Ignore for now ----------------------------------------------- #
