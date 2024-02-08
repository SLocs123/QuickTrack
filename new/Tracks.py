import time
from util import calculateTimeDependents, averageShape
# or import util ?????????????????????????????


class Tracks:
    def __init__(self, Id, Obj, Frame, Colour, Distance, Bounds=[6, 2]):
        self.Id = Id
        self.loc = [Obj[:4]]
        self.cls = round(Obj[5])
        self.conf = Obj[4]
        self.initialFrame = Frame
        self.colour = Colour
        self.shape = self.__calculateShape([Obj[:4]])
        self.distance = [Distance]
        self.speed = []
        self.acceleration = []
        self.upper = Bounds[0]
        self.lower = Bounds[1]
        self.kf = util.create_kalman_filter()

    def updatePosition(self, position):
        self.__setLoc(position)
        self.__setShape()

    def updateMovement(self, distance):
        self.__setDistance(distance)

        if len(self.distance) <= self.lower:
            return None
        self.__setSpeed()

        if len(self.speed) <= self.lower:
            return None
        self.__setAcceleration()

    def __setShape(self):
        self.shape = self.__calculateShape(self.loc)

    def __setDistance(self, distance):
        self.distance.append([distance, time.time()])
        if len(self.distance) > self.upper:
            self.distance = self.distance[-self.upper:]

    def __setSpeed(self):
        speed = self.__calculateSpeed()
        self.speed.append(speed)
        if len(self.speed) > self.upper:
            self.speed = self.speed[-self.upper:]

    def __setAcceleration(self):
        accel = self.__calculateAccel()
        self.acceleration.append(accel)
        if len(self.acceleration) > self.upper:
            self.acceleration = self.acceleration[-self.upper:]

    def __setLoc(self, coords):
        self.loc.append(coords)
        if len(self.loc) > self.upper:
            self.loc = self.loc[-self.upper:]

    def __calculateSpeed(self):
        prev = self.distance[-2]
        current = self.distance[-1]
        t = current[1] - prev[1]
        return calculateTimeDependents(current, prev, t)

    def __calculateAccel(self):
        prev = [self.speed[-2], self.distance[-2][1]]
        current = [self.speed[-1], self.distance[-1][1]]
        t = current[1] - prev[1]
        return calculateTimeDependents(current, prev, t)

    def __compareToTracklet(self, tracklet):
        return tracklet

    def getId(self):
        return self.id

    @staticmethod
    def __calculateShape(loc):
        return averageShape(loc)

