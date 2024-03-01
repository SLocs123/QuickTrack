from .util import averageShape, getMiddle


class Tracklet:
    def __init__(self, Id, Obj, Colour):
        self.Id = Id
        self.bbox = Obj[:4]
        self.loc = getMiddle(self.bbox)
        self.cls = round(Obj[5])
        self.colour = Colour
        self.shape = self.__calculateShape([Obj[:4]])
        self.size = (self.bbox[2]-self.bbox[0])*(self.bbox[3]-self.bbox[1])

    @staticmethod
    def __calculateShape(loc):
        return averageShape(loc)
