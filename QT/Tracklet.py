from .util import averageShape


class Tracklet:
    def __init__(self, Id, Obj, Colour):
        self.Id = Id
        self.loc = [Obj[:4]]
        self.cls = round(Obj[5])
        self.colour = Colour
        self.shape = self.__calculateShape([Obj[:4]])

    @staticmethod
    def __calculateShape(loc):
        return averageShape(loc)
