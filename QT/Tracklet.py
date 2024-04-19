from .util import averageShape, bbox_to_z


class Tracklet:
    def __init__(self, Id, Obj, Colour, frame):
        self.Id = Id
        self.bbox = Obj[:4]
        z = bbox_to_z(self.bbox)
        self.loc = z[:2]
        self.cls = round(Obj[5])
        self.conf = Obj[4]
        self.colour = Colour
        self.frame = frame
        self.size = z[2]
        self.shape = z[3]


    @staticmethod
    def __calculateShape(loc):
        return averageShape(loc)
