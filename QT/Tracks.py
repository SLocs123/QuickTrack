# import time
from .util import averageShape, getMiddle, bbox_to_z, x_to_bbox
from .KF_Traj.kf_to_Traj.kf_to_traj import Kf_Trajectory
from .kalmanFilter import createKF, KFTrustworthy
import numpy as np
# or import util ?????????????????????????????
import time


class Tracks:
    def __init__(self, Id, Obj, Frame, Colour, Bounds=[6, 2]):
        filename = 'CAM_HAZEL_TRAJS.pkl' # This needs to be updated if using different video and labels
        self.Id = Id
        self.bbox = Obj[:4]
        self.bboxes = [Obj[:4]]
        z = bbox_to_z(self.bbox)
        z = z.flatten().tolist()
        self.loc = z[:2]
        self.cls = round(Obj[5])
        self.conf = Obj[4]
        self.frame = Frame
        self.colour = Colour
        self.size = z[2]
        self.shape = z[3]
        self.kf = createKF(self.loc[0], self.loc[1], self.size, self.shape)
        self.predictedPOS = []
        self.predictedbbox = []
        self.assigned = False
        self.kf_to_traj = Kf_Trajectory(filename)
        self.age = 0

    def updateTrack(self, tracklet): # could be better to incorporate these parameters in a smarter way, using previous observations as support. Future work for now
        self.bbox = tracklet.bbox
        self.bboxes.append(tracklet.bbox)
        self.loc = tracklet.loc
        self._updateKF([tracklet.loc[0], tracklet.loc[1], tracklet.size, tracklet.shape])
        self.shape = tracklet.shape
        self.colour = tracklet.colour
        self.size = tracklet.size
        self.conf = tracklet.conf
        self.frame = tracklet.frame
        self.assigned = True
        self.age += 1

    
    def assignTracklet(self, tracklet): # I dont think this is being used
         self._updateKF([tracklet.loc[0], tracklet.loc[1], tracklet.size, tracklet.shape])


    def _updateKF(self, newxysr):
        print('run')
        # newxy contains [x, y, s, r]
        self.kf.update(np.array(newxysr))
        self.kf.predict()
        # print(self.kf.x)
        # if KFTrustworthy(self, [10, 10, 5, 5]):
        #         # print('trustworthy')
        x,y,dx,dy,s,r = self.kf.x
        out,  _ = self.kf_to_traj.update(self.loc, dx, dy, s, r, kf_loc=[x,y], bbox=True, both=True)
        bbox = out[1]
        xysr = out[0]
        xy = xysr[0]
        self.predictedPOS.append(xy)
        self.predictedbbox.append(bbox)


    def getId(self):
        return self.id


    @staticmethod
    def __calculateShape(bbox):
        return averageShape(bbox)
