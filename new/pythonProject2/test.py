import time

from Tracklet import Tracklet

t1 = Tracklet(1, [100, 0, 120, 70, 0.7, 2], [], 0, [100, 0])

for i in range(0, 10):
    t1.updatePosition([100, i * 10, 120, 70])
    print(t1.loc, t1.shape)
    time.sleep(1)
