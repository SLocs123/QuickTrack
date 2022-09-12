import pickle
import os
import numpy as np
from PIL import Image
import sys
import pandas as pd

np.set_printoptions(threshold=sys.maxsize)


def depth_read(filename):
    # loads depth map D from png file
    # and returns it as a numpy array,
    # for details see readme.txt

    depth_png = np.array(Image.open(filename), dtype=int)
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert(np.max(depth_png) > 255)

    depth = depth_png.astype(float) / 256.
    depth[depth_png == 0] = -1.
    return depth


os.chdir(os.path.dirname(os.path.abspath(__file__)))
with open('dist.pickle', 'rb') as f:
    dist = pickle.load(f)
with open('det.pickle', 'rb') as f:
    det = pickle.load(f)

frames = dist[-1][-1] - 5

x1, y1, x2, y2 = det[-1][0][0][:4]
x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

# print(dist[0])
KITTIDist = []
scale = 1

data = []

for i in range(62, 63):
    file = format(i, '010')

    groundTruth = depth_read('2011_09_26_drive_0013_sync/proj_depth/groundtruth/image_03/{}.png'.format(file))
    closest = 999
    mark = 0
    average = []
    coords = []
    for index, obj in enumerate(det[i-1][0]):
        x1, y1, x2, y2 = obj[:4]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        for item in groundTruth[y1:y2, x1:x2]:
            for x in item:
                if x != -1:
                    average.append(x)

        distance = np.mean(average)
        print(distance)
        if distance < closest:
            closest, mark,  = distance, index
            coords = [x1, y1, x2, y2]

    print(closest, coords, len(groundTruth[0]))
    # Distance = dist[i-1][0][mark] / scale
    # KITTIDistance = closest
    #
    # if i % 50 == 0 or i == 5:
    #     Diff = dist[i-1][0][mark] / KITTIDistance
    #     scale = Diff
    #     Distance = dist[i-1][0][mark] / scale

    # print('-------------------------', 'Frame: ', i)

    # for index, obj in enumerate(det[i-1][0]):
    #     x1, y1, x2, y2 = obj[:4]
    #     x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    #     # print('coords: ', x1, y1, x2, y2)
    #     groundTruth = depth_read('2011_09_26_drive_0013_sync/proj_depth/groundtruth/image_03/{}.png'.format(file))
    #     average = []
    #     for item in groundTruth[y1:y2, x1:x2]:
    #         for x in item:
    #             if x != -1:
    #                 average.append(x)
    #
    #     KITTIDist = np.mean(average)
    #     predDist = dist[i-1][0][index] / scale
    #     Diff = predDist / KITTIDist
    #
    #     # print('Kitti dist: ', KITTIDist)
    #     # print('Predicted Dist: ', predDist)
    #     # print('Difference: ', Diff)
    #     # print('-------')
    # data.append([KITTIDistance, Distance, scale, i])
    #
    #
    #     # print('Scaled', scale, Diff, KITTIDistance, Distance)

# df = pd.DataFrame(data)
#
# df.to_excel('out2.xlsx', sheet_name='sheet1', index=False)
