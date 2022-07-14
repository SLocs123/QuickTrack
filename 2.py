import itertools
import os
import numpy as np
from tqdm import tqdm
import pickle

confValues = np.arange(9)/10
xValues = np.arange(11)*50
yValues = np.arange(11)*50
weightDisp = np.arange(15)
weightCol = np.arange(15)
weightShape = np.arange(15)
weightClass = np.arange(15)

count = 0
# for x in tqdm(list(itertools.product(confValues, xValues, yValues, weightDisp, weightCol, weightShape, weightClass))):

# read
os.chdir(os.path.dirname(os.path.abspath(__file__)))
with open('X.pickle', 'rb') as f:
    detList = pickle.load(f)
print(detList[0][0][0])
'''

 if frame == 1:
                    for obj in detList:
                        width = obj[2] - obj[0]
                        height = obj[3] - obj[1]

                        b, g, r = im0[int((obj[1] + height / 2)), int((obj[0] + width / 2))]
                        obj_counter += 1
                        # width = obj[2]-obj[0]
                        # height = obj[3] - obj[1]

                        objects.append({
                            'id': obj_counter,
                            'loc': obj[:4],
                            'prev_loc': [],
                            'conf': obj[4],
                            'cls': round(obj[5]),
                            'initialFrame': frame,
                            'colour': [r, g, b],
                            'speed': 0,
                            'potential': []
                        })
                else:
                    tol = 0.7
                    # a low tolerance threshold will prevent erratic tracking, but if too low will stop new trakers being created.
                    # Must found correct blalace between gradients and thresholds
                    # To completed, same colour gives 0.33
                    # distance, if same gives 0.75, but a deviance of more than about 75 pixels should be looked at

                    maxDisp = [150, 100]  # [x, y]
                    grad = [1/maxDisp[0], 1/maxDisp[1]]

                    # gradCol = 255*0.05
                    # gradSize = (0.25/35000)
                    # gradShape = 0.25/100

                    # 1,2,3,4,5,6
                    # weightDisp, weightCol, weightsize, weightShape, weightClass
                    weights = [20, 1]

                    for det_obj in detList:
                        flag = True
                        width = det_obj[2] - det_obj[0]
                        height = det_obj[3] - det_obj[1]
                        b, g, r = im0[int((det_obj[1] + height / 2)), int((det_obj[0] + width / 2))]

                        for index, obj in enumerate(objects):

                            # using y = -mc + c  c = 1 for confidence, y = 0 is f x is beyond tolerance.
                            conf = []

                            # displacment - change in xyxy
                            confDispX = (-grad[0] * np.absolute((obj['loc'][0] - det_obj[0])) + 1) if np.absolute((obj['loc'][0] - det_obj[0])) < maxDisp[0] else 0
                            confDispY = (-grad[1] * np.absolute((obj['loc'][1] - det_obj[1])) + 1) if np.absolute((obj['loc'][1] - det_obj[1])) < maxDisp[1] else 0
                            conf.append(np.average([confDispX, confDispY]))

                            # color - get colour of centre pixel
                            # colourDiff = np.average(np.absolute(b - obj['colour'][0]), np.absolute(g - obj['colour'][1]), np.absolute(r - obj['colour'][2]))
                            # conf.append(-gradCol * colourDiff + 0.6)

                            # # # shape - dx/dy
                            # conf.append(-gradShape * np.absolute((shape - obj['shape'])) + 0.25) if np.absolute((shape - obj['shape'])) < 100 else 0)

                            # # class
                            conf.append(1 if round(det_obj[5]) == obj['cls'] else 0)

                            numerator = sum([conf[i]*weights[i] for i in range(len(conf))])
                            denominator = sum(weights)

                            conf = numerator/denominator

                            if conf > tol:
                                #  potential = tracking confidence, location, prediction conf, class
                                objects[index]['potential'].append([conf, det_obj[:4], det_obj[4], round(det_obj[5])])
                                flag = False

                            if frame - obj['initialFrame'] >= 4:
                                del objects[index]
                        if flag:
                            obj_counter += 1
                            objects.append({
                                'id': obj_counter,
                                'loc': det_obj[:4],
                                'prev_loc': [],
                                'conf': det_obj[4],
                                'cls': round(det_obj[5]),
                                'initialFrame': frame,
                                # 'colour': [255, 255, 255],
                                'speed': 0,
                                'potential': []
                            })
                    for index, obj in enumerate(objects):
                        if len(obj['potential']) > 0:
                            top_conf = sorted(obj['potential'], key=lambda x: x[0], reverse=True)[0]
                            # width = top_conf[1][2] - top_conf[1][0]
                            # height = top_conf[1][3] - top_conf[1][1]
                            objects[index]['prev_loc'] = obj['loc']
                            objects[index]['loc'] = top_conf[1]
                            objects[index]['initialFrame'] = frame
                            objects[index]['conf'] = top_conf[2]
                            objects[index]['potential'] = []
'''