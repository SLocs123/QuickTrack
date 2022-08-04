import argparse
import os
import platform
import shutil
import time
from pathlib import Path
# from skimage.color import rgb2lab, deltaE_cmc, deltaE_ciede94, deltaE_ciede2000, deltaE_cie76
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000, delta_e_cmc, delta_e_cie1994, delta_e_cie1976

import scipy.cluster
import sklearn.cluster
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np

from PIL import Image
# from deep_sort_realtime.deepsort_tracker import DeepSort

from utils.google_utils import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer)
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

from models.models import *
from utils.datasets import *
from utils.general import *

from PIL import Image
from pprint import pprint

# det format tensor([xtopleft, ytopleft, xbottomright, ybottomright, confidence, class) ------- use .tolist() to convert det to list of coords,confidence and class


def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)


def speed(prev_loc, loc, cls, t):
    # Crude estimation for distance, based on pixel size and distance references
    # Can use Dist-Yolo run on images to give a correct reference

    referenceValues = {
        'person': [60, 20],
        'car': [100, 20],
        'truck': [120, 20],
        'van': [140, 20],
        'bus': [170, 20]
    }

    # current distance
    currheight = loc[3]-loc[1]
    referenceDist = referenceValues[cls]  # reference height, reference distance
    currentDist = referenceDist[1] / (currheight / referenceDist[0])

    # previous distance
    prevheight = prev_loc[3] - prev_loc[1]
    previousDistance = referenceDist[1] / (prevheight / referenceDist[0])

    if previousDistance > currentDist:
        direction = 'Towards'
        objSpeed = (previousDistance - currentDist) / t
    elif currentDist > previousDistance:
        direction = 'Away'
        objSpeed = (currentDist - previousDistance) / t
    else:
        objSpeed = 0
        direction = 'Same'

    return objSpeed, direction


def crop(img, multiplier):  # multiplier is the amount of height you want to see from the bottom
    shape = img.shape
    change = int(shape[0] * multiplier)

    crop_img = img[0 - change:shape[0], 0:shape[1]]
    resized = cv2.resize(crop_img, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
    return resized

#
# def reshape(img, multiplier, detList):
#     # shifts the bounding boxes back down, assuming that the top has been cropped
#     width, height = img.size
#     diff = height * multiplier
#
#     # get items in det, find the height, add the diff to each height, then can be put back on normal image
#     for i in range(len(detList)):
#         detList[i][1] = detList[i][1] + diff
#         detList[i][3] = detList[i][3] + diff
#
#     # return new det
#     return diff


# def classFilter(detList, filterList):
#     # gives the co-ords and class of boxes that are in the filter list
#     coords = []
#     for i in range(len(detList)):
#         if int(detList[i][5]) in filterList:
#             coords.append(detList[i])
#     return coords

def dominant_colors(image):  # PIL image input # https://stackoverflow.com/questions/3241929/python-find-dominant-most-common-color-in-an-image

    image = image.resize((150, 150))      # optional, to reduce time
    ar = np.asarray(image)
    shape = ar.shape
    ar = ar.reshape(np.product(shape[:2]), shape[2]).astype(float)

    kmeans = sklearn.cluster.MiniBatchKMeans(
        n_clusters=10,
        init="k-means++",
        max_iter=20,
        random_state=1000
    ).fit(ar)
    codes = kmeans.cluster_centers_

    vecs, _dist = scipy.cluster.vq.vq(ar, codes)         # assign codes
    counts, _bins = np.histogram(vecs, len(codes))    # count occurrences

    index = np.argsort(counts)[::-1][0]
    color = tuple([int(code) for code in codes[index]])

    return color                   # returns colors in order of dominance


def detect(save_img=False):
    prevTime = 0
    out, source, weights, view_img, save_txt, save_det, imgsz, cfg, names = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.save_det, opt.img_size, opt.cfg, opt.names
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = Darknet(cfg, imgsz)#.cuda() #if you want cuda remove the comment

    model.load_state_dict(torch.load(weights[0], map_location=device)['model'])
    # model = attempt_load(weights, map_location=device)  # load FP32 model
    # imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    model.to(device).eval()
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz, auto_size=64)

    # Get names and colors
    names = load_classes(names)
    colors = [[np.random.randint(0, 50) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    # Globals
    frame = 0
    objects = []
    obj_counter = 0
    detWhole = []
    for path, img, im0s, vid_cap in dataset:
        frame += 1

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections

        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size

                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()  # original code

                detList = det.tolist()
                # Initialize on first frame
                if frame == 1:
                    for obj in detList:
                        x1, x2 = int(obj[0]), int(obj[2])  # getting bounding box values
                        y1, y2 = int(obj[1]), int(obj[3])

                        width = x2 - x1
                        height = y2 - y1

                        # crop_img = im0[y1:y2, x1:x2]
                        #
                        # # im_pil = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                        # im_pil = Image.fromarray(crop_img)
                        # color = dominant_colors(im_pil)
                        # b, g, r = color

                        b, g, r = im0[int((obj[1] + height / 2)), int((obj[0] + width / 2))]  # finding the color for the middle pixel
                        # color = np.array([[[r, g, b]]])
                        # color = rgb2lab(color)
                        color = sRGBColor(r, g, b)
                        color = convert_color(color, LabColor)  # convert to LAB domain
                        obj_counter += 1

                        objects.append({  # defining a tracking object
                            'id': obj_counter,
                            'loc': obj[:4],
                            'prev_loc': [],
                            'conf': obj[4],
                            'cls': round(obj[5]),
                            'initialFrame': frame,
                            'colour': color,
                            'shape': height/width,
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
                    maxColDif = 2000  # max delta e/euclidean difference
                    maxShapeDif = 0.5

                    gradDisp = [1 / maxDisp[0], 1 / maxDisp[1]]
                    gradCol = 1/maxColDif
                    gradShape = 1/maxShapeDif

                    # 1,2,3,4,5,6
                    # weightDisp, weightCol, weightClass
                    weights = [15, 2, 2]

                    for det_obj in detList:
                        flag = True

                        x1, x2 = int(det_obj[0]), int(det_obj[2])
                        y1, y2 = int(det_obj[1]), int(det_obj[3])

                        width = x2 - x1
                        height = y2 - y1

                        # crop_img = im0[y1:y2, x1:x2]
                        # # im_pil = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                        # im_pil = Image.fromarray(crop_img)
                        # obj_color = dominant_colors(im_pil)
                        # b, g, r = obj_color

                        # shape = height/width

                        b, g, r = im0[int((det_obj[1] + height / 2)), int((det_obj[0] + width / 2))]
                        # obj_color = np.array([[[r, g, b]]])
                        # obj_color = rgb2lab(obj_color)

                        obj_color = sRGBColor(r, g, b)
                        obj_color = convert_color(obj_color, LabColor)

                        for index, obj in enumerate(objects):

                            # using y = -mc + c  c = 1 for confidence, y = 0 is f x is beyond tolerance.
                            conf = []

                            # displacment - change in xyxy from 1 to 0
                            confDispX = (-gradDisp[0] * np.absolute((obj['loc'][0] - det_obj[0])) + 1) if np.absolute((obj['loc'][0] - det_obj[0])) < maxDisp[0] else 0
                            confDispY = (-gradDisp[1] * np.absolute((obj['loc'][1] - det_obj[1])) + 1) if np.absolute((obj['loc'][1] - det_obj[1])) < maxDisp[1] else 0
                            conf.append(np.average([confDispX, confDispY]))

                            # colour
                            delta_e = delta_e_cie1976(obj_color, obj['colour'])  # using the 1976 colour difference standard for performance
                            # delta_e = deltaE_cie76(obj_color, obj['colour'])
                            conf.append(-gradCol * delta_e + 1 if delta_e < maxColDif else 0)

                            # # shape - dx/dy
                            # diff = np.absolute((shape - obj['shape']))
                            # conf.append(-gradShape * diff + 1 if diff < maxShapeDif else 0)

                            # # class
                            conf.append(1 if round(det_obj[5]) == obj['cls'] else 0)

                            numerator = sum([conf[i]*weights[i] for i in range(len(conf))])  # finding a weighted average
                            denominator = sum(weights)

                            conf = numerator/denominator

                            if conf > tol:
                                #  potential = tracking confidence, location, prediction conf, class
                                objects[index]['potential'].append([conf, det_obj[:4], det_obj[4], round(det_obj[5])])  # potential tracks
                                flag = False

                            if frame - obj['initialFrame'] >= 4:  # deletes object if it hasnt been updated in 4 frames
                                del objects[index]
                        if flag:  # defines a new object if there are no potentials
                            obj_counter += 1
                            objects.append({
                                'id': obj_counter,
                                'loc': det_obj[:4],
                                'prev_loc': [],
                                'conf': det_obj[4],
                                'cls': round(det_obj[5]),
                                'initialFrame': frame,
                                'colour': obj_color,
                                # 'shape': shape,
                                'speed': 0,
                                'potential': []
                            })
                    for index, obj in enumerate(objects):  # updates tracked objects
                        if len(obj['potential']) > 0:
                            top_conf = sorted(obj['potential'], key=lambda x: x[0], reverse=True)[0]

                            # x1, x2 = int(top_conf[1][0]), int(top_conf[1][2])
                            # y1, y2 = int(top_conf[1][1]), int(top_conf[1][3])
                            # crop_img = im0[y1:y2, x1:x2]
                            #
                            # im_pil = Image.fromarray(crop_img)
                            # color = dominant_colors(im_pil)
                            # b, g, r = color

                            width = top_conf[1][2] - top_conf[1][0]
                            height = top_conf[1][3] - top_conf[1][1]
                            b, g, r = im0[int((top_conf[1][1] + height / 2)), int((top_conf[1][0] + width / 2))]
                            # color = np.array([[[r, g, b]]])
                            # color = rgb2lab(color)
                            color = sRGBColor(r, g, b)
                            color = convert_color(color, LabColor)

                            objects[index]['prev_loc'] = obj['loc']
                            objects[index]['loc'] = top_conf[1]
                            objects[index]['initialFrame'] = frame
                            objects[index]['conf'] = top_conf[2]
                            objects[index]['potential'] = []
                            # objects[index]['shape'] = [height/width]
                            objects[index]['colour'] = color

                if save_det:
                    detWhole.append(detList)

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for obj in objects:
                    xyxy = obj['loc']
                    label = '%s | %d | %.2f' % (names[obj['cls']], obj['id'], obj['speed'])
                    plot_one_box(xyxy, im0, label=label, color=colors[obj['cls']], line_thickness=3)
                for *xyxy, conf, cls in det:
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format
                    # if save_img or view_img:  # Add bbox to image
                    #     label = '%s %.2f' % (names[int(cls)], conf)
                    #     plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

            # Print time (inference + NMS)
            # print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                currTime = time.time()
                fps = 1 / (currTime - prevTime)
                prevTime = currTime
                cv2.putText(im0, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 196, 255), 2)
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % Path(out))
        if platform == 'darwin' and not opt.update:  # MacOS
            os.system('open ' + save_path)

    if save_det:
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        with open("X.pickle", 'wb') as f:
            pickle.dump(detWhole, f)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolor_p6.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=1280, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-det', action='store_true',  help='save det list')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--cfg', type=str, default='cfg/yolor_p6.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/coco.names', help='*.cfg path')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
