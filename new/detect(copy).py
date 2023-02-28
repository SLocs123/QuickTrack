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

from QuickTrack import QuickTrack

from PIL import Image
from pprint import pprint

# det format tensor([xtopleft, ytopleft, xbottomright, ybottomright, confidence, class) ------- use .tolist() to convert det to list of coords,confidence and class


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
    path = "KITTI.names"
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

    # !!!!!!!!!!!!!!!!!!!!
    qt = QuickTrack(path)
    for path, img, im0s, vid_cap in dataset:

        frame += 1

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        qt.setImage(img)

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
                imageHeight, _, _ = im0.shape

                # det format [xtopleft, ytopleft, xbottomright, ybottomright, confidence, class]
                # Initialize on first frame
                if frame == 1:
                    qt.generateInitialTracks(detList)
                else:

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
                            confDispX = (-gradDisp[0] * np.absolute((obj['loc'][-1][0] - det_obj[0])) + 1) if np.absolute((obj['loc'][-1][0] - det_obj[0])) < maxDisp[0] else 0
                            confDispY = (-gradDisp[1] * np.absolute((obj['loc'][-1][1] - det_obj[1])) + 1) if np.absolute((obj['loc'][-1][1] - det_obj[1])) < maxDisp[1] else 0
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
                                'loc': [det_obj[:4]],
                                'conf': det_obj[4],
                                'cls': round(det_obj[5]),
                                'initialFrame': frame,
                                'colour': obj_color,
                                'Distance': [distance(det_obj[:4], names[round(det_obj[5])], imageHeight)],
                                'speed': 0,
                                'acceleration': 0,
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

                            objects[index]['loc'].append(top_conf[1])
                            objects[index]['initialFrame'] = frame
                            objects[index]['conf'] = top_conf[2]
                            objects[index]['potential'] = []
                            objects[index]['Distance'].append(distance(top_conf[1], names[objects[index]['cls']], imageHeight))
                            objects[index]['colour'] = color

                            if objects[index]['speed'] == 0:
                                objects[index]['speed'] = [speed(objects[index]['Distance'][-2], objects[index]['Distance'][-1], t2 - prevTime)]
                            else:
                                objects[index]['speed'].append(speed(objects[index]['Distance'][-2], objects[index]['Distance'][-1], t2 - prevTime))

                            if len(objects[index]['speed']) > 1:
                                if objects[index]['acceleration'] == 0:
                                    objects[index]['acceleration'] = [acceleration(objects[index]['speed'][-2], objects[index]['speed'][-1], t2 - prevTime)]
                                else:
                                    objects[index]['acceleration'].append(acceleration(objects[index]['speed'][-2], objects[index]['speed'][-1], t2 - prevTime))

                if save_det:
                    detWhole.append(detList)

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for obj in objects:
                    xyxy = obj['loc'][-1]
                    label = '%s | %d | %.3f | %.3f' % (names[obj['cls']], obj['id'], float(np.mean(obj['Distance'])), float(np.mean(obj['speed'])))
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
