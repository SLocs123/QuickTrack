import yolov5
import QT
import cv2
import random


def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

tracker = QT.QuickTrack() # Initialise the quicktracker

model = yolov5.load('yolov5x6.pt')

model.conf = 0.25  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.max_det = 1000  # maximum number of detections per image
model.classes = [2]

print('innit')  
# Open the video file
cap = cv2.VideoCapture('output.mp4')
output_video_path = 'output_video.avi'
fps = 30  # Frames per second
codec = cv2.VideoWriter_fourcc(*'XVID')
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_video_path, codec, fps, (frame_width, frame_height))


tracker = QT.QuickTrack()
# print('innit2')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Inference
    results = model(frame)
    pred = results.pred[0]
    detectionList = pred.tolist()
    print(len(detectionList))
    # for track in detectionList:
    #     xyxy = track[:4]
    #     plot_one_box(xyxy, frame)
    _ = tracker.update(detectionList, frame)
    outFrame = tracker.show()
    print("-------------------------------")

    out.write(outFrame)
    if cv2.waitKey(1) == ord('q'):
        break



print('Done')
cap.release()
cv2.destroyAllWindows()
