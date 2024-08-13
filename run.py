import yolov5
import QT
import cv2
import random
import time


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

model = yolov5.load('yolov5x6.pt')

model.conf = 0.25  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.max_det = 1000  # maximum number of detections per image
model.classes = [2]

print('Initialized YOLOv5 model')  
# Open the video file
cap = cv2.VideoCapture('CAM-HAZELDELL-126THST.mp4')
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get the width and height of the frames
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
output_video_path = 'SAE-PhD-overveiw-labelled.mp4'
fps = 30  # Frames per second
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

tracker = QT.QuickTrack()
print('Initialized QuickTrack')


def draw_detections_and_write(frame, detections):
    """
    Draws detections on the frame and writes the frame to the video writer.

    Args:
    - frame (ndarray): The current frame.
    - detections (list): A list of detections, where each detection is a dictionary
                         with keys 'bbox' (bounding box as [x1, y1, x2, y2]), 'conf' (confidence score), and 'cls' (class label).
    """
    for detection in detections:
        bbox = detection[:4]
        conf = detection[4]
        cls = detection[5]

        # Draw the bounding box
        x1, y1, x2, y2 = bbox
        # print(x1,y1,x2,y2)
        # time.sleep(60)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # # Put the class label and confidence score
        # label = f"{cls} {conf:.2f}"
        # cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

frame_number = 0
printlabelled = True
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Print the current frame number
    print(f'Processing frame {frame_number}', end='\r')
    # Inference
    results = model(frame)
    pred = results.pred[0]
    detectionList = pred.tolist()
    # _ = tracker.update(detectionList, frame)
    #outFrame = tracker.show()
    if printlabelled:
        draw_detections_and_write(frame, detectionList)

    out.write(frame)
    if cv2.waitKey(1) == ord('q'):
        break
    frame_number += 1
print('Done')
cap.release()
cv2.destroyAllWindows()
