import yolov5
import QT
import cv2

tracker = QT.QuickTrack() # Initialise the quicktracker

model = yolov5.load('yolov5s.pt')

model.conf = 0.25  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.max_det = 1000  # maximum number of detections per image

# Open the video file
cap = cv2.VideoCapture('output.mp4')

tracker = QT.QuickTrack()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Inference
    results = model(frame)
    pred = results.pred[0]
    detectionList = pred.tolist()
    tracker.update(detectionList, frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
