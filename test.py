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

results = model.predict('output.mp4') # Runs yolov5 inference, not the quickest option though

predictions = results.pred[0] # pulls tensor from the results object

# Open the video file
cap = cv2.VideoCapture('output.mp4')

tracker = QT.QuickTrack()
frame = 0

while cap.isOpened():
    frame +=1
    ret, img = cap.read()
    if not ret:
        break
#--------------------------------------------------------------------------------------------------------------#
    
    # Inference
    results = model(img)
    pred = results.pred[0]
    detectionList = pred.tolist()
    #Tracking
    if frame == 1:
        tracker.generateInitialTracks()
    else:
        tracker.update(detectionList, img)

#--------------------------------------------------------------------------------------------------------------#
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
