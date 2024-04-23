import yolov5
import QT
import cv2

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
    # print(detectionList)
    _ = tracker.update(detectionList, frame)
    outFrame = tracker.show()
    print("-------------------------------")

    out.write(outFrame)
    if cv2.waitKey(1) == ord('q'):
        break



print('Done')
cap.release()
cv2.destroyAllWindows()
