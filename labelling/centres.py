import yolov5
import argparse
import cv2

# Load the YOLOv5 model
model = yolov5.load('yolov5x6.pt')
model.conf = 0.25  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.max_det = 1000  # maximum number of detections per image
model.classes = [2]

def bbox_to_center(bbox):
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return cx, cy

def main(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    output_video_path = 'output_video.mp4'
    fps = 30  # Frames per second
    codec = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' codec for MP4 format
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, codec, fps, (frame_width, frame_height))
    
    centresList = []  # List to store all centers for the entire video, formatted
    all_centers = []  # List to store all centers for the entire video

    while cap.isOpened():
        current_centers = []
        centres = []
        ret, frame = cap.read()
        if not ret:
            break

        # Inference
        results = model(frame)
        pred = results.pred[0]
        detectionList = pred.tolist()
        for det in detectionList:
            centre = bbox_to_center(det[:4])
            center_point = tuple(map(int, centre))
            centres.append(centre)
            current_centers.append(center_point)
        
        centresList.append(centres)  # Store centers for this frame
        all_centers.extend(current_centers)
        
        for center_point in all_centers:
            cv2.circle(frame, center_point, 5, (0, 0, 255), -1)  # Draw center as red circle


        out.write(frame)

        if cv2.waitKey(1) == ord('q'):
            break

    # After processing all frames, print or use centresList as needed
    print('Done')

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a video file.")
    parser.add_argument("video_path", type=str, help="The path to the video file")
    args = parser.parse_args()

    main(args.video_path)