import json
import numpy as np

def labelstudio_labels_to_yolo(labelstudio_labels_json_path: str, index_video):
    labels = json.load(open(labelstudio_labels_json_path))[index_video]
    
    # every box stores the frame count of the whole video, so we get it from the first box
    frames_count = labels['annotations'][0]['result'][0]['value']['framesCount'] +1
    yolo_labels = [[] for _ in range(frames_count)]

    # iterate through boxes
    for box in labels['annotations'][0]['result']:
       
        label_numbers = [2 for label in box['value']['labels']]
        # iterate through keypoints (we omit the last keypoint because no interpolation after that)
        for i, keypoint in enumerate(box['value']['sequence'][:-1]):
            start_point = keypoint
            end_point = box['value']['sequence'][i + 1]
            start_frame = start_point['frame']
            end_frame = end_point['frame']

            n_frames_between = end_frame - start_frame
            delta_x = (end_point['x'] - start_point['x']) / n_frames_between
            delta_y = (end_point['y'] - start_point['y']) / n_frames_between
            delta_width = (end_point['width'] - start_point['width']) / n_frames_between
            delta_height = (end_point['height'] - start_point['height']) / n_frames_between

            # In YOLO, x and y are in the center of the box. In Label Studio, x and y are in the corner of the box.
            x = start_point['x'] + start_point['width'] / 2
            y = start_point['y'] + start_point['height'] / 2
            width = start_point['width']
            height = start_point['height']
            car_id = box["id"]            
            
            # iterate through frames between two keypoints
            for frame in range(start_frame, end_frame):
                # Support for multilabel
                yolo_labels = _append_to_yolo_labels(yolo_labels, frame, label_numbers, x, y, width, height, car_id)
                x += delta_x + delta_width / 2
                y += delta_y + delta_height / 2
                width += delta_width
                height += delta_height            
         

        # Handle last keypoint
        yolo_labels = _append_to_yolo_labels(yolo_labels, frame+1, label_numbers, x, y, width, height, car_id)
       
       
    
    # json output file
    frames_data = {}
    
    # Loop through each frame and its labels
    yolo_labels = yolo_labels[1:]
    for frame, frame_labels in enumerate(yolo_labels):
        # Create a unique ID for each frame, e.g., "frame_1", "frame_2", etc.
        frame_id = frame
        frames_data[frame_id] = []

        # Process each label in the frame
        for label in frame_labels:
            # Extract the values
            car_id, label_type, min_x, min_y, max_x, max_y = label

            # Create a dictionary for the label
            label_data = {
                "car_id": car_id,
                "type": label_type,
                "min_x": min_x,
                "min_y": min_y,
                "max_x": max_x,
                "max_y": max_y
            }

            # Append the label data to the frame data
            frames_data[frame_id].append(label_data)

    # Write to a JSON file
    #with open('frame_labels.json', 'w') as outfile:
        #json.dump(frames_data, outfile, indent=4)
    
    return frames_data
        

def _append_to_yolo_labels(yolo_labels: list, frame: int, label_numbers: list, x, y, width, height, car_id):

    # we need min_x, min_y, max_x, max_y
    min_x = x - width / 2
    min_y = y - height / 2
    max_x = x + width / 2
    max_y = y + height / 2


    for label_number in label_numbers:
        yolo_labels[frame].append(
            [car_id, label_number, min_x, min_y, max_x, max_y])
        #print(f"appended {[label_number, x, y, width, height]}")
    return yolo_labels


def transform_labelstudio_input(label_studio_json_path, videoindex):
    # input = ".data/labels_CAM-HAZELDELL-126THST.json"
    # videoindex = 44


    #data = json.load(open(label_studio_json_path))
    #json.dump(data, open(input, 'w'), indent=4)

    transformed_json = labelstudio_labels_to_yolo(label_studio_json_path, int(videoindex))

    return transformed_json

def x_to_bbox(x): # conversions originally from SORT code!
    """
    x in form [x,y,s,r]
    Takes a bounding box in the center form [x, y, s, r] and returns it in the form
    [x1, y1, x2, y2] where x1, y1 is the top left and x2, y2 is the bottom right.
    """
    width = np.sqrt(x[2] * x[3])
    height = x[2] / width
    x1= x[0] - width / 2.0
    y1 = x[1] - height / 2.0
    x2 = x[0] + width / 2.0
    y2 = x[1] + height / 2.0
    return np.array([x1, y1, x2, y2])


def bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1, y1, x2, y2] and returns z in the form
    [x, y, s, r] where x, y is the center of the box, s is the scale/area, and r is
    the aspect ratio.
    """
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    x = bbox[0] + width / 2.0
    y = bbox[1] + height / 2.0
    scale = width * height  # scale is just area
    aspectRatio = width / float(height)
    return np.array([x, y, scale, aspectRatio])

