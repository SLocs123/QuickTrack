import pickle
from shapely.geometry import Polygon, Point
import time
import numpy as np
import cv2
import math

class Kf_Trajectory:
    def __init__(self, traj_dir) -> None:
        self.traj = None
        self.polygon_set = self.read_pkl(traj_dir)
        self.polygons = self.polygon_set.pop('polygons')
        self.assigned = None
        self.trajectories = None
        self.sr = None

    def update(self, loc, dx, dy):
        loc,  _ = self.kf_to_traj(loc, dx,dy)
        return loc, self.trajectories

    def kf_to_traj(self, track_pos, kf_dx, kf_dy):
        if not self.trajectories:
            point = Point(track_pos[0], track_pos[1])
            for polygon in self.polygons:
                if polygon.contains(point):
                    self.assigned = polygon
                    break

            if not self.assigned:
                return (track_pos[0] + kf_dx, track_pos[1] + kf_dy), False

            self.sr = []
            self.trajectories = []
            for internal_dict in self.polygon_set[self.assigned].values():
                self.trajectories.append(np.array(internal_dict[:,0]))
                self.sr.append(np.array(internal_dict[:,1]))

        xys = []
        for i, traj in enumerate(self.trajectories):
            srs = self.sr[i]
            xy, sr = self.calculate_positions_along_trajectory(track_pos, traj, srs, kf_dx, kf_dy)
            xys.append([xy, sr])
        return xys, True
    
    def calculate_positions_along_trajectory(self, track_pos, trajectory, srs, dx, dy):
        current_position = np.array(track_pos, dtype=float)
        movement_scalar = np.linalg.norm(np.array([dx, dy], dtype=float))

        distances = []
        index = len(trajectory) - 1
        for i, point in enumerate(trajectory): # use middle point of start and end segments
            if i == 0:
                continue
            point = self.find_midpoint(point, trajectory[i-1])
            dist = [np.linalg.norm(current_position - point), i]
            distances.append(dist)
        _, index = min(distances, key=lambda x: x[0])

        segment_index = index
        sr = srs[segment_index]
        while segment_index < len(trajectory) - 1:
            segment_start = np.array(trajectory[segment_index - 1], dtype=float)
            segment_end = np.array(trajectory[segment_index], dtype=float)
            segment_vector = segment_end - segment_start
            segment_length = np.linalg.norm(segment_vector)
            segment_unit_vector = segment_vector / segment_length
            sr = srs[segment_index]
            
            # print(segment_end)
            # print('segment_start: ', segment_start)
            # print(segment_vector)
            # print('-----------------------------------------------------------------------------------------------------------------------------------------------')

            if not self.is_between(segment_start, segment_end, current_position):
                # Calculate correction towards trajectory
                projection = np.dot(current_position - segment_start, segment_unit_vector)
                closest_point_on_segment = segment_start + projection * segment_unit_vector
                adjustment_vector = closest_point_on_segment - current_position
                adjustment_length = np.linalg.norm(adjustment_vector)
                
                if adjustment_length > 0:
                    adjustment_vector = 0.4 * (adjustment_vector / adjustment_length)
                else:
                    adjustment_vector = np.zeros_like(adjustment_vector)

                move_vector = movement_scalar * (segment_unit_vector + adjustment_vector)
                move_length = np.linalg.norm(move_vector)
                if move_length > 0:
                    if move_length > movement_scalar:
                        current_position += movement_scalar * (move_vector / move_length)
                        # print(current_position)
                        # print('segment_unit_vector: ', segment_unit_vector)
                        # print('adjustment_vector: ', adjustment_vector)
                        # print(move_vector)
                        # print('offline')
                        break
                    else:
                        current_position += move_vector
                        # print('offline')
                        break

            else:
                # If current_position is already on the segment, move along it
                dist_to_seg_end = np.linalg.norm(current_position - segment_end)
                if movement_scalar > dist_to_seg_end:
                    movement_scalar = movement_scalar - dist_to_seg_end
                    current_position = segment_end
                    segment_index += 1
                    # print('resest movement scallar: ', movement_scalar)
                    # print('reset')
                    # print(current_position)
                else:
                    # print('movement_scalar: ', movement_scalar)
                    # print('segment_unit_vector: ', segment_unit_vector)
                    current_position = current_position + movement_scalar * segment_unit_vector
                    # print(current_position)
                    # print('less than')
                    break

        return current_position, sr
    
    def read_pkl(self, traj_dir):
        with open(traj_dir, 'rb') as pkl_file:
            loaded_data = pickle.load(pkl_file) 
        polygon_set = loaded_data
        return polygon_set

    def show_structure(self, d, indent=0):
        for key, value in d.items():
            print(' ' * indent + f'{key} ({type(value).__name__})')
            if isinstance(value, dict):
                self.show_structure(value, indent + 2)

    def is_between(self, a, b, c, epsilon=300):
        crossproduct = (c[1] - a[1]) * (b[0] - a[0]) - (c[0] - a[0]) * (b[1] - a[1])
        if abs(crossproduct) > epsilon:
            # print('epsilon check: ', crossproduct, epsilon)
            return False
        dotproduct = (c[0] - a[0]) * (b[0] - a[0]) + (c[1] - a[1]) * (b[1] - a[1])
        if dotproduct < 0:
            # print('dot product check: ', dotproduct)
            return False
        squaredlengthba = (b[0] - a[0]) * (b[0] - a[0]) + (b[1] - a[1]) * (b[1] - a[1])
        if dotproduct > squaredlengthba:
            # print('squared lengthba: ', dotproduct, squaredlengthba)
            return False
        return True
    
    def find_midpoint(self, point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        xm = (x1 + x2) / 2
        ym = (y1 + y2) / 2
        return (xm, ym)
    
def xysr_to_bbox(xy, sr):
    """
    x in form [x,y,s,r]
    Takes a bounding box in the center form [x, y, s, r] and returns it in the form
    [x1, y1, x2, y2] where x1, y1 is the top left and x2, y2 is the bottom right.
    """
    width = np.sqrt(sr[0] * sr[1])
    height = sr[0] / width
    x1= xy[0] - width / 2.0
    y1 = xy[1] - height / 2.0
    x2 = xy[0] + width / 2.0
    y2 = xy[1] + height / 2.0
    return np.array([x1, y1, x2, y2])

coords1 = (640, 1757)
track_polyTraj = Kf_Trajectory('CAM_HAZEL_TRAJS.pkl')
image = cv2.imread('useframe.jpg')

output_video = 'example-use.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
fps = 10
frame_width, frame_height = image.shape[1], image.shape[0]
video_writer = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))
cv2.circle(image, coords1, 10, (0, 0, 255), -1)
video_writer.write(image)
coords = coords1

for i in range(400):
    loc, traj = track_polyTraj.update(coords,10,15)
    image_with_position = image.copy()
    coords = loc[1][0] 
    sr = loc[1][1]
    traj = traj[1]
    center = (int(coords[0]), int(coords[1]))
    bbox = xysr_to_bbox(coords, sr)
    top_left = tuple([int(bbox[0]), int(bbox[1])])
    bottom_right = tuple([int(bbox[2]), int(bbox[3])])


    for point in traj:
        cv2.circle(image_with_position, (int(point[0]), int(point[1])),  2, (0, 255, 255), -1)
        for i in range(len(traj) - 1):
            cv2.line(image_with_position, (int(traj[i][0]), int(traj[i][1])), (int(traj[i+1][0]), int(traj[i+1][1])), (0, 255, 255), 1)
    cv2.circle(image_with_position, center, 10, (0, 0, 255), -1)
    cv2.rectangle(image_with_position, top_left, bottom_right, (0, 0, 255), 2)

    video_writer.write(image_with_position)
video_writer.release()
cv2.destroyAllWindows()