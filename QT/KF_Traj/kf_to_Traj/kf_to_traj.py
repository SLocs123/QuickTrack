import pickle
from shapely.geometry import Polygon, Point
import time
import numpy as np
import cv2
import math
import os

class Kf_Trajectory:
    def __init__(self, traj_dir) -> None:
        current_script_directory = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(current_script_directory, traj_dir)
        self.polygon_set = self.read_pkl(full_path) # Polygons, the polygons they link to and the tracjectory that connects them sahpe: {polygon1: {polygons3: traj, polygon4: traj}, polygon2: {polygon5: traj}} etc
        self.polygons = self.polygon_set.pop('polygons') # all polygons
        self.active_polygons = list(self.polygon_set.keys()) # Identify the set of polygons that act as the starting points
        self.assigned = None
        self.trajectories = None
        self.active_traj = None
        self.sr = None

    def update(self, loc, dx, dy, kf_s, kf_r, kf_loc=None, bbox=False, both=False):
        """
        Return in the format [[x,y], [s,r]] or [x1,x2,y1,y2] if bbox=True
        if both true return [[[x,y], [s,r]], [x1,x2,y1,y2]]

        input tuples or list of x,y coords !!have not implemented input bbox, need KF output anyway
        """
        if loc[0] != int: # This is for use in QuickTrack, If this is causing issues then commant out and input tuple for loc
            loc = (loc[0][0], loc[1][0])

        if kf_loc == None and self.active_traj == None:
            output_loc = [[loc[0] + dx, loc[1] + dy], [kf_s, kf_r]]
        elif kf_loc == None:
            location,  _ = self.kf_to_traj(loc, dx, dy, active=True)
            output_loc = location
        else:
            kf_loc = np.array(kf_loc)
            locs, _ = self.kf_to_traj(loc, dx,dy,)

            closest = [[float('inf')], [0]]
            for location in locs:
                current = np.linalg.norm(location[0] - kf_loc)
                if current < closest[0]:
                    closest = [[current], location]
            output_loc = closest[1]

        if output_loc[1] is None:
            output_loc[1] = [kf_s, kf_r]

        if bbox:
            output_bbox = self.convert_to_bbox(output_loc)
            if both:
                return [output_loc, output_bbox], self.trajectories
            else:
                return output_bbox, self.trajectories
        return output_loc, self.trajectories

    def kf_to_traj(self, track_pos, kf_dx, kf_dy, active=False):
        # print('self.trajectories1: ', self.trajectories)
        if not self.trajectories:
            point = Point(track_pos[0], track_pos[1])
            for polygon in self.active_polygons:
                if polygon.contains(point):
                    self.assigned = polygon
                    break

            if not self.assigned:
                return [[[track_pos[0] + kf_dx, track_pos[1] + kf_dy], None]], False


            self.sr = []
            self.trajectories = []
            for internal_dict in self.polygon_set[self.assigned].values():
                self.trajectories.append(np.array(internal_dict[:,0]))
                self.sr.append(np.array(internal_dict[:,1]))
        
        if active:
            current_trajs = self.active_traj
        else:
            current_trajs = self.trajectories

        # print('self.trajectories2: ', self.trajectories)
        xys = []
        for i, traj in enumerate(current_trajs):
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
            print('loop')
            segment_start = np.array(trajectory[segment_index - 1], dtype=float)
            segment_end = np.array(trajectory[segment_index], dtype=float)
            segment_vector = segment_end - segment_start
            segment_length = np.linalg.norm(segment_vector)
            segment_unit_vector = segment_vector / segment_length
            sr = srs[segment_index]
            
          
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
                        # print('current_position: ', current_position)
                        # print('movement_scalar: ', movement_scalar)
                        # print('move_vector: ', move_vector)
                        # print('move_length: ', move_length)
                        current_position += movement_scalar * (move_vector / move_length)
                        break
                    else:
                        current_position += move_vector
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
    
    def convert_to_bbox(self, loc):
        x,y = loc[0]
        s,r = loc[1]
        width = np.sqrt(s * r)
        height = s / width
        x1= x - width / 2.0
        y1 = y - height / 2.0
        x2 = x + width / 2.0
        y2 = y + height / 2.0
        return [x1,y1,x2,y2]


    
# def xysr_to_bbox(xy, sr):
#     """
#     x in form [x,y,s,r]
#     Takes a bounding box in the center form [x, y, s, r] and returns it in the form
#     [x1, y1, x2, y2] where x1, y1 is the top left and x2, y2 is the bottom right.
#     """
#     width = np.sqrt(sr[0] * sr[1])
#     height = sr[0] / width
#     x1= xy[0] - width / 2.0
#     y1 = xy[1] - height / 2.0
#     x2 = xy[0] + width / 2.0
#     y2 = xy[1] + height / 2.0
#     return np.array([x1, y1, x2, y2])

# coords1 = (640, 1757)
# track_polyTraj = Kf_Trajectory('CAM_HAZEL_TRAJS.pkl')
# # image = cv2.imread('useframe.jpg')

# # output_video = 'example-use.mp4'
# # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
# # fps = 10
# # frame_width, frame_height = image.shape[1], image.shape[0]
# # video_writer = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))
# # cv2.circle(image, coords1, 10, (0, 0, 255), -1)
# # video_writer.write(image)
# coords = coords1

# for i in range(400):
#     loc, traj = track_polyTraj.update(coords,10,15, kf_loc=[10,10])
#     # image_with_position = image.copy()
#     coords = loc
#     print(coords)
#     # sr = loc[1][1]
#     # traj = traj[1]
#     # center = (int(coords[0]), int(coords[1]))
#     # bbox = xysr_to_bbox(coords, sr)
#     # top_left = tuple([int(bbox[0]), int(bbox[1])])
#     # bottom_right = tuple([int(bbox[2]), int(bbox[3])])


#     # for point in traj:
#     #     cv2.circle(image_with_position, (int(point[0]), int(point[1])),  2, (0, 255, 255), -1)
#     #     for i in range(len(traj) - 1):
#     #         cv2.line(image_with_position, (int(traj[i][0]), int(traj[i][1])), (int(traj[i+1][0]), int(traj[i+1][1])), (0, 255, 255), 1)
#     # cv2.circle(image_with_position, center, 10, (0, 0, 255), -1)
#     # cv2.rectangle(image_with_position, top_left, bottom_right, (0, 0, 255), 2)

# #     video_writer.write(image_with_position)
# # video_writer.release()
# # cv2.destroyAllWindows()