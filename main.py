import cv2
import logging
import csv
from statistics import mean
import numpy as np
from scipy.spatial import distance as dist


logging.basicConfig(level=logging.INFO)


class VideoReader:
    def __init__(self, video_path):
        if not video_path:
            logging.error("Video path is empty.")
            raise ValueError("Video path must not be empty.")
        
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            logging.error(f"Could not open video file: {video_path}")
            raise IOError("Failed to open video file.")
        
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logging.info(f"Successfully opened video file with {self.frame_count} frames.")

    def read_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            logging.warning("Failed to read frame. End of video or error.")
            return None
        return frame

    def close(self):
        self.cap.release()
        logging.info("Released video capture resources.")



class ROIDrawer:
    def __init__(self):
        self.roi_points = []  # To store (center_x, center_y, radius) tuples
        self.current_roi = []

    def _select_roi(self, event, x, y, flags, param):
        frame_copy = param

        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_roi = [(x, y)]
        elif event == cv2.EVENT_LBUTTONUP:
            self.current_roi.append((x, y))
            center = (
                (self.current_roi[0][0] + self.current_roi[1][0]) // 2,
                (self.current_roi[0][1] + self.current_roi[1][1]) // 2,
            )
            radius = int(
                np.sqrt(
                    (self.current_roi[1][0] - self.current_roi[0][0]) ** 2
                    + (self.current_roi[1][1] - self.current_roi[0][1]) ** 2
                )
                // 2
            )

            if center[0] >= 0 and center[1] >= 0:
                cv2.circle(frame_copy, center, radius, (0, 255, 0), 2)
                self.roi_points.append((center[0], center[1], radius))
                self.current_roi = []

            cv2.imshow("Select Regions of Interest", frame_copy)

    def draw_rois(self, frame):
        cv2.imshow("Select Regions of Interest", frame)
        cv2.setMouseCallback("Select Regions of Interest", self._select_roi, param=frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def get_rois(self):
        return self.roi_points


class Preprocessor:
    def __init__(self, mean_frame=None):
        if mean_frame is not None and len(mean_frame.shape) == 3:
            mean_frame = cv2.cvtColor(mean_frame, cv2.COLOR_BGR2GRAY)
        self.mean_frame = mean_frame



    def preprocess(self, frame):
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Frame difference
        diff_frame = cv2.absdiff(self.mean_frame, frame)
       
        alpha = 255 / np.max(diff_frame)
        abs_frame = cv2.convertScaleAbs(diff_frame, alpha=alpha, beta=0)



        
        # Thresholding
        _, thresh = cv2.threshold(abs_frame, 30, 255, cv2.THRESH_BINARY)
        # thresh = cv2.adaptiveThreshold(abs_frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

        
        # Morphological Operations
        kernel = np.ones((3, 3), np.uint8)
        closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return closing

    # def preprocess(self, frame):
    #     if len(frame.shape) == 3:
    #         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    #     diff_frame = cv2.absdiff(self.mean_frame, frame)
    #     equalized_frame = cv2.equalizeHist(diff_frame)
        
    #     alpha = 255 / np.max(equalized_frame)
    #     abs_frame = cv2.convertScaleAbs(equalized_frame, alpha=alpha, beta=0)
        
    #     _, thresh = cv2.threshold(abs_frame, 30, 255, cv2.THRESH_BINARY)
    #     kernel = np.ones((3, 3), np.uint8)
    #     closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
    #     return closing


    def update_mean_frame(self, new_frame, exclusion_zones, exclusion_radius=10):
        if len(new_frame.shape) == 3:
            new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
        
        # Create an exclusion mask
        mask = np.ones(new_frame.shape, dtype=np.uint8) * 255
        for x, y, _ in exclusion_zones:  # Assuming exclusion_zones is a list of (x, y, area)
            cv2.circle(mask, (x, y), exclusion_radius, 0, -1)  # Draw black circles on white mask
        
        # Update mean frame only where the mask is white
        self.mean_frame = np.where(mask == 255, new_frame, self.mean_frame)


class MosquitoDetector:
    def __init__(self, rois, min_size=30, max_size=200, liberal_size=120):
        self.rois = rois  # [(center_x, center_y, radius), ...]
        self.min_size = min_size  # Minimum contour area
        self.max_size = max_size  # Maximum contour area
        self.liberal_size = liberal_size  # A more lenient maximum size
        self.repeated_positions = {}  # {(x, y): count}

    def _is_within_roi(self, x, y):
        for cx, cy, r in self.rois:
            if (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2:
                return True
        return False

    def detect_mosquitoes(self, frame):
        detected_mosquitoes = []
        dilated = cv2.dilate(frame, None, iterations=2)

        # Find contours
        #3 options here RTREXTERNAL, RETLIST, RETCCOMP
        #could also do morphological closing

        contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # contours = cv2.findContours(frame, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        # contours, _ = cv2.findContours(frame, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Filter based on size
            area = cv2.contourArea(contour)
            # logging.info(f"Contour area: {area}")
            if area < self.min_size or area > self.max_size:
                continue

            

            # Get the center of the contour
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            # Check if the center is within any of the ROIs
            if not self._is_within_roi(cX, cY):
                continue

            detected_mosquitoes.append((cX, cY, area))

        return detected_mosquitoes


class MosquitoTracker:
    def __init__(self, ttl=20, feeding_frames=150, stationary_ttl=100, walking_distance=7):
        self.tracklets = {} 
        self.next_id = 0
        self.ttl = ttl
        self.feeding_frames = feeding_frames
        self.stationary_ttl = stationary_ttl
        self.stationary_memory = {}
        self.walking_distance = walking_distance
        self.log_metrics = {'dropped': 0, 'updated': 0, 'stationary': 0}  # Logging metrics

    def _distance(self, x1, y1, x2, y2):
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

    def _update_tracklet(self, x, y):
        closest_id = None
        min_distance = float('inf')
        distance_threshold = 5

        for id, tracklet in self.tracklets.items():
            distance = self._distance(x, y, tracklet['position'][0], tracklet['position'][1])
            if distance < min_distance and distance < distance_threshold:
                closest_id = id
                min_distance = distance

        if closest_id is not None:
            tracklet = self.tracklets[closest_id]
            prev_x, prev_y = tracklet['position']
            tracklet['position'] = (x, y)
            tracklet['frame_count'] += 1
            distance_moved = self._distance(x, y, prev_x, prev_y)

            if tracklet['state'] == 'feeding':
                tracklet['feeding_duration'] += 1
                tracklet['ttl'] = self.stationary_ttl
                tracklet['consecutive_missed'] = 0

                if distance_moved > self.walking_distance:
                    tracklet['state'] = 'moving'
                    tracklet['walking_duration'] = 1
            else:
                tracklet['walking_duration'] += 1
                tracklet['ttl'] = self.ttl
                tracklet['consecutive_missed'] = 0

                if distance_moved < self.walking_distance and tracklet['frame_count'] >= self.feeding_frames:
                    tracklet['state'] = 'feeding'
                    tracklet['feeding_duration'] = 1

            # Commented out because state updates are already handled above
            # if tracklet['frame_count'] >= self.feeding_frames:
            #     tracklet['state'] = 'feeding'
            # elif tracklet['walking_duration'] > self.feeding_frames:
            #     tracklet['state'] = 'moving'
        else:
            self.tracklets[self.next_id] = {
                'position': (x, y),
                'state': 'moving',
                'ttl': self.ttl,
                'frame_count': 0,
                'feeding_duration': 0,
                'walking_duration': 0,
                'consecutive_missed': 0
            }
            self.next_id += 1



    def _check_stationary_memory(self, detected_mosquitoes):
        new_detections = []
        for id, (x, y) in self.stationary_memory.items():
            for dx, dy, _ in detected_mosquitoes:
                distance = self._distance(x, y, dx, dy)
                if distance < 2:
                    new_detections.append((dx, dy))
                    self.log_metrics['stationary'] += 1  # Increment stationary metric

        return new_detections
    
    def track_mosquitoes(self, detected_mosquitoes):
        detected_ids = set()
        for x, y, _ in detected_mosquitoes:
            self._update_tracklet(x, y)
            detected_ids.add((x, y))
            pos = (x, y)
            self.stationary_memory[pos] = self.stationary_memory.get(pos, 0) + 1

        for pos in list(self.stationary_memory.keys()):
            if pos not in detected_mosquitoes:
                del self.stationary_memory[pos]

        additional_detections = self._check_stationary_memory(detected_mosquitoes)
        detected_mosquitoes.extend(additional_detections)

        ids_to_remove = []
        for id, tracklet in self.tracklets.items():
            if tracklet['position'] not in detected_ids:
                decrement = 2 if tracklet['state'] == 'feeding' else 4
                tracklet['ttl'] -= decrement
                tracklet['consecutive_missed'] += 1

            if tracklet['consecutive_missed'] > 15:
                tracklet['ttl'] = 0

            if tracklet['ttl'] <= 0:
                ids_to_remove.append(id)
                self.log_metrics['dropped'] += 1  # Increment dropped metric

        for id in ids_to_remove:
            if id in self.stationary_memory:
                del self.stationary_memory[id]
            del self.tracklets[id]

        return self.tracklets 





class DataAnalysis:
    def __init__(self, csv_file_path, rois):
        self.csv_file_path = csv_file_path
        self.rois = rois
        self.tracklet_data = {}  # {id: {'positions': [(x1, y1), (x2, y2), ...], 'state_changes': [(frame1, state1), ...]}}
        self.frame_count = 1
        self.total_mosquitoes = 0
        self.csv_headers = ["ID", "Total Time", "Distance Traveled", "Avg Speed", "Time Feeding", "Time Moving"]
        
    def log_data(self, frame_number, tracklets):
        self.frame_count += 1
        self.total_mosquitoes += len(tracklets)
        
        for id, tracklet in tracklets.items():
            if id not in self.tracklet_data:
                self.tracklet_data[id] = {'positions': [], 'state_changes': []}
                
            self.tracklet_data[id]['positions'].append(tracklet['position'])
            state_changes = self.tracklet_data[id]['state_changes']
            if not state_changes or state_changes[-1][1] != tracklet['state']:
                state_changes.append((frame_number, tracklet['state']))

    def calculate_metrics(self):
        metrics = []
        for id, data in self.tracklet_data.items():
            positions = data['positions']
            total_time = len(positions)
            
            distance = sum(self._distance(positions[i], positions[i+1]) for i in range(len(positions)-1))
            avg_speed = distance / total_time if total_time > 0 else 0
            
            time_feeding = sum(1 for _, state in data['state_changes'] if state == 'feeding')
            time_moving = total_time - time_feeding
            
            metrics.append([id, total_time, distance, avg_speed, time_feeding, time_moving])
        
        return metrics
    
    def _distance(self, p1, p2):
        return ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
    
    def export_data(self):
        with open(self.csv_file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(self.csv_headers)
            for row in self.calculate_metrics():
                writer.writerow(row)
                
    def summarize(self):
        metrics = self.calculate_metrics()
        if not metrics:
            return
        
        avg_time = mean(row[1] for row in metrics)
        avg_distance = mean(row[2] for row in metrics)
        avg_speed = mean(row[3] for row in metrics)
        avg_time_feeding = mean(row[4] for row in metrics)
        avg_time_moving = mean(row[5] for row in metrics)
        
        print(f"Summary:")
        print(f"Total Frames: {self.frame_count}")
        print(f"Total Mosquitoes: {self.total_mosquitoes}")
        print(f"Average Time: {avg_time}")
        print(f"Average Distance: {avg_distance}")
        print(f"Average Speed: {avg_speed}")
        print(f"Average Time Feeding: {avg_time_feeding}")
        print(f"Average Time Moving: {avg_time_moving}")



# class Visualization:
#     def __init__(self):
#         pass

#     def draw_circle(self, frame, center, radius, color, thickness=2):
#         cv2.circle(frame, center, radius, color, thickness)

#     def overlay_detected(self, frame, detected_mosquitoes, rois):
#         # Draw ROIs and detected mosquitoes
#         for cx, cy, r in rois:
#             self.draw_circle(frame, (cx, cy), r, (255, 255, 255))
#         for x, y, _ in detected_mosquitoes:
#             self.draw_circle(frame, (x, y), 5, (255, 51, 153))
            
#     def overlay_tracked(self, frame, tracklets, rois):
#         # Draw ROIs and tracked mosquitoes
#         for cx, cy, r in rois:
#             self.draw_circle(frame, (cx, cy), r, (255, 255, 255))
#         for id, tracklet in tracklets.items():
#             x, y = tracklet['position']
#             if tracklet['state'] == 'moving':
#                 self.draw_circle(frame, (x, y), 5, (0, 255, 0))
#             elif tracklet['state'] == 'feeding':
#                 self.draw_circle(frame, (x, y), 5, (0, 0, 255))


#     def display_side_by_side(self, original_frame, processed_frame):
#     # Convert processed frame to 3-channel image to match original_frame
#         processed_frame_colored = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)
        
#         # Concatenate frames horizontally
#         concatenated_frame = cv2.hconcat([original_frame, processed_frame_colored])
        
#         # Display the concatenated frame
#         cv2.imshow('Original vs Processed', concatenated_frame)
#         cv2.waitKey(1)

class Visualization:
    def __init__(self):
        pass

    def draw_circle(self, frame, center, radius, color, thickness=2):
        cv2.circle(frame, center, radius, color, thickness)

    def overlay_data(self, frame, detected_mosquitoes, rois):
        # Draw ROIs
        for cx, cy, r in rois:
            self.draw_circle(frame, (cx, cy), r, (255, 255, 255))

        # Draw detected but not tracked mosquitoes
        for x, y, _ in detected_mosquitoes:
            self.draw_circle(frame, (x, y), 5, (255, 51, 153))

        # Display the frame
        cv2.imshow('Mosquito Detection', frame)
        cv2.waitKey(1)



    def display_side_by_side(self, original_frame, processed_frame):
    # Convert processed frame to 3-channel image to match original_frame
        processed_frame_colored = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)
        
        # Concatenate frames horizontally
        concatenated_frame = cv2.hconcat([original_frame, processed_frame_colored])
        
        # Display the concatenated frame
        cv2.imshow('Original vs Processed', concatenated_frame)
        cv2.waitKey(1)



def calculate_mean_frame(video_reader, sample_size=8):
        frames = []
        for i in range(sample_size):
            frame = video_reader.read_frame()
            if frame is None:
                break
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray_frame)
        mean_frame = np.mean(frames, axis=0).astype(np.uint8)
        return mean_frame

def main(video_path, csv_file_path):
    # Initialize classes
    video_reader = VideoReader(video_path)
    roi_drawer = ROIDrawer()
    # data_logger = DataLogger(csv_file_path, roi_drawer.get_rois())
    visualization = Visualization()

    # Draw ROIs
    first_frame = video_reader.read_frame()
    roi_drawer.draw_rois(first_frame)
    rois = roi_drawer.get_rois()
    mosquito_detector = MosquitoDetector(rois)
    mosquito_tracker = MosquitoTracker()
    data_analysis = DataAnalysis(csv_file_path, rois)
    mean_frame = calculate_mean_frame(video_reader)
    preprocessor = Preprocessor(mean_frame)

    frame_number = 0
    while True:
        # Read and preprocess frame
        frame = video_reader.read_frame()
        if frame is None:
            break
        preprocessed_frame = preprocessor.preprocess(frame)

        
        # Detect and track mosquitoes
        total_detected_mosquitoes = 0
        detected_mosquitoes = mosquito_detector.detect_mosquitoes(preprocessed_frame)
        total_detected_mosquitoes += len(detected_mosquitoes)
        # logging.info(f"Number of detected mosquitoes in frame {frame_number}: {len(detected_mosquitoes)}")

        total_tracked_mosquitoes = 0
        tracklets = mosquito_tracker.track_mosquitoes(detected_mosquitoes)
        total_tracked_mosquitoes += len(tracklets)
        # logging.info(f"Number of tracked mosquitoes in frame {frame_number}: {len(tracklets)}")
        if frame_number % 300 == 0:  # X is the frequency at which you want to update
            preprocessor.update_mean_frame(frame, detected_mosquitoes)
        # Log data
        data_analysis.log_data(frame_number, tracklets)
        # logging.info(mosquito_tracker.__dict__)

        # frame_for_detected = frame.copy()
        # frame_for_tracked = frame.copy()
        
        # Overlay tracking and detection data on separate frames

       

        # Overlay tracking data on the frame
        visualization.overlay_data(frame, detected_mosquitoes, rois)
        visualization.display_side_by_side(frame, preprocessed_frame)

        cv2.imshow('Computer Vision vs Detected', frame)
        cv2.waitKey(1)
        
        frame_number += 1

    # Export tracking data to CSV
    # data_logger.export_data()
    data_analysis.export_data()
    data_analysis.summarize()
    # Release video resources
    video_reader.close()



if __name__ == "__main__":
    main("trim3.mov", "output.csv")



    



#machine learning approach
#make ROI for the Blood feeders
#DLC 
#bounding box detector on the data 
#pytorch 
#landing.ai 
# 

#elastic segmentation 
#shows the transformations to train the model 
#check if some of those transformations picks up on the organism across poses 

