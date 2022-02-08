import mxnet
from gluoncv import data, model_zoo, utils
from gluoncv.data.transforms.pose import detector_to_simple_pose, heatmap_to_coord

import os
from natsort import natsorted
import numpy as np
import matplotlib.pyplot as plt

from collections import deque

class PersonDetection():
    def __init__(self, model_name):
        # Person detector
        self.detector = model_zoo.get_model(model_name, pretrained=True)
        self.detector.reset_class(["person"], reuse_weights=['person']) # Target: person
    
    def detection(self, data):
        return self.detector(data)

class PoseEstimation():
    def __init__(self, model_name):
        self.pose_est = model_zoo.get_model(model_name, pretrained=True)
    
    def getpose(self, image, class_IDs, scores, bounding_boxs):
        pose_input, upscale_bbox = detector_to_simple_pose(image, class_IDs, scores, bounding_boxs)
        if len(upscale_bbox) == 0:
            return (None, None)

        predicted_heatmap = self.pose_est(pose_input)
        return heatmap_to_coord(predicted_heatmap, upscale_bbox)

class getPose():
    def __init__(self):
        personDct_model = 'yolo3_mobilenet1.0_coco' # Person Detection Model
        self.personDct = PersonDetection(personDct_model)
        poseEst_model = 'simple_pose_resnet152_v1d' # Pose Est Estimation Model
        self.poseEst = PoseEstimation(poseEst_model)

        self.window_ba_len = 1      # Window-length before and after the target
        self.window_entire_len = 9  # Entire window-length

        self.ob_data_queue = deque([])  # Observation data queue
        self.ob_avg_xydata = np.zeros((17, 2))  # Average value of x,y coordinates

        self.lower_threshold = 0.0  # Lower threshold
        self.upper_threshold = 0.0   # Upper threshold
        # The x and y coordinates must be within this threshold interval.

    # Data preprocessing
    def dataPrep(self, framedir_path):
        print("target: {} ...".format(framedir_path), end="")
        frame_list = []
        for frame_name in natsorted(os.listdir(framedir_path)):
            # Image loading and normalization
            # Output : (x, frame)
            # x : For network input (MXNet n-D array)
            # frame : for display (Numpy n-D array)
            x, frame = data.transforms.presets.ssd.load_test(os.path.join(framedir_path, frame_name), short=512)
            frame_list.append((x, frame))
        print("Done")

        return frame_list

    #! [incomplete]
    #! If the future data is abnormal data to be corrected, no correction effect or change normal to abnormal.
    def dataCorrect(self):
        # Calculate the average value of the coordinates
        for j in range(self.window_ba_len*2+1):
            if j == self.window_ba_len: # The correction target is not included.
                continue
            self.ob_avg_xydata += self.ob_data_queue[j][0][0].asnumpy()
        self.ob_avg_xydata /= self.window_ba_len*2

        std = np.sqrt(((self.ob_avg_xydata-(self.ob_avg_xydata/(self.window_ba_len*2)))**2.0)/2.0) # Standard deviation
        self.lower_threshold = self.ob_avg_xydata - std
        self.upper_threshold = self.ob_avg_xydata + std

        # Correction process
        for k, v in enumerate(self.ob_data_queue[self.window_ba_len][0][0].asnumpy()):
            x, y = v
            if x < self.lower_threshold[k][0] or x > self.upper_threshold[k][0]:
                self.ob_data_queue[self.window_ba_len][0][0][k][0] = self.ob_avg_xydata[k][0]
            if y < self.lower_threshold[k][1] or y > self.upper_threshold[k][1]:
                self.ob_data_queue[self.window_ba_len][0][0][k][1] = self.ob_avg_xydata[k][1]

    # Saving the results
    def saveResult(self, data, i, outpath):
        plt.axis('off') # Axis deletion
        plt.style.use('dark_background') # Black background
        data = data.get_figure()
        data.subplots_adjust(left=0, right=1, bottom=0, top=1) # Margin removal
        data.savefig(os.path.join(outpath, str(i)+".png"), bbox_inches='tight', pad_inches=0)
        plt.close()

    def run(self, frame_list, outpath):
        os.makedirs(outpath, exist_ok=True) # Create a save location
        ob_cnt = 2

        for i, data in enumerate(frame_list):
            print("\r{0} / {1} frames ...".format(i+1, len(frame_list)), end="")
            x, frame = data
    
            # Person detection
            class_IDs, scores, bounding_boxs = self.personDct.detection(x)

            # Crop input image
            x = np.floor(bounding_boxs[0][0][0].asscalar()).astype(np.int32)
            xw = np.floor(bounding_boxs[0][0][2].asscalar()).astype(np.int32)
            y = np.floor(bounding_boxs[0][0][1].asscalar()).astype(np.int32)
            yh = np.floor(bounding_boxs[0][0][3].asscalar()).astype(np.int32)
            frame = frame[y:yh, x:xw]
            bg = np.zeros(frame.shape) # background

            # Start at bbox coordinate (0,0)
            bounding_boxs[0][0][2] -= bounding_boxs[0][0][0] # x+w
            bounding_boxs[0][0][3] -= bounding_boxs[0][0][1] # y+h
            bounding_boxs[0][0][0] = 0.0 # x
            bounding_boxs[0][0][1] = 0.0 # y

            # Pose Estimation
            pred_coords, confidence = self.poseEst.getpose(frame, class_IDs, scores, bounding_boxs)
            # If no joints are detected.
            if pred_coords is None:
                continue

            results_data = [pred_coords, confidence, class_IDs, bounding_boxs, scores]

            self.ob_data_queue.append(results_data) # Observation data queue
            if len(self.ob_data_queue) == 1: # The first data will be output.
                # Saving the results
                result = utils.viz.plot_keypoints(bg, *results_data, box_thresh=1, keypoint_thresh=0)
                self.saveResult(result, i+1, outpath)
                continue

            result = utils.viz.plot_keypoints(bg, *results_data, box_thresh=1, keypoint_thresh=0)
            self.saveResult(result, i+1, outpath)

            if len(self.ob_data_queue) == self.window_ba_len*2+1:
                self.dataCorrect() # Data correction

                # Saving the results
                result = utils.viz.plot_keypoints(bg, *self.ob_data_queue[self.window_ba_len], box_thresh=1, keypoint_thresh=0)
                self.saveResult(result, ob_cnt, outpath)

                # 9 == 9 : dequeue
                if len(self.ob_data_queue) == self.window_entire_len:
                    self.ob_data_queue.popleft()
                elif len(self.ob_data_queue) < self.window_entire_len:
                    self.window_ba_len += 1
                ob_cnt += 1
        print("Done")
