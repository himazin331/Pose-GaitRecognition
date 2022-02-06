import mxnet
from gluoncv import data, model_zoo, utils
from gluoncv.data.transforms.pose import detector_to_simple_pose, heatmap_to_coord

import os
import argparse as arg

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

        self.ob_ba_num = 1
        self.ob_total_num = 9

        self.ob_data_deque = deque([])
        self.ob_avg_xydata = np.zeros((17, 2))

        self.under_threshold = 0.0
        self.over_threshold = 0.0

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

    def dataCorrect(self):
        pass

    # Saving the results
    def saveResult(self, data, i, outpath):
        # Axis deletion
        
        plt.axis('off')
        plt.style.use('dark_background')
        data = data.get_figure()
        data.subplots_adjust(left=0, right=1, bottom=0, top=1) # Margin removal
        data.savefig(os.path.join(outpath, str(i)+".png"), bbox_inches='tight', pad_inches=0)
        plt.close()

    def run(self, frame_list, outpath):
        os.makedirs(outpath, exist_ok=True)
        
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

            self.ob_data_deque.append(results_data)

            if len(self.ob_data_deque) == 1:
                result = utils.viz.plot_keypoints(bg, *results_data, box_thresh=1, keypoint_thresh=0)
                self.saveResult(result, i+1, outpath) # Saving the results

            if len(self.ob_data_deque) == self.ob_ba_num*2+1:
                
                for j in range(self.ob_ba_num*2+1):
                    if j == self.ob_ba_num:
                        continue
                    self.ob_avg_xydata += self.ob_data_deque[j][0][0].asnumpy()
                self.ob_avg_xydata /= self.ob_ba_num*2

                std = np.sqrt(((self.ob_avg_xydata-(self.ob_avg_xydata/(self.ob_ba_num*2)))**2.0)/2.0)

                self.under_threshold = self.ob_avg_xydata - std
                self.over_threshold = self.ob_avg_xydata + std


                for k, v in enumerate(self.ob_data_deque[self.ob_ba_num][0][0].asnumpy()):
                    x, y = v
                    if x < self.under_threshold[k][0] or x > self.over_threshold[k][0]:
                        self.ob_data_deque[self.ob_ba_num][0][0][k][0] = self.ob_avg_xydata[k][0]

                    if y < self.under_threshold[k][1] or y > self.over_threshold[k][1]:
                        self.ob_data_deque[self.ob_ba_num][0][0][k][1] = self.ob_avg_xydata[k][1]

                result = utils.viz.plot_keypoints(bg, *self.ob_data_deque[self.ob_ba_num], box_thresh=1, keypoint_thresh=0)
                self.saveResult(result, ob_cnt, outpath) # Saving the results

                if len(self.ob_data_deque) == self.ob_total_num:
                    self.ob_data_deque.popleft()
                elif len(self.ob_data_deque) < self.ob_total_num:
                    self.ob_ba_num += 1

                ob_cnt += 1

        print("Done")
