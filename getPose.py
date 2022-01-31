import mxnet
from gluoncv import data, model_zoo, utils
from gluoncv.data.transforms.pose import detector_to_simple_pose, heatmap_to_coord

import os
from natsort import natsorted
import numpy as np
import matplotlib.pyplot as plt

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
        if pose_input is None:
            return (None, None)

        predicted_heatmap = self.pose_est(pose_input)
        return heatmap_to_coord(predicted_heatmap, upscale_bbox)

class getPose():
    def __init__(self):
        personDct_model = 'yolo3_mobilenet1.0_coco' # Person Detection Model
        self.personDct = PersonDetection(personDct_model)
        poseEst_model = 'simple_pose_resnet152_v1d' # Pose Est Estimation Model
        self.poseEst = PoseEstimation(poseEst_model)

    # Data preprocessing
    def dataPrep(self, framedir_path):
        frame_list = []
        for frame_name in natsorted(os.listdir(framedir_path)):
            # Image loading and normalization
            # Output : (x, frame)
            # x : For network input (MXNet n-D array)
            # frame : for display (Numpy n-D array)
            x, frame = data.transforms.presets.ssd.load_test(os.path.join(framedir_path, frame_name), short=512)
            frame_list.append((x, frame))

        return frame_list

    def run(self, frame_list, outpath):
        os.makedirs(outpath, exist_ok=True)

        for i, data in enumerate(frame_list):
            print("{0} / {1} frames".format(i+1, len(frame_list)), end="\n")
            x, frame = data

            # Person detection
            class_IDs, scores, bounding_boxs = self.personDct.detection(x)
            # Pose Estimation
            pred_coords, confidence = self.poseEst.getpose(frame, class_IDs, scores, bounding_boxs)
            if pred_coords is None:
                continue

            bg = np.zeros(frame.shape)
            """             
            plt.rcParams['figure.figsize'] = (15.0, 15.0)
            real_result = utils.viz.plot_keypoints(frame, pred_coords, confidence,
                                        class_IDs, bounding_boxs, scores,
                                        box_thresh=0.5, keypoint_thresh=0.2)
            plt.show()
            """

            _bg_result = utils.viz.plot_keypoints(bg, pred_coords, confidence,
                                        class_IDs, bounding_boxs, scores,
                                        box_thresh=0.5, keypoint_thresh=0.2)
            bg_result = _bg_result.get_figure()
            bg_result.savefig(os.path.join(outpath, str(i)+".png"))