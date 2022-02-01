import os
import argparse as arg

from getPose import getPose
from personIden import personIden

from natsort import natsorted

import cv2

def read_file(dir_path):
    file_list = []
    for f in natsorted(os.listdir(dir_path)):
        f = cv2.imread(os.path.join(dir_path, f))
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        file_list.append(f)
    return file_list

def runGaitRecog(frameA_path, frameB_path, outpath):
    gp = getPose()

    # Read video images
    frameA_list = gp.dataPrep(frameA_path)
    frameB_list = gp.dataPrep(frameB_path)

    # Person detection / Pose Estimation
    jointsA_path = os.path.join(outpath, "A")
    jointsB_path = os.path.join(outpath, "B")
    gp.run(frameA_list, jointsA_path)
    gp.run(frameB_list, jointsB_path)

    # Read joint images
    jointsA_list = read_file(jointsA_path)
    jointsB_list = read_file(jointsB_path)

    # Person Identication
    input_shape = jointsA_list[0].shape
    personiden = personIden((None, input_shape[0], input_shape[1], input_shape[2]))
    personiden.run(jointsA_list, jointsB_list)

    return False

def main():
    # Command line options
    parser = arg.ArgumentParser(description="Gait Recognition with Pose-Estimation model")
    parser.add_argument('--frame_a', '-fa', type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "A_frame"),
                        help='Directory containing the frame images of video 1. (Required)')
    parser.add_argument('--frame_b', '-fb', type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "B_frame"),
                        help='Directory containing the frame images of video 2. (Required)')
    parser.add_argument('--out_path', '-o', type=str, default=os.path.dirname(os.path.abspath(__file__)),
                        help='Where to save the results. (Default: ./gait_result_xxxxxxxxxxxxxx.h5)')
    args = parser.parse_args()

    # Setting information
    print("=== Setting information ===")
    print("# Video1 frame images path: {}".format(args.frame_a))
    print("# Video2 frame images path: {}".format(args.frame_b))
    print("# Save location: {}".format(args.out_path))
    print("===========================\n")

    err = runGaitRecog(args.frame_a, args.frame_b, args.out_path)
    if err:
        print("error")

if __name__ == "__main__":
    main()