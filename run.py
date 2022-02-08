import os
import argparse as arg

from personIden import personIden

from natsort import natsorted

import cv2
import tensorflow as tf

def read_joint_images(dir_path):
    joint_list = []
    for f in natsorted(os.listdir(dir_path)):
        f = cv2.imread(os.path.join(dir_path, f))
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        f = tf.image.resize_with_crop_or_pad(f, 480, 300)
        f = tf.cast(f, tf.float32) / 255.0
        joint_list.append(f)
    return joint_list

def runGaitRecog(jointsA_path, jointsB_path, jointsFalse_path):
    print("Start person identification.")
    # Read joint images
    jointsA_list = read_joint_images(jointsA_path)
    jointsB_list = read_joint_images(jointsB_path)
    jointsFalse_list = read_joint_images(jointsFalse_path)

    # Person Identication
    input_shape = jointsA_list[0].shape
    personiden = personIden((None, input_shape[0], input_shape[1], 3))
    personiden.run(jointsA_list, jointsB_list, jointsFalse_list)

def main():
    # Command line options
    parser = arg.ArgumentParser(description="Gait Recognition with Pose-Estimation model")
    parser.add_argument('--joint_a', '-ja', type=str, default=None,
                        help='Directory containing the joint images of person 1. (Required)')
    parser.add_argument('--joint_b', '-jb', type=str, default=None,
                        help='Directory containing the joint images of person 2. (Required)')
    parser.add_argument('--joint_f', '-jf', type=str, default=None,
                        help='Directory containing the joint images of false. (Required)')
    args = parser.parse_args()

    # Setting information
    print("=== Setting information ===")
    print("# Person1 joint images path: {}".format(args.joint_a))
    print("# Person2 joint images path: {}".format(args.joint_b))
    print("# False joint images path: {}".format(args.joint_f))
    print("===========================\n")

    runGaitRecog(args.joint_a, args.joint_b, args.joint_f)

if __name__ == "__main__":
    main()