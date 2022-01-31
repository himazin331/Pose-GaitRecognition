import os
import argparse as arg

from getPose import getPose
from personIden import personIden

def runGaitRecog(frameA_path, frameB_path, outpath):
    gp = getPose()
    # Read video images
    frameA_list = gp.dataPrep(frameA_path)
    frameB_list = gp.dataPrep(frameB_path)

    # Person detection / Pose Estimation
    gp.run(frameA_list, os.path.join(outpath, "A"))
    gp.run(frameB_list, os.path.join(outpath, "B"))

    # Person Identication
    #personiden = personIden()
    #personiden.run()

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
    print("# Video1 frame images path: {}".format(args.video_a))
    print("# Video2 frame images path: {}".format(args.video_b))
    print("# Save location: {}".format(args.out_path))
    print("===========================\n")

    err = runGaitRecog(args.video_a, args.video_b, args.out_path)
    if err:
        print("error")

if __name__ == "__main__":
    main()