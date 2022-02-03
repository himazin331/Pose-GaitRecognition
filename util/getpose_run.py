import os
import argparse as arg

from getPose import getPose

def runPoseEst(frame_dir, outpath_p):
    gp = getPose()

    for f in os.listdir(frame_dir):
        target_path = os.path.join(frame_dir, f)

        # Read frame images
        print("Reading frame images.")
        frame_list = gp.dataPrep(target_path)

        # Person detection / Pose Estimation
        print("Start pose estimation.")
        outpath = os.path.join(outpath_p, f+"_pose")
        gp.run(frame_list, outpath)
    return False

def main():
    # Command line options
    parser = arg.ArgumentParser(description="Getting Pose-Estimation result")
    parser.add_argument('--frame_dir', '-d', type=str, default=None,
                        help='Specify the directory containing the frame image of the video. (Required)')
    parser.add_argument('--out_path', '-o', type=str, default=os.path.dirname(os.path.abspath(__file__)),
                        help='Where to save the results. (Default: ./<video_name>_pose)')
    args = parser.parse_args()

    # The target is not specified -> Exception
    if args.frame_dir is None:
        raise Exception("The target is not specified.")

    # Setting information
    print("=== Setting information ===")
    print("# Target direcotry path: {}".format(args.frame_dir))
    print("# Save location: {}/<video_name>_pose".format(args.out_path))
    print("===========================\n")

    err = runPoseEst(args.frame_dir, args.out_path)
    if err:
        raise Exception(err)
    print("Complete.\n")

if __name__ == "__main__":
    main()