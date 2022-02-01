import os
import cv2

import argparse as arg

# Convert video to image
def convert(video_dir, outpath_p):
    for f in os.listdir(video_dir):
        target_path = os.path.join(video_dir, f)
        print("Conversion of video. target: {} ...".format(target_path), end="")

        video = cv2.VideoCapture(target_path) # Load the video
        if video.isOpened():
            # Create a frame directory
            outpath = os.path.join(outpath_p, f[:-4])
            os.makedirs(outpath, exist_ok=True)

            num = 1
            while True:
                ret, frame = video.read() # Get the frame
                if ret:
                    cv2.imwrite(os.path.join(outpath, str(num)+"_frame.jpg"), frame) # Save the frame
                    print("\rConversion of video. target: {} ...{}".format(f, num), end="")
                    num += 1
                else:
                    break
        else:
            raise Exception("Failed to load the video.")
        video.release()
        print("\rConversion of video. target: {} ...Done!".format(f))

def main():
    # Command line options
    parser = arg.ArgumentParser(description="Video to Image Conversion for Gait Recognition")
    parser.add_argument('--video_dir', '-d', type=str, default=None,
                        help='Specify the directory containing the video. (Required)')
    parser.add_argument('--out_path', '-o', type=str,
                        default=os.path.dirname(os.path.abspath(__file__)),
                        help='Where to save the results. (Default: ./<video_name>)')
    args = parser.parse_args()

    # The target is not specified -> Exception
    if args.video_dir is None:
        raise Exception("The target is not specified.")

    # Setting information
    print("=== Setting information ===")
    print("# Target directory path: {}".format(args.video_dir))
    print("# Save location: {}/<video_name>".format(args.out_path))
    print("===========================\n")

    print("Start conversion")
    convert(args.video_dir, args.out_path)
    print("Successfully!")

if __name__ == "__main__":
    main()