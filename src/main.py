import argparse
import scipy.misc
from edge_detector import EdgeDetector

def main():
    parser = argparse.ArgumentParser("Perform Canny edge detection")
    parser.add_argument("in_image", type=str, help="input image file")
    parser.add_argument("out_image", type=str, help="output image file")
    parser.add_argument("tu",
                        nargs='?',
                        type=float,
                        default=None,
                        help="upper threshold")
    parser.add_argument("tl",
                        nargs='?',
                        type=float,
                        default=None,
                        help="upper threshold")
    args = parser.parse_args()
    img = scipy.misc.imread(args.in_image, flatten=True)
    detector = EdgeDetector(img)
    edges = detector.detect_edges(args.tu, args.tl)
    scipy.misc.imsave(args.out_image, edges)

if __name__ == "__main__":
    main()

