import sys
sys.path.append("./src")
import  argparse
from reg import infer_reg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('IMAGE_PATH', type=str)
    parser.add_argument('--method', type=str, default='r',
                        help="'r'=regression, 's'=segmentation")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.method == 'r':
        infer_reg.show_pred(args.IMAGE_PATH, "./models/sequential_mse=1242.16_Adam_ep290.hdf5")
    elif args.method == 's':
        pass
