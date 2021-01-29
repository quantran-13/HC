import  argparse
from src.reg import infer_reg
# from infer_reg import show_pred

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('IMAGE_PATH', type=str)
    parser.add_argument('--method', type=str, default='r', help="'r'=regression, 's'=segmentation")
    return parser.parse_args()

if __name__=="__main__":
    args = parse_args()
    if args.method == 'r':
        infer_reg.show_pred(args.IMAGE_PATH)
    elif args.method == 's':
        pass