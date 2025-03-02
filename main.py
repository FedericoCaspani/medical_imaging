import argparse
import os
from src.object_detection.prepair_roi_det_dataset import prepare_roi_det_dataset
from src.object_detection.train_yolox import train_yolox

def parse_args():
    parser = argparse.ArgumentParser('Entrypoint of the project')
    parser.add_argument('--mode',
                        type=str,
                        default='object_detection',
                        choices=['object_detection', 'segmentation', 'classification'],
                        help='Choose the mode: "object_detection", "segmentation", or "classification".')
    parser.add_argument('--dataset',
                        type=str,
                        default='breast_cancer',
                        choices=['breast_cancer'],
                        help='Choose the dataset: "breast_cancer".')
    args = parser.parse_args()
    return args

def object_detection():
    prepare_roi_det_dataset()
    train_yolox()

def main(args):
    if args.mode == 'object_detection':
        object_detection()

if __name__ == '__main__':
    args = parse_args()
    main(args)