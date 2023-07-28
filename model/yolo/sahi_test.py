#!/usr/bin/env python3
import argparse
import os
import platform
import sys

root_folder = os.path.abspath(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(root_folder)  # to enable import from parent directory

import subprocess
from tools import wandb_logger
from tools.dataset import (
    WITUAVDataset,
    HITUAVDatasetTrain,
    HITUAVDatasetVal,
    HITUAVDatasetTest,
)
from model import load_model
import sahi
from sahi.predict import get_sliced_prediction
from tqdm import tqdm
from parse_config import parse_data_config
from utils import print_dict, load_classes
import torch
import torchvision.transforms as transforms
from PIL import Image


def print_env() -> None:
    print("Environment information:")

    # Print OS information
    print(f"System: {platform.system()} {platform.release()}")

    # Print poetry package version
    try:
        print(
            f"Current Version: {subprocess.check_output(['poetry', 'version'], stderr=subprocess.DEVNULL).decode('ascii').strip()}"
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Not using the poetry package")

    # Print commit hash if possible
    try:
        print(
            f"Current Commit Hash: {subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], stderr=subprocess.DEVNULL).decode('ascii').strip()}"
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("No git or repo found")


def parse_arg() -> dict:
    parser = argparse.ArgumentParser(
        description="Slicing Aided Hyper Inference for YOLOv3 test script"
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="./model/yolo/yolov3-custom.cfg",
        help="Path to model definition file (.cfg)",
    )
    parser.add_argument(
        "-w",
        "--weights",
        type=str,
        default="./model/yolo/devansh_desktop_all_yolov3_ckpt_370.pth",
        help="Path to weights or checkpoint file (.weights or .pth)",
    )
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        default="dataset.cfg",
        help="Path to data config file (.data)",
    )
    parser.add_argument(
        "-b", "--batch_size", type=int, default=8, help="Size of each image batch"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Makes the validation more verbose"
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=416,
        help="Size of each image dimension for yolo",
    )
    parser.add_argument(
        "--n_cpu",
        type=int,
        default=8,
        help="Number of cpu threads to use during batch generation",
    )
    parser.add_argument(
        "--iou_thres",
        type=float,
        default=0.5,
        help="IOU threshold required to qualify as detected",
    )
    parser.add_argument(
        "--conf_thres", type=float, default=0.01, help="Object confidence threshold"
    )
    parser.add_argument(
        "--nms_thres",
        type=float,
        default=0.4,
        help="IOU threshold for non-maximum suppression",
    )
    return vars(parser.parse_args())


if __name__ == "__main__":
    print_env()
    config = parse_arg()
    print_dict(config)

    data_config = parse_data_config(config["data"])
    class_names = load_classes(data_config["names"])

    model = load_model(config["model"], config["weights"]).eval()
    model.set_thresholds(
        conf_thres=0.01, nms_thres=0.4, iou_thres=0.5
    )  #! NOTE: need to set manually because changing existing constructor is dangerous, remember to do this, otherwise sahi will not work, this is a compromise to SAHI implementation
    model.set_id2name_mapping(
        {str(id): name for id, name in enumerate(class_names)}
    )  #! NOTE: need to set manually because changing existing constructor is dangerous

    workers = 4  # number of workers for loading data in the DataLoader
    val_dataset = HITUAVDatasetTest("./", yolo=True)
    dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,  # batch_size is 1 because we're slicing the image
        shuffle=True,
        collate_fn=val_dataset.yolo_collate_fn,
        num_workers=workers,
        pin_memory=True,
    )  # note that we're passing the collate function here

    for batch_i, (imgs, targets) in enumerate(tqdm(dataloader, desc="Visualizing")):
        img = torch.squeeze(imgs[0])
        # NOTE: unnormalize the image since we need to pass it through SAHI framework
        invTrans = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.0, 0.0, 0.0], std=[1 / 0.229, 1 / 0.229, 1 / 0.229]
                ),
                transforms.Normalize(
                    mean=[-0.485, -0.485, -0.485], std=[1.0, 1.0, 1.0]
                ),
            ]
        )
        img = (torch.clamp(invTrans(img), 0, 1) * 255).to(torch.uint8)

        img = (
            img.permute(1, 2, 0).numpy().astype("uint8")
        )  # NOTE: change from (C, W, H) to (W, H, C) to pass through SAHI framework

        result = get_sliced_prediction(
            Image.fromarray(img, "RGB"),
            model,  #! NOTE: remember to turn on eval() mode otherwise strange error will occur
            slice_height=300,
            slice_width=300,
            overlap_height_ratio=0.5,
            overlap_width_ratio=0.5,
        )

        # yolov8n_model_path = "./model/yolo/yolov8n.pt"
        # model = sahi.AutoDetectionModel.from_pretrained(
        #     model_type="yolov8",
        #     model_path=yolov8n_model_path,
        #     confidence_threshold=0.3,
        #     device="cuda",
        # )
        # result = get_sliced_prediction(
        #     "/root/WIT-UAS-Dataset/sliced_prediction/small-vehicles1.jpg",
        #     model,
        #     slice_height=256,
        #     slice_width=256,
        #     overlap_height_ratio=0.2,
        #     overlap_width_ratio=0.2,
        # )

        result.export_visuals(export_dir="./sliced_prediction", file_name=str(batch_i))
