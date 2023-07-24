#!/usr/bin/env python3
from __future__ import division

import argparse
import os
import sys

root_folder = os.path.abspath(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(root_folder)

import albumentations as A
import torch
import torch.optim as optim
import tqdm
from albumentations.pytorch import ToTensorV2
from terminaltables import AsciiTable
from torch.utils.data import DataLoader
from torchsummary import summary

from tools import wandb_logger
from tools.dataset import CombinedDataset
from tools.dataset import HITUAVDatasetTrain
from tools.dataset import HITUAVDatasetVal
from tools.dataset import WITUAVDataset
from tools.logger import Logger
from loss import compute_loss
from model import load_model
from parse_config import parse_data_config
from test import _evaluate
from utils import load_classes
from utils import print_environment_info
from utils import provide_determinism
from utils import to_cpu
from utils import worker_seed_set


def run():
    print_environment_info()
    parser = argparse.ArgumentParser(description="Trains the YOLO model.")
    parser.add_argument(
        "-n", "--name", type=str, default="unnamed", help="Name of experiment"
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="./model/yolo/yolov3-custom.cfg",
        help="Path to model definition file (.cfg)",
    )
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        default="all",
        help="dataset to use, can be all/hit/wit",
    )
    parser.add_argument(
        "--wit-sensor",
        type=str,
        default="both",
        help="set to flir/seek/both to configure sensors in wit, applies to both train and val",
    )
    parser.add_argument(
        "-e", "--epochs", type=int, default=900, help="Number of epochs"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        default=False,
        action="store_true",
        help="Makes the training more verbose",
    )
    parser.add_argument(
        "--n-cpu",
        type=int,
        default=12,
        help="Number of cpu threads to use during batch generation",
    )
    parser.add_argument(
        "--pretrained-weights",
        type=str,
        help="Path to checkpoint file (.weights or .pth). Starts training from checkpoint model",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10,
        help="Interval of epochs between saving model weights",
    )
    parser.add_argument(
        "--evaluation-interval",
        type=int,
        default=1,
        help="Interval of epochs between evaluations on validation set",
    )
    parser.add_argument(
        "--multiscale-training", action="store_true", help="Allow multi-scale training"
    )
    parser.add_argument(
        "--iou-thres",
        type=float,
        default=0.5,
        help="Evaluation: IOU threshold required to qualify as detected",
    )
    parser.add_argument(
        "--conf-thres",
        type=float,
        default=0.1,
        help="Evaluation: Object confidence threshold",
    )
    parser.add_argument(
        "--nms-thres",
        type=float,
        default=0.5,
        help="Evaluation: IOU threshold for non-maximum suppression",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default="logs",
        help="Directory for training log files (e.g. for TensorBoard)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="Makes results reproducable. Set -1 to disable.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="set batch size of training, depends on your GPU memory capacity",
    )
    args = parser.parse_args()
    print(f"Command line arguments: {args}")

    if args.seed != -1:
        provide_determinism(args.seed)

    logger = Logger(args.logdir)  # Tensorboard logger

    # Create output directories if missing
    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Get data configuration
    data_config = parse_data_config("dataset.cfg")
    # train_path = data_config["train"]
    # valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ############
    # Create model
    # ############

    model = load_model(args.model, args.pretrained_weights)

    wandb_logger.init(name=args.name, config=args)  # batch-size, lr, epochs, dataset
    wandb_logger.set_config(
        config={
            "model architecture": "YOLO",
            "learning rate": model.hyperparams["learning_rate"],
        }
    )  # learning rate

    # Print model
    if args.verbose:
        summary(
            model,
            input_size=(3, model.hyperparams["height"], model.hyperparams["height"]),
        )

    mini_batch_size = model.hyperparams["batch"] // model.hyperparams["subdivisions"]

    image_transform = A.Compose(
        [
            A.RandomRotate90(),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ColorJitter(p=0.5),
            A.CLAHE(clip_limit=(1, 4), p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            A.GridDistortion(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.RandomGamma(p=0.5),
            A.HueSaturationValue(p=0.5),
            A.RandomShadow(
                p=0.5, num_shadows_lower=1, num_shadows_upper=3, shadow_dimension=3
            ),
            A.RandomResizedCrop(
                height=300,
                width=300,
                scale=(0.8, 1.0),
                ratio=(0.75, 1.3333333333333333),
                p=1.0,
            ),
            A.Perspective(),
        ],
        # HIT dataset is in unnormalized YOLO format, which is x_center, y_center, width, height.
        bbox_params=A.BboxParams(format="yolo", label_fields=["labels"]),
    )

    # #################
    # Create Dataloader
    # #################

    # # Load training dataloader
    # dataloader = _create_data_loader(
    #     train_path,
    #     mini_batch_size,
    #     model.hyperparams['height'],
    #     args.n_cpu,
    #     args.multiscale_training)

    # # Load validation dataloader
    # validation_dataloader = _create_validation_data_loader(
    #     valid_path,
    #     mini_batch_size,
    #     model.hyperparams['height'],
    #     args.n_cpu)
    if args.data == "hit":
        train_dataset = HITUAVDatasetTrain(
            root="./", yolo=True, image_transform=image_transform
        )
        val_dataset = HITUAVDatasetVal(root="./", yolo=True)
    elif args.data == "wit":
        train_dataset = WITUAVDataset(
            root="./WIT-UAV-Dataset_split/train/",
            sensor=args.wit_sensor,
            yolo=True,
            image_transform=image_transform,
        )
        val_dataset = WITUAVDataset(
            root="./WIT-UAV-Dataset_split/val/", sensor=args.wit_sensor, yolo=True
        )
    elif args.data == "all":
        train_dataset = CombinedDataset(
            [
                HITUAVDatasetTrain(
                    root="./", yolo=True, image_transform=image_transform
                ),
                WITUAVDataset(
                    root="./WIT-UAV-Dataset_split/train/",
                    sensor=args.wit_sensor,
                    yolo=True,
                    image_transform=image_transform,
                ),
            ]
        )
        val_dataset = CombinedDataset(
            [
                HITUAVDatasetVal(root="./", yolo=True),
                WITUAVDataset(
                    root="./WIT-UAV-Dataset_split/val/",
                    sensor=args.wit_sensor,
                    yolo=True,
                ),
            ]
        )

    batch_size = args.batch_size  # batch size
    workers = args.n_cpu  # number of workers for loading data in the DataLoader
    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_dataset.yolo_collate_fn,
        num_workers=workers,
        pin_memory=True,
    )  # note that we're passing the collate function here
    validation_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=val_dataset.yolo_collate_fn,
        num_workers=2,  # for some reason using less workers here makes faster validation
        pin_memory=True,
    )  # note that we're passing the collate function here

    # ################
    # Create optimizer
    # ################

    params = [p for p in model.parameters() if p.requires_grad]

    if model.hyperparams["optimizer"] in [None, "adam"]:
        optimizer = optim.Adam(
            params,
            lr=model.hyperparams["learning_rate"],
            weight_decay=model.hyperparams["decay"],
        )
    elif model.hyperparams["optimizer"] == "sgd":
        optimizer = optim.SGD(
            params,
            lr=model.hyperparams["learning_rate"],
            weight_decay=model.hyperparams["decay"],
            momentum=model.hyperparams["momentum"],
        )
    else:
        print("Unknown optimizer. Please choose between (adam, sgd).")

    # skip epoch zero, because then the calculations for when to evaluate/checkpoint makes more intuitive sense
    # e.g. when you stop after 30 epochs and evaluate every 10 epochs then the evaluations happen after: 10,20,30
    # instead of: 0, 10, 20
    for epoch in range(1, args.epochs + 1):
        print("\n---- Training Model ----")

        model.train()  # Set model to training mode

        metrics = {
            "losses": [],
            "IoU_losses": [],
            "object_losses": [],
            "class_losses": [],
        }

        for batch_i, (imgs, targets) in enumerate(
            tqdm.tqdm(dataloader, desc=f"Training Epoch {epoch}")
        ):
            batches_done = len(dataloader) * epoch + batch_i

            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device)

            outputs = model(imgs)

            loss, loss_components = compute_loss(outputs, targets, model)

            loss.backward()

            ###############
            # Run optimizer
            ###############

            if batches_done % model.hyperparams["subdivisions"] == 0:
                # Adapt learning rate
                # Get learning rate defined in cfg
                lr = model.hyperparams["learning_rate"]
                if batches_done < model.hyperparams["burn_in"]:
                    # Burn in
                    lr *= batches_done / model.hyperparams["burn_in"]
                else:
                    # Set and parse the learning rate to the steps defined in the cfg
                    for threshold, value in model.hyperparams["lr_steps"]:
                        if batches_done > threshold:
                            lr *= value
                # Log the learning rate
                logger.scalar_summary("train/learning_rate", lr, batches_done)
                # Set learning rate
                for g in optimizer.param_groups:
                    g["lr"] = lr

                # Run optimizer
                optimizer.step()
                # Reset gradients
                optimizer.zero_grad()

            # ############
            # Log progress
            # ############
            if args.verbose:
                print(
                    AsciiTable(
                        [
                            ["Type", "Value"],
                            ["IoU loss", float(loss_components[0])],
                            ["Object loss", float(loss_components[1])],
                            ["Class loss", float(loss_components[2])],
                            ["Loss", float(loss_components[3])],
                            ["Batch loss", to_cpu(loss).item()],
                        ]
                    ).table
                )

            # Tensorboard logging
            tensorboard_log = [
                ("train/iou_loss", float(loss_components[0])),
                ("train/obj_loss", float(loss_components[1])),
                ("train/class_loss", float(loss_components[2])),
                ("train/loss", to_cpu(loss).item()),
            ]
            logger.list_of_scalars_summary(tensorboard_log, batches_done)

            model.seen += imgs.size(0)

            metrics["losses"].append(loss)
            metrics["IoU_losses"].append(loss_components[0])
            metrics["object_losses"].append(loss_components[1])
            metrics["class_losses"].append(loss_components[2])

        wandb_logger.log(
            {
                "train/loss": sum(metrics["losses"]) / len(metrics["losses"]),
                "train/IoU_loss": sum(metrics["IoU_losses"])
                / len(metrics["IoU_losses"]),
                "train/object_loss": sum(metrics["object_losses"])
                / len(metrics["object_losses"]),
                "train/class_loss": sum(metrics["class_losses"])
                / len(metrics["class_losses"]),
                "epoch": epoch,
            }
        )

        # #############
        # Save progress
        # #############

        # Save model to checkpoint file
        if epoch % args.checkpoint_interval == 0:
            checkpoint_path = f"yolov3_ckpt_{epoch}.pth"
            checkpoint_path = os.path.join(logger.log_dir, checkpoint_path)
            print(f"---- Saving checkpoint to: '{checkpoint_path}' ----")
            state = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(state, checkpoint_path)

        # ########
        # Evaluate
        # ########

        if epoch % args.evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            metrics_output = _evaluate(
                model,
                validation_dataloader,
                class_names,
                img_size=model.hyperparams["height"],
                iou_thres=args.iou_thres,
                conf_thres=args.conf_thres,
                nms_thres=args.nms_thres,
                verbose=args.verbose,
                epoch=epoch,
            )

            if metrics_output is not None:
                precision, recall, AP, f1, ap_class = metrics_output
                evaluation_metrics = [
                    ("validation/precision", precision.mean()),
                    ("validation/recall", recall.mean()),
                    ("validation/mAP", AP.mean()),
                    ("validation/f1", f1.mean()),
                ]
                logger.list_of_scalars_summary(evaluation_metrics, epoch)

                wandb_logger.log(
                    {
                        "eval/precision": precision.mean(),
                        "eval/recall": recall.mean(),
                        "eval/mAP": AP.mean(),
                        "eval/f1": f1.mean(),
                        "epoch": epoch,
                    }
                )


if __name__ == "__main__":
    run()
