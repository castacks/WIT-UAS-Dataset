#!/usr/bin/env python3
import argparse
import copy
import os
import sys

root_folder = os.path.abspath(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(root_folder)  # to enable import from parent directory

import time
from pprint import PrettyPrinter
import albumentations as A
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
import torch.utils.data

# from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from tools import wandb_logger
from tools.dataset import CombinedDataset
from tools.dataset import HITUAVDatasetTrain
from tools.dataset import HITUAVDatasetVal
from tools.dataset import WITUAVDataset
from tools.logger import Logger
from model import MultiBoxLoss
from model import SSD300
from utils import *

cudnn.benchmark = True


pp = PrettyPrinter()

# Data parameters
data_folder = "./"  # folder with data files
keep_difficult = True  # use objects considered difficult to detect?
logdir = "ssd_logs"
logger = Logger(logdir)  # Tensorboard logger

# Model parameters
# Not too many here since the SSD300 has a very specific structure
n_classes = len(label_map)  # number of different types of objects
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Learning parameters
print_freq = 200  # print training status every __ batches
lr = 5e-4  # learning rate
decay_lr_at = [80000, 100000]  # decay learning rate after these many iterations
decay_lr_to = 0.1  # decay learning rate to this fraction of the existing learning rate
momentum = 0.9  # momentum
weight_decay = 5e-4  # weight decay
grad_clip = None  # clip if gradients are exploding, which may happen at larger batch sizes (sometimes at 32) - you will recognize it by a sorting error in the MuliBox loss calculation


cudnn.benchmark = True


def main():
    """
    Training.
    """
    global start_epoch, label_map, epoch, checkpoint, decay_lr_at

    # Initialize model or load checkpoint
    start_epoch = 0
    model = SSD300(n_classes=n_classes)
    # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
    biases = list()
    not_biases = list()
    for param_name, param in model.named_parameters():
        if param.requires_grad:
            if param_name.endswith(".bias"):
                biases.append(param)
            else:
                not_biases.append(param)
    optimizer = torch.optim.SGD(
        params=[{"params": biases, "lr": 2 * lr}, {"params": not_biases}],
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    # Move to default device
    model = model.to(device)

    if args.pretrained_weights:
        checkpoint = torch.load(args.pretrained_weights, map_location=device)
        start_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        print("\nLoaded checkpoint from epoch %d.\n" % start_epoch)

    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)

    # Custom dataloaders

    image_transform = A.Compose(
        [
            # Randomly rotate the image and bounding boxes
            A.RandomRotate90(),
            # Randomly flip the image and bounding boxes horizontally
            A.HorizontalFlip(p=0.5),
            # Randomly flip the image and bounding boxes vertically
            A.VerticalFlip(p=0.5),
            # Apply a color jitter to the image
            A.ColorJitter(p=0.5),
            # Apply CLAHE to the image
            A.CLAHE(clip_limit=(1, 4), p=0.5),
            # Add random noise to the image
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            # Add a grid of lines to the image
            A.GridDistortion(p=0.5),
            # Randomly change brightness, contrast and saturation
            A.RandomBrightnessContrast(p=0.5),
            A.RandomGamma(p=0.5),
            A.HueSaturationValue(p=0.5),
            # Add a black rectangle to the image
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
        # however for SSD it transforms it to pascal_voc format
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
    )

    if args.data == "hit":
        train_dataset = HITUAVDatasetTrain(root="./", image_transform=image_transform)
        val_dataset = HITUAVDatasetVal(root="./")
    elif args.data == "wit":
        train_dataset = WITUAVDataset(
            root="./WIT-UAV-Dataset_split/train/",
            sensor=args.wit_sensor,
            image_transform=image_transform,
        )
        val_dataset = WITUAVDataset(
            root="./WIT-UAV-Dataset_split/val/", sensor=args.wit_sensor
        )
    elif args.data == "all":
        train_dataset = CombinedDataset(
            [
                HITUAVDatasetTrain(root="./", image_transform=image_transform),
                WITUAVDataset(
                    root="./WIT-UAV-Dataset_split/train/",
                    sensor=args.wit_sensor,
                    image_transform=image_transform,
                ),
            ]
        )
        val_dataset = CombinedDataset(
            [
                HITUAVDatasetVal(root="./"),
                WITUAVDataset(
                    root="./WIT-UAV-Dataset_split/val/", sensor=args.wit_sensor
                ),
            ]
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=args.n_cpu,
        pin_memory=True,
    )  # note that we're passing the collate function here

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
        num_workers=2,  # for some reason using less workers here makes faster validation
        pin_memory=True,
    )  # note that we're passing the collate function here

    # test_dataset = HITUAVDatasetTest(data_folder)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
    #                                            collate_fn=test_dataset.collate_fn, num_workers=workers,
    #                                            pin_memory=True)  # note that we're passing the collate function here

    # Calculate total number of epochs to train and the epochs to decay learning rate at (i.e. convert iterations to epochs)
    # To convert iterations to epochs, divide iterations by the number of iterations per epoch
    # The paper trains for 120,000 iterations with a batch size of 32, decays after 80,000 and 100,000 iterations
    decay_lr_at = [it // (len(train_dataset) // 32) for it in decay_lr_at]

    wandb_logger.init(name=args.name, config=args)

    wandb_logger.set_config(
        config={
            "model architecture": "SSD",
            "learning rate": lr,
            "momentum": momentum,
            "weight decay": weight_decay,
        }
    )

    # Epochs
    for epoch in range(start_epoch, args.epochs):
        # Decay learning rate at particular epochs
        if epoch in decay_lr_at:
            adjust_learning_rate(optimizer, decay_lr_to)

        # One epoch's training
        train(
            train_loader=train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
        )

        # Save checkpoint
        if args.checkpoint_interval > 0 and epoch % args.checkpoint_interval == 0:
            save_checkpoint(epoch, model, optimizer, logger.log_dir)

        if args.evaluation_interval > 0 and epoch % args.evaluation_interval == 0:
            evaluate(test_loader=val_loader, model=model)


def train(train_loader, model, criterion, optimizer, epoch):
    """
    One epoch's training.
    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: MultiBox loss
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    model.train()  # training mode enables dropout

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss

    start = time.time()

    # Batches
    for i, (images, boxes, labels) in enumerate(tqdm(train_loader, desc="training")):
        batches_done = len(train_loader) * epoch + i

        data_time.update(time.time() - start)

        # Move to default device
        images = images.to(device)  # (batch_size (N), 3, 300, 300)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        # Forward prop.
        predicted_locs, predicted_scores = model(
            images
        )  # (N, 8732, 4), (N, 8732, n_classes)

        # Loss
        loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar

        # Backward prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients, if necessary
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        # Update model
        optimizer.step()

        losses.update(loss.item(), images.size(0))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print(
                "Epoch: [{0}][{1}/{2}]\t"
                "Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t".format(
                    epoch,
                    i,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                )
            )
        # break  # debug overfit one batch

    # Tensorboard logging
    tensorboard_log = [("train/loss", to_cpu(loss).item())]
    logger.list_of_scalars_summary(tensorboard_log, batches_done)

    # VISUALIZE ONLY THE LAST BATCH
    image_list = []
    det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(
        predicted_locs, predicted_scores, min_score=0.01, max_overlap=0.45, top_k=200
    )
    boxes = [b.to(device) for b in boxes]
    labels = [l.to(device) for l in labels]
    wandb_logger.add_batch(
        images=images,
        predictions=[
            torch.cat([box, score.reshape(-1, 1), label.reshape(-1, 1)], dim=1)
            for (box, score, label) in zip(
                det_boxes_batch, det_scores_batch, det_labels_batch
            )
        ],  # each box: [x1, y1, x2, y2, prediction_score, label]
        ground_truths=[
            torch.cat([label.reshape(-1, 1), box], dim=1)
            for (label, box) in zip(labels, boxes)
        ],  # each box: [label, x1, y1, x2, y2]
        class_id_to_label={id: name for name, id in label_map.items()},
        image_list=image_list,
    )  # add batch to image list before bulk upload
    wandb_logger.log(
        {"train/loss": to_cpu(loss).item(), "train/images": image_list, "epoch": epoch}
    )

    del (
        predicted_locs,
        predicted_scores,
        images,
        boxes,
        labels,
    )  # free some memory since their histories may be stored


def to_cpu(tensor):
    return tensor.detach().cpu()


def evaluate(test_loader, model):
    """
    Evaluate.
    :param test_loader: DataLoader for test data
    :param model: model
    """

    # Make sure it's in eval mode
    model.eval()

    # Lists to store detected and true boxes, labels, scores
    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()
    true_difficulties = (
        list()
    )  # it is necessary to know which objects are 'difficult', see 'calculate_mAP' in utils.py
    image_list = []  # for wandb slider visualization

    with torch.no_grad():
        # Batches
        for i, (images, boxes, labels) in enumerate(
            tqdm(test_loader, desc="Evaluating")
        ):
            images = images.to(device)  # (N, 3, 300, 300)

            # Forward prop.
            predicted_locs, predicted_scores = model(images)

            # Detect objects in SSD output
            det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(
                predicted_locs,
                predicted_scores,
                min_score=0.01,
                max_overlap=0.45,
                top_k=200,
            )
            # Evaluation MUST be at min_score=0.01, max_overlap=0.45, top_k=200 for fair comparision with the paper's results and other repos

            # Store this batch's results for mAP calculation
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]
            difficulties = [torch.zeros_like(x).to(device) for x in labels]

            det_boxes.extend(det_boxes_batch)
            det_labels.extend(det_labels_batch)
            det_scores.extend(det_scores_batch)
            true_boxes.extend(boxes)
            true_labels.extend(labels)
            true_difficulties.extend(difficulties)

            wandb_logger.add_batch(
                images=images,
                predictions=[
                    torch.cat([box, score.reshape(-1, 1), label.reshape(-1, 1)], dim=1)
                    for (box, score, label) in zip(
                        det_boxes_batch, det_scores_batch, det_labels_batch
                    )
                ],  # each box: [x1, y1, x2, y2, prediction_score, label]
                ground_truths=[
                    torch.cat([label.reshape(-1, 1), box], dim=1)
                    for (label, box) in zip(labels, boxes)
                ],  # each box: [label, x1, y1, x2, y2]
                class_id_to_label={id: name for name, id in label_map.items()},
                image_list=image_list,
            )  # add batch to image list before bulk upload

        # Calculate mAP
        APs, mAP = calculate_mAP(
            det_boxes,
            det_labels,
            det_scores,
            true_boxes,
            true_labels,
            true_difficulties,
        )

    # Print AP for each class
    pp.pprint(APs)

    print("\nMean Average Precision (mAP): %.3f" % mAP)

    evaluation_metrics = [("validation/mAP", mAP)]
    logger.list_of_scalars_summary(evaluation_metrics, epoch)

    wandb_logger.log({"eval/mAP": mAP, "eval/images": image_list, "epoch": epoch})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trains the SSD model.")
    parser.add_argument(
        "-n", "--name", type=str, default="unnamed", help="Name of experiment"
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

    main()
