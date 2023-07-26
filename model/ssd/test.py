#!/usr/bin/env python3
import os
import sys

root_folder = os.path.abspath(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(root_folder)  # to enable import from parent directory

from pprint import PrettyPrinter

import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from tqdm import tqdm

from tools.dataset import HITUAVDatasetTest
from tools.dataset import HITUAVDatasetTrain
from tools.dataset import HITUAVDatasetVal
from model import MultiBoxLoss
from model import SSD300
from utils import *

pp = PrettyPrinter()

# Data parameters
data_folder = "./"  # folder with data files

# Model parameters
# Not too many here since the SSD300 has a very specific structure
n_classes = len(label_map)  # number of different types of objects
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Learning parameters
checkpoint = "./ssd_logs/2022_11_01__22_35_59/950_checkpoint_ssd300_state_dict.pt"  # path to model checkpoint, None if none
batch_size = 8  # batch size
iterations = 120000  # number of iterations to train
workers = 4  # number of workers for loading data in the DataLoader
print_freq = 2  # print training status every __ batches
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

    if checkpoint is not None:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint["epoch"] + 1
        print("\nLoaded checkpoint from epoch %d.\n" % start_epoch)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

    # Move to default device
    model = model.to(device)
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)

    # Custom dataloaders
    # train_dataset = HITUAVDatasetTrain(data_folder)
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
    #                                            collate_fn=train_dataset.collate_fn, num_workers=workers,
    #                                            pin_memory=True)  # note that we're passing the collate function here

    # val_dataset = HITUAVDatasetVal(data_folder)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True,
    #                                            collate_fn=val_dataset.collate_fn, num_workers=workers,
    #                                            pin_memory=True)  # note that we're passing the collate function here

    test_dataset = HITUAVDatasetTest(data_folder)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=test_dataset.collate_fn,
        num_workers=workers,
        pin_memory=True,
    )  # note that we're passing the collate function here

    # Calculate total number of epochs to train and the epochs to decay learning rate at (i.e. convert iterations to epochs)
    # To convert iterations to epochs, divide iterations by the number of iterations per epoch
    # The paper trains for 120,000 iterations with a batch size of 32, decays after 80,000 and 100,000 iterations
    # epochs = iterations // (len(train_dataset) // 32)
    # decay_lr_at = [it // (len(train_dataset) // 32) for it in decay_lr_at]

    evaluate(test_loader=test_loader, model=model)


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


if __name__ == "__main__":
    main()
