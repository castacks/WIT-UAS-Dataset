from datetime import datetime

import torch

import wandb

PROJECT_NAME = "HIT-Object-Detection"
WANDB_ENTITY = "cmu-ri-wildfire"
BATCH_INTERVAL = 2


def init(config=None, name=""):
    """init wandb logger, sets train and eval blobs hooked to epoch as x-axis

    Args:
        config (dict, optional): settings to upload. Defaults to None.
    """
    wandb.init(
        project=PROJECT_NAME,
        entity=WANDB_ENTITY,
        config=config,
        name=str(datetime.now()) + " " + name,
    )

    # define plot x-axis and variables
    wandb.define_metric("epoch")  # x-axis
    wandb.define_metric("train/*", step_metric="epoch")
    wandb.define_metric("eval/*", step_metric="epoch")


def set_config(config):
    """set config incrementally any time

    Args:
        config (dict): add anything as config in format {name(str): info(str, number_like), ...}
    """
    wandb.config.update(config)


def log(data):
    """log any info any time incrementally, mostly when training

    Args:
        data (dict): log anything in format {name(str): info(str, number_like), ...}
    """
    wandb.log(data)


def draw_boxes(image, prediction, ground_truth, class_id_to_label: dict, box_unit=None):
    """draw predicted and ground truth boxes on given image and produce a wandb image object

    Args:
        image (image): image from dataset
        prediction (list_like): list containing predictions of boxes over the entire image in format [x1, y1, x2, y2, prediction_score, label]
        ground_truth (list_like): list containing ground truth boxes over the entire image in format [label, x1, y1, x2, y2], from dataset
        class_id_to_label (dict): dictionary mapping from id(int starting from 0) to name(str), e.g. {0: car, 1: people}
        box_unit (str, optional): unit of box corner coordinates, specify as "pixel" if pixel. Defaults to None, which is fraction.

    Returns:
        wandb_image(wandb.Image): 1 single wandb image object
    """
    wandb_image = wandb.Image(
        image,
        boxes={
            "predictions": {  # predicted box: [x1, y1, x2, y2, prediction_score, label]
                "box_data": [
                    {
                        "position": {
                            "minX": box[0].item(),
                            "maxX": box[2].item(),
                            "minY": box[1].item(),
                            "maxY": box[3].item(),
                        },
                        "domain": box_unit,
                        "class_id": int(box[-1].item()),
                        "box_caption": str(class_id_to_label[int(box[-1].item())])
                        + "_pred",
                        "scores": {"prediction score": box[4].item()},
                    }
                    for box in prediction
                ],
                "class_labels": class_id_to_label,
            },
            "ground truth": {  # ground truth box: [label, x1, y1, x2, y2]
                "box_data": [
                    {
                        "position": {
                            "minX": box[1].item(),
                            "maxX": box[3].item(),
                            "minY": box[2].item(),
                            "maxY": box[4].item(),
                        },
                        "domain": box_unit,
                        "class_id": int(box[0].item()),
                        "box_caption": str(class_id_to_label[int(box[0].item())])
                        + "_gt",
                    }
                    for box in ground_truth
                ],
                "class_labels": class_id_to_label,
            },
        },
    )

    return wandb_image


def add_batch(
    images,
    predictions,
    ground_truths,
    class_id_to_label: dict,
    image_list=None,
    box_unit=None,
):
    """add wandb image objects to given image_list

    Args:
        images (image): a batch of images from dataset
        predictions (list_like): a batch of predictions on given images in dimension [n_images, n_predicted_boxes]
        ground_truths (list_like): a batch of ground truth of given images from dataset in dimension [n_images, n_gound_truth_boxes]
        class_id_to_label (dict): dictionary mapping from id(int starting from 0) to name(str), e.g. {0: car, 1: people}
        image_list (list_like, optional): list of wandb image objects of this whole epoch to be uploaded. Defaults to None.
        box_unit (str, optional): unit of box corner coordinates, specify as "pixel" if pixel. Defaults to None, which is fraction.
    """
    #! batch size is small (5 for now), so only logging first image of each batch
    wandb_image = draw_boxes(
        image=images[0],
        prediction=predictions[0],
        ground_truth=ground_truths[0],
        class_id_to_label=class_id_to_label,
        box_unit=box_unit,
    )

    image_list.append(wandb_image)
