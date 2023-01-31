import wandb
import torch

PROJECT_NAME = "HIT-Object-Detection"
WANDB_ENTITY = "cmu-ri-wildfire"
BATCH_INTERVAL = 2


def init(config=None):
    wandb.init(project=PROJECT_NAME,
               entity=WANDB_ENTITY,
               config=config)

    # define plot x-axis and variables
    wandb.define_metric("epoch")  # x-axis
    wandb.define_metric("train/*", step_metric="epoch")
    wandb.define_metric("eval/*", step_metric="epoch")


def set_config(config):
    wandb.config.update(config)


def log(data):
    wandb.log(data)


def draw_boxes(image, prediction, ground_truth, class_id_to_label: dict):
    wandb_image = wandb.Image(image, boxes={
        "predictions": {  # predicted box: [x1, y1, x2, y2, prediction_score, label]
            "box_data": [{"position": {"minX": box[0].item(),
                                       "maxX": box[2].item(),
                                       "minY": box[1].item(),
                                       "maxY": box[3].item()},
                          "domain": "pixel",
                          "class_id": int(box[-1].item()),
                          "color": [0, 0, 255],
                          "box_caption": str(class_id_to_label[box[-1].item()]) + "_pred",
                          "scores": {"prediction score": box[4].item()}} for box in prediction],
            "class_labels": class_id_to_label
        },
        "ground truth": {  # ground truth box: [label, x1, y1, x2, y2]
            "box_data": [{"position": {"minX": box[1].item(),
                                       "maxX": box[3].item(),
                                       "minY": box[2].item(),
                                       "maxY": box[4].item()},
                          "domain": "pixel",
                          "class_id": int(box[0].item()),
                          "color": [255, 0, 0],
                          "box_caption": str(class_id_to_label[box[0].item()]) + "_gt"} for box in ground_truth],
            "class_labels": class_id_to_label
        }
    })

    return wandb_image


def add_batch(images, predictions, ground_truths, epoch, num_batch, class_id_to_label, image_list=None):
    #! batch size is small (5 for now), so only logging first image of each batch
    wandb_image = draw_boxes(image=images[0],
                             prediction=predictions[0],
                             ground_truth=ground_truths[ground_truths[:, 0]
                                                        == 0][:, 1:],
                             class_id_to_label=class_id_to_label)

    image_list.append(wandb_image)
