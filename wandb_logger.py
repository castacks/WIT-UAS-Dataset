import wandb
import torch

PROJECT_NAME = "HIT-Object-Detection"
WANDB_ENTITY = "cmu-ri-wildfire"
BATCH_INTERVAL = 2


def init(config=None):
    wandb.init(project=PROJECT_NAME,
               entity=WANDB_ENTITY,
               config=config)


def set_config(config):
    wandb.config.update(config)


def log(data):
    wandb.log(data)


def log_image(image, prediction, ground_truth, image_name: str, class_id_to_label: dict):
    pred_box_data = []
    gt_box_data = []

    for box in prediction:  # box: [x1, y1, x2, y2, prediction_score, label]
        pred_box_data.append({"position": {"minX": box[0].item(),
                                           "maxX": box[2].item(),
                                           "minY": box[1].item(),
                                           "maxY": box[3].item()},
                              "domain": "pixel",
                              "class_id": int(box[-1].item()),
                              "box_caption": class_id_to_label[box[-1].item()],
                              "scores": {"prediction score": box[4].item()}})

    for box in ground_truth:  # box: [label, x1, y1, x2, y2]
        gt_box_data.append({"position": {"minX": box[1].item(),
                                         "maxX": box[3].item(),
                                         "minY": box[2].item(),
                                         "maxY": box[4].item()},
                            "domain": "pixel",
                            "class_id": int(box[0].item()),
                            "box_caption": class_id_to_label[box[0].item()]})

    wandb_image = wandb.Image(image, boxes={
        "predictions": {
            "box_data": pred_box_data,
            "class_labels": class_id_to_label
        },
        # Log each meaningful group of boxes with a unique key name
        "ground truth": {
            "box_data": gt_box_data,
            "class_labels": class_id_to_label
        }
    })

    wandb.log({image_name: wandb_image})


def log_batch(images, predictions, ground_truths, batch_name, class_id_to_label):
    #! batch size is small (5 for now), so only logging first image of each batch
    log_image(image=images[0],
              prediction=predictions[0],
              ground_truth=ground_truths[ground_truths[:, 0] == 0][:, 1:],
              image_name=batch_name,
              class_id_to_label=class_id_to_label)
