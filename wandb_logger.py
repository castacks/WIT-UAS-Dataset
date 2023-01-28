import wandb

PROJECT_NAME = "HIT-Object-Detection"
WANDB_ENTITY = "cmu-ri-wildfire"
BATCH_INTERVAL = 2


def init(config=None):
    wandb.init(project=PROJECT_NAME,
               entity=WANDB_ENTITY,
               config=config)


def set_config(config):
    wandb.config.update(config)


def log_image(image, prediction, ground_truth, image_name: str, class_id_to_label: dict):
    pred_box_data = []
    gt_box_data = []

    for box in prediction:  # box: [x1, y1, x2, y2, prediction_score, label]
        pred_box_data.append({"position": {"minX": box[0],
                                           "minY": box[1],
                                           "maxX": box[2],
                                           "maxY": box[3]},
                              "domain": "pixel",
                              "class_id": box[-1],
                              "box_caption": class_id_to_label[box[-1]],
                              "scores": box[4]})

    for box in ground_truth:  # box: [?, label, x1, y1, x2, y2]
        gt_box_data.append({"position": {"minX": box[2],
                                         "minY": box[3],
                                         "maxX": box[4],
                                         "maxY": box[5]},
                            "domain": "pixel",
                            "class_id": box[1],
                            "box_caption": class_id_to_label[box[1]]})

    wandb_image = wandb.Image(image, boxes={
        "predictions": {
            "box_data": pred_box_data,
            "class_labels": class_id_to_label
        },
        # Log each meaningful group of boxes with a unique key name
        "ground_truth": {
            "box_data": gt_box_data,
            "class_labels": class_id_to_label
        }
    })

    wandb.log({image_name: wandb_image})


def log_batch(images, predictions, ground_truths, batch_name, class_id_to_label):
    #! batch size is small (5 for now), so only logging first image of each batch
    log_image(image=images[0],
              prediction=predictions[0],
              ground_truth=ground_truths[0],
              image_name=batch_name,
              class_id_to_label=class_id_to_label)
