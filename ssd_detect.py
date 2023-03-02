from torchvision import transforms
from ssd_utils import *
from PIL import Image, ImageDraw, ImageFont
import os
import torch
from ssd_model import SSD300
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model or load checkpoint
n_classes = len(label_map)
start_epoch = 0
model = SSD300(n_classes=n_classes)
# Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
biases = list()
not_biases = list()
for param_name, param in model.named_parameters():
    if param.requires_grad:
        if param_name.endswith('.bias'):
            biases.append(param)
        else:
            not_biases.append(param)

# Transforms
resize = transforms.Resize((300, 300))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def detect(original_image, min_score, max_overlap, top_k, suppress=None, gen_trace=False):
    """
    Detect objects in an image with a trained SSD300, and visualize the results.
    :param original_image: image, a PIL Image
    :param min_score: minimum threshold for a detected box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :param suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
    :return: annotated image, a PIL Image
    """

    # Transform
    image = to_tensor(resize(original_image))
    transform = transforms.ToPILImage()
    original_image = transform(image)
    image = normalize(image)

    # Move to default device
    image = image.to(device)

    # Forward prop.
    if gen_trace == True:
        traced_script_module = torch.jit.trace(model, image.unsqueeze(0))
        traced_script_module.save("traced_ssd_model.torchscript")
        return
    
    predicted_locs, predicted_scores = model(image.unsqueeze(0))

    # Detect objects in SSD output
    det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                             max_overlap=max_overlap, top_k=top_k)

    # Move detections to the CPU
    det_boxes = det_boxes[0].to('cpu')

    # Transform to original image dimensions
    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims

    # Decode class integer labels
    det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]

    # If no objects found, the detected labels will be set to ['0.'], i.e. ['background'] in SSD300.detect_objects() in model.py
    if det_labels == ['background']:
        # Just return original image
        return original_image

    # Annotate
    annotated_image = original_image
    draw = ImageDraw.Draw(annotated_image)
    font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 15)

    print("detected boxes: ", det_boxes.size(0))

    # Suppress specific classes, if needed
    for i in range(det_boxes.size(0)):
        if suppress is not None:
            if det_labels[i] in suppress:
                continue

        # Boxes
        box_location = det_boxes[i].tolist()
        draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]])
        draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[
            det_labels[i]])  # a second rectangle at an offset of 1 pixel to increase line thickness
        # draw.rectangle(xy=[l + 2. for l in box_location], outline=label_color_map[
        #     det_labels[i]])  # a third rectangle at an offset of 1 pixel to increase line thickness
        # draw.rectangle(xy=[l + 3. for l in box_location], outline=label_color_map[
        #     det_labels[i]])  # a fourth rectangle at an offset of 1 pixel to increase line thickness

        # Text
        text_size = font.getsize(det_labels[i].upper())
        text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
        textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
                            box_location[1]]
        draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])
        draw.text(xy=text_location, text=det_labels[i].upper(), fill='white',
                  font=font)
    # del draw
    plt.imshow(annotated_image)
    return annotated_image


if __name__ == '__main__':
    # Load model checkpoint
    checkpoint_dict = {"HIT_SSD300": "./trained_weights/mukai_desktop_hit_230_checkpoint_ssd300_state_dict.pt",
                    "WIT_SSD300": "./trained_weights/brady_continue_nayana_wit_160_checkpoint_ssd300_state_dict.pt",
                    "ALL_SSD300": "./trained_weights/andrew_desktop_all_120_checkpoint_ssd300_state_dict.pt",
    }

    # Load all checkpoints from checkpoint dict and run inference on each
    for checkpoint_name, checkpoint_path in checkpoint_dict.items():
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint['epoch'] + 1
        print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
        model.load_state_dict(checkpoint['model'])
        model = model.to(device)
        model.eval()

        # /home/devansh/airlab/WIT-UAS-Asset-Detection/WIT-UAV-Dataset/2022-11-08_FIRE-SGL-108-Reade-Township_M600/2022-11-08_14-40-21_seek/time_0575.25.png
        # /home/devansh/airlab/WIT-UAS-Asset-Detection/WIT-UAV-Dataset/2022-11-08_FIRE-SGL-108-Reade-Township_M600/2022-11-08_14-40-21_seek/time_1212.19.png
        # img_name = "1_60_50_0_00281.jpg"
        # img_name = "time_0098.32.png"
        img_name = "time_0575.25.png"
        # img_name = "time_0338.81.png"

        # img_path = os.path.join('./', "HIT-UAV-Infrared-Thermal-Dataset/normal_json/val/", img_name)
        img_path = os.path.join('./', "WIT-UAV-Dataset/2022-11-08_FIRE-SGL-108-Reade-Township_M600/2022-11-08_14-40-21_seek/", img_name)
        original_image = Image.open(img_path, mode='r')
        original_image = original_image.convert('RGB')
        inference = detect(original_image, min_score=0.1, max_overlap=0.5, top_k=200, suppress="noobject")#.show()
        # save annotated image
        save_name = f"./assets/inferences/{checkpoint_name}_{img_name.split('.')[0]}.jpg"
        inference.save(save_name)