import os
import torch
from PIL import Image
import json
import torchvision.transforms.functional as FT

class HITUAVDataset(torch.utils.data.Dataset):
    def __init__(self, root):
        self.root = root
        self.imgs_train = os.path.join(root, "HIT-UAV-Infrared-Thermal-Dataset/normal_json/train/")
        self.imgs_train_list = list(sorted(os.listdir(self.imgs_train)))
        self.annotations_file = os.path.join(root, "HIT-UAV-Infrared-Thermal-Dataset/normal_json/annotations/train.json")
        self.annotations_json = json.load(open(self.annotations_file))
        self.annotations_bbox_category = self.annotations_json['annotation']
        self.imgs_train_name_ids = self.annotations_json['images']
    
    def __len__(self):
        return len(self.imgs_train_list)
    
    def __getitem__(self, idx):
        img_name_id = self.imgs_train_name_ids[idx]
        img_name = img_name_id['filename']
        image = Image.open(os.path.join(self.imgs_train, img_name), mode='r').convert('RGB')
        objects_dict = [x for x in self.annotations_bbox_category if x['image_id'] == img_name_id['id']]
        bboxes_list = []
        labels_list = []
        for object in objects_dict:
            bbox = object['bbox']
            xmin = (bbox[0] - bbox[2]/2) if (bbox[0] - bbox[2]/2)/640 >= 0 else 0
            ymin = (bbox[1] - bbox[3]/2) if (bbox[1] - bbox[3]/2)/512 >= 0 else 0
            xmax = (bbox[0] + bbox[2]/2) if (bbox[0] + bbox[2]/2)/640 <= 1 else 1
            ymax = (bbox[1] + bbox[3]/2) if (bbox[1] + bbox[3]/2)/512 <= 1 else 1
            bbox = [xmin, ymin, xmax, ymax]
            bboxes_list.append(bbox)
            labels_list.append(object['category_id'] + 1)
        if len(bboxes_list) == 0 and len(labels_list) == 0:
            bbox = [0, 0, 640, 512]
            label = 6
            bboxes_list.append(bbox)
            labels_list.append(label)
        boxes = torch.FloatTensor(bboxes_list)
        labels = torch.LongTensor(labels_list)
        image, boxes = self.resize(image, boxes)
        image = FT.to_tensor(image)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        image = FT.normalize(image, mean=mean, std=std)
        return image, boxes, labels
    
    def collate_fn(self, batch):
        images = list()
        boxes = list()
        labels = list()
        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
        images = torch.stack(images, dim=0)
        return images, boxes, labels
    
    def resize(self, image, boxes, dims=[300, 300], return_percent_coords=True):
        """
        Resize image. For the SSD300, resize to (300, 300).
        Since percent/fractional coordinates are calculated for the bounding boxes (w.r.t image dimensions) in this process,
        you may choose to retain them.
        :param image: image, a PIL Image
        :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
        :return: resized image, updated bounding box coordinates (or fractional coordinates, in which case they remain the same)
        """
        try:
            # Resize image
            new_image = FT.resize(image, dims)

            # Resize bounding boxes
            old_dims = torch.FloatTensor([image.width, image.height, image.width, image.height]).unsqueeze(0)
            new_boxes = boxes / old_dims  # percent coordinates

            if not return_percent_coords:
                new_dims = torch.FloatTensor([dims[1], dims[0], dims[1], dims[0]]).unsqueeze(0)
                new_boxes = new_boxes * new_dims

            return new_image, new_boxes
        except:
            print('unexpected error')
            return new_image, boxes