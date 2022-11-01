import os
import torch
from PIL import Image
import json
import torchvision.transforms.functional as FT
import numpy as np
import torchvision.transforms as transforms

class HITUAVDatasetTrain(torch.utils.data.Dataset):
    def __init__(self, root, yolo=False, yolo_dim=[416, 416]):
        self.yolo = yolo
        self.yolo_dim = yolo_dim
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
            if self.yolo == False:
                xmin = (bbox[0] - bbox[2]/2) if (bbox[0] - bbox[2]/2)/640 >= 0 else 0
                ymin = (bbox[1] - bbox[3]/2) if (bbox[1] - bbox[3]/2)/512 >= 0 else 0
                xmax = (bbox[0] + bbox[2]/2) if (bbox[0] + bbox[2]/2)/640 <= 1 else 640
                ymax = (bbox[1] + bbox[3]/2) if (bbox[1] + bbox[3]/2)/512 <= 1 else 512
                bbox = [xmin, ymin, xmax, ymax]
            else:
                bbox = [bbox[0]/640, bbox[1]/512, bbox[2]/640, bbox[3]/512]
            bboxes_list.append(bbox)
            labels_list.append(object['category_id'] + 1)
        if len(bboxes_list) == 0 and len(labels_list) == 0:
            if self.yolo == False:
                bbox = [0, 0, 640, 512]
                label = 6
                bboxes_list.append(bbox)
                labels_list.append(label)
            else:
                bbox = [0, 0, 1, 1]
                label = 5
                bboxes_list.append(bbox)
                labels_list.append(label)
        if self.yolo == False:
            boxes = torch.FloatTensor(bboxes_list)
            labels = torch.LongTensor(labels_list)
            image, boxes = self.resize(image, boxes)
            image = FT.to_tensor(image)
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            image = FT.normalize(image, mean=mean, std=std)
            return image, boxes, labels
        else:
            np_targets = []
            for b, l in zip(bboxes_list, labels_list):
                np_targets.append([l] + b)
            targets = np.array(np_targets)
            bb_targets = torch.zeros((len(targets), 6))
            bb_targets[:, 1:] = transforms.ToTensor()(targets)
            image = FT.resize(image, self.yolo_dim)
            image = np.array(image, dtype=np.uint8)
            image = transforms.ToTensor()(image)
            return image, bb_targets
    
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
    
    def yolo_collate_fn(self, batch):
        # self.batch_count += 1

        # Drop invalid images
        batch = [data for data in batch if data is not None]

        imgs, bb_targets = list(zip(*batch))

        # # Selects new image size every tenth batch
        # if self.multiscale and self.batch_count % 10 == 0:
        #     self.img_size = random.choice(
        #         range(self.min_size, self.max_size + 1, 32))

        # Resize images to input shape
        # imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        imgs = torch.stack([img for img in imgs])

        # Add sample index to targets
        for i, boxes in enumerate(bb_targets):
            boxes[:, 0] = i
        bb_targets = torch.cat(bb_targets, 0)

        return imgs, bb_targets
    
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

class HITUAVDatasetVal(torch.utils.data.Dataset):
    def __init__(self, root, yolo=False, yolo_dim=[416, 416]):
        self.yolo = yolo
        self.yolo_dim = yolo_dim
        self.root = root
        self.imgs_train = os.path.join(root, "HIT-UAV-Infrared-Thermal-Dataset/normal_json/val/")
        self.imgs_train_list = list(sorted(os.listdir(self.imgs_train)))
        self.annotations_file = os.path.join(root, "HIT-UAV-Infrared-Thermal-Dataset/normal_json/annotations/val.json")
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
            if self.yolo == False:
                xmin = (bbox[0] - bbox[2]/2) if (bbox[0] - bbox[2]/2)/640 >= 0 else 0
                ymin = (bbox[1] - bbox[3]/2) if (bbox[1] - bbox[3]/2)/512 >= 0 else 0
                xmax = (bbox[0] + bbox[2]/2) if (bbox[0] + bbox[2]/2)/640 <= 1 else 640
                ymax = (bbox[1] + bbox[3]/2) if (bbox[1] + bbox[3]/2)/512 <= 1 else 512
                bbox = [xmin, ymin, xmax, ymax]
            else:
                bbox = [bbox[0]/640, bbox[1]/512, bbox[2]/640, bbox[3]/512]
            bboxes_list.append(bbox)
            labels_list.append(object['category_id'] + 1)
        if len(bboxes_list) == 0 and len(labels_list) == 0:
            if self.yolo == False:
                bbox = [0, 0, 640, 512]
                label = 6
                bboxes_list.append(bbox)
                labels_list.append(label)
            else:
                bbox = [0, 0, 1, 1]
                label = 5
                bboxes_list.append(bbox)
                labels_list.append(label)
        if self.yolo == False:
            boxes = torch.FloatTensor(bboxes_list)
            labels = torch.LongTensor(labels_list)
            image, boxes = self.resize(image, boxes)
            image = FT.to_tensor(image)
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            image = FT.normalize(image, mean=mean, std=std)
            return image, boxes, labels
        else:
            np_targets = []
            for b, l in zip(bboxes_list, labels_list):
                np_targets.append([l] + b)
            targets = np.array(np_targets)
            bb_targets = torch.zeros((len(targets), 6))
            bb_targets[:, 1:] = transforms.ToTensor()(targets)
            image = FT.resize(image, self.yolo_dim)
            image = np.array(image, dtype=np.uint8)
            image = transforms.ToTensor()(image)
            return image, bb_targets
    
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
    
    def yolo_collate_fn(self, batch):
        # self.batch_count += 1

        # Drop invalid images
        batch = [data for data in batch if data is not None]

        imgs, bb_targets = list(zip(*batch))

        # # Selects new image size every tenth batch
        # if self.multiscale and self.batch_count % 10 == 0:
        #     self.img_size = random.choice(
        #         range(self.min_size, self.max_size + 1, 32))

        # Resize images to input shape
        # imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        imgs = torch.stack([img for img in imgs])

        # Add sample index to targets
        for i, boxes in enumerate(bb_targets):
            boxes[:, 0] = i
        bb_targets = torch.cat(bb_targets, 0)

        return imgs, bb_targets
    
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

class HITUAVDatasetTest(torch.utils.data.Dataset):
    def __init__(self, root, yolo=False, yolo_dim=[416, 416]):
        self.yolo = yolo
        self.yolo_dim = yolo_dim
        self.root = root
        self.imgs_train = os.path.join(root, "HIT-UAV-Infrared-Thermal-Dataset/normal_json/test/")
        self.imgs_train_list = list(sorted(os.listdir(self.imgs_train)))
        self.annotations_file = os.path.join(root, "HIT-UAV-Infrared-Thermal-Dataset/normal_json/annotations/test.json")
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
            if self.yolo == False:
                xmin = (bbox[0] - bbox[2]/2) if (bbox[0] - bbox[2]/2)/640 >= 0 else 0
                ymin = (bbox[1] - bbox[3]/2) if (bbox[1] - bbox[3]/2)/512 >= 0 else 0
                xmax = (bbox[0] + bbox[2]/2) if (bbox[0] + bbox[2]/2)/640 <= 1 else 640
                ymax = (bbox[1] + bbox[3]/2) if (bbox[1] + bbox[3]/2)/512 <= 1 else 512
                bbox = [xmin, ymin, xmax, ymax]
            else:
                bbox = [bbox[0]/640, bbox[1]/512, bbox[2]/640, bbox[3]/512]
            bboxes_list.append(bbox)
            labels_list.append(object['category_id'] + 1)
        if len(bboxes_list) == 0 and len(labels_list) == 0:
            if self.yolo == False:
                bbox = [0, 0, 640, 512]
                label = 6
                bboxes_list.append(bbox)
                labels_list.append(label)
            else:
                bbox = [0, 0, 1, 1]
                label = 5
                bboxes_list.append(bbox)
                labels_list.append(label)
        if self.yolo == False:
            boxes = torch.FloatTensor(bboxes_list)
            labels = torch.LongTensor(labels_list)
            image, boxes = self.resize(image, boxes)
            image = FT.to_tensor(image)
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            image = FT.normalize(image, mean=mean, std=std)
            return image, boxes, labels
        else:
            np_targets = []
            for b, l in zip(bboxes_list, labels_list):
                np_targets.append([l] + b)
            targets = np.array(np_targets)
            bb_targets = torch.zeros((len(targets), 6))
            bb_targets[:, 1:] = transforms.ToTensor()(targets)
            image = FT.resize(image, self.yolo_dim)
            image = np.array(image, dtype=np.uint8)
            image = transforms.ToTensor()(image)
            return image, bb_targets
    
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
    
    def yolo_collate_fn(self, batch):
        # self.batch_count += 1

        # Drop invalid images
        batch = [data for data in batch if data is not None]

        imgs, bb_targets = list(zip(*batch))

        # # Selects new image size every tenth batch
        # if self.multiscale and self.batch_count % 10 == 0:
        #     self.img_size = random.choice(
        #         range(self.min_size, self.max_size + 1, 32))

        # Resize images to input shape
        # imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        imgs = torch.stack([img for img in imgs])

        # Add sample index to targets
        for i, boxes in enumerate(bb_targets):
            boxes[:, 0] = i
        bb_targets = torch.cat(bb_targets, 0)

        return imgs, bb_targets
    
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