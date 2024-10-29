pip install wget

import matplotlib.pyplot as plt
from PIL import Image
import xml.etree.ElementTree as ET
from collections import Counter
import seaborn as sns
import glob

import os
import random
import shutil
from sklearn.model_selection import train_test_split
import wget

data_dir = './VOC_dataset/'
train_dir = './VOC_dataset/train/'
test_dir = './VOC_dataset/test/'

def download_voc_data(url, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    wget.download(url, out=output_dir)

url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar"

print("Downloading Pascal VOC 2007 dataset...")
download_voc_data(url, data_dir)

print("Unzipping the dataset...")
os.system(f'tar -xvf {data_dir}VOCtrainval_06-Nov-2007.tar -C {data_dir}')

images = os.listdir(os.path.join(data_dir, 'VOCdevkit/VOC2007/JPEGImages'))
random.shuffle(images)

images = images[:1000]

train_images, test_images = train_test_split(images, test_size=0.2, random_state=42)

if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

def move_images(image_list, src_folder, dst_folder):
    for image in image_list:
        image_path = os.path.join(src_folder, image)
        shutil.copy(image_path, dst_folder)

move_images(train_images, os.path.join(data_dir, 'VOCdevkit/VOC2007/JPEGImages'), train_dir)
move_images(test_images, os.path.join(data_dir, 'VOCdevkit/VOC2007/JPEGImages'), test_dir)

print(f"Training set contains {len(train_images)} images")
print(f"Testing set contains {len(test_images)} images")

annotation_dir = os.path.join(data_dir, 'VOCdevkit/VOC2007/Annotations')
annotation_files = glob.glob(os.path.join(annotation_dir, '*.xml'))

def get_object_types(annotation_file):
    tree = ET.parse(annotation_file)
    root = tree.getroot()
    object_types = []
    for obj in root.findall('object'):
        obj_name = obj.find('name').text
        object_types.append(obj_name)
    return object_types

object_type_counter = Counter()
for annotation_file in annotation_files:
    object_types = get_object_types(annotation_file)
    object_type_counter.update(object_types)

plt.figure(figsize=(10, 6))
sns.barplot(x=list(object_type_counter.keys()), y=list(object_type_counter.values()))
plt.title('Distribution of Object Types in Pascal VOC Dataset')
plt.xticks(rotation=90)
plt.ylabel('Frequency')
plt.show()

def show_images_with_annotations(image_files, num_images=5):
    fig, axes = plt.subplots(1, num_images, figsize=(15, 15))
    selected_images = random.sample(image_files, num_images)

    for i, image_file in enumerate(selected_images):
        img_path = os.path.join(data_dir, 'VOCdevkit/VOC2007/JPEGImages', image_file)
        img = Image.open(img_path)
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f'Image: {image_file}')
    plt.show()

# Show 5 random images from the training set
show_images_with_annotations(train_images, num_images=5)

import torch
from torchvision import transforms
from PIL import Image
import os

train_dir = './VOC_dataset/train/'
test_dir = './VOC_dataset/test/'

preprocess_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def preprocess_image(image_path, preprocess_pipeline):
    img = Image.open(image_path)
    img = preprocess_pipeline(img)
    return img

def preprocess_dataset(image_list, src_folder, preprocess_pipeline, num_images=5):
    for i, image_file in enumerate(image_list[:num_images]):
        img_path = os.path.join(src_folder, image_file)
        processed_img = preprocess_image(img_path, preprocess_pipeline)
        print(f"Processed Image {i+1}: Shape {processed_img.shape}, Min pixel value: {processed_img.min()}, Max pixel value: {processed_img.max()}")

preprocess_dataset(train_images, train_dir, preprocess_pipeline, num_images=5)

import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import functional as F
import os
from PIL import Image

def get_faster_rcnn_model(num_classes):

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    return model

def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img_tensor = F.to_tensor(img)
    return img_tensor

def test_faster_rcnn(image_list, model):
    model.eval()
    with torch.no_grad():
        for image_file in image_list[:2]:
            img_path = os.path.join(train_dir, image_file)
            img_tensor = preprocess_image(img_path)
            prediction = model([img_tensor])

            print(f"Predicted labels: {prediction[0]['labels']}")
            print(f"Predicted boxes: {prediction[0]['boxes']}")
            print(f"Predicted scores: {prediction[0]['scores']}")

num_classes = 21

model = get_faster_rcnn_model(num_classes)

test_faster_rcnn(train_images, model)


import torch
import torchvision
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import VOCDetection
from torchvision import transforms
import xml.etree.ElementTree as ET
import os

transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor()
])

def parse_voc_annotations(ann_path):
    tree = ET.parse(ann_path)
    root = tree.getroot()
    boxes = []
    labels = []
    for obj in root.findall('object'):
        label = obj.find('name').text
        labels.append(label_to_idx[label])

        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        boxes.append([xmin, ymin, xmax, ymax])

    return torch.tensor(boxes), torch.tensor(labels)

class VOCDataset(VOCDetection):
    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)

        ann_path = os.path.join(self.root, 'VOCdevkit', 'VOC2007', 'Annotations', target['annotation']['filename'] + '.xml')

        if not os.path.exists(ann_path):
            print(f"Annotation file for {target['annotation']['filename']} is missing. Skipping.")
            return None, None

        boxes, labels = parse_voc_annotations(ann_path)
        target = {'boxes': boxes, 'labels': labels}
        return img, target

label_to_idx = {
    'aeroplane': 1, 'bicycle': 2, 'bird': 3, 'boat': 4, 'bottle': 5,
    'bus': 6, 'car': 7, 'cat': 8, 'chair': 9, 'cow': 10, 'diningtable': 11,
    'dog': 12, 'horse': 13, 'motorbike': 14, 'person': 15, 'pottedplant': 16,
    'sheep': 17, 'sofa': 18, 'train': 19, 'tvmonitor': 20
}

# Load Pascal VOC dataset with transform
dataset = VOCDataset('./VOC_dataset', year='2007', image_set='train', download=False, transform=transform)

# Split dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# DataLoader for batching
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# Load Faster R-CNN model with ResNet-50 backbone
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes=21)

# Define optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)

# Training loop
for epoch in range(10):
    model.train()
    for images, targets in train_loader:
        if images[0] is None:  # Skip instances where annotations are missing
            continue
        images = list(image for image in images)
        targets = [{k: v for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    print(f'Epoch {epoch+1} completed.')
