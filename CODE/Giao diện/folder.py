import os
import cv2
import torch
import numpy as np
import tkinter as tk
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tkinter import Tk, filedialog
import matplotlib.pyplot as plt

def show_message(message, title="Thông báo"):
    tk.messagebox.showinfo(title, message)

def obj_detector(img):
    img = cv2.imread(img)
    if img is None:
        show_message("Không thể đọc hình ảnh từ đường dẫn:", img)
        raise ValueError("Không thể đọc hình ảnh từ đường dẫn:", img)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img /= 255.0
    img = torch.from_numpy(img)
    img = img.unsqueeze(0)
    img = img.permute(0, 3, 1, 2)

    model.eval()
    detection_threshold = 0.50

    img = [image.to(device) for image in img]
    output = model(img)

    for i, image in enumerate(img):
        boxes = output[i]['boxes'].detach().cpu().numpy()
        scores = output[i]['scores'].detach().cpu().numpy()
        labels = output[i]['labels'].detach().cpu().numpy()

        labels = labels[scores >= detection_threshold]
        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        scores = scores[scores >= detection_threshold]

        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

    sample = img[0].permute(1, 2, 0).cpu().numpy()
    sample = np.array(sample)
    boxes = output[0]['boxes'].detach().cpu().numpy()
    name = output[0]['labels'].detach().cpu().numpy()
    scores = output[0]['scores'].detach().cpu().numpy()
    boxes = boxes[scores >= detection_threshold].astype(np.int32)
    names = name.tolist()

    return names, boxes, sample

# Tạo cửa sổ chọn thư mục
root = Tk()
root.withdraw()

# Hiển thị hộp thoại để chọn thư mục
pred_path = filedialog.askdirectory(title="Chọn thư mục chứa hình ảnh")

# Kiểm tra nếu người dùng không chọn thư mục
if not pred_path:
    show_message("Bạn chưa chọn thư mục.")
else:
    # Load a model pre-trained on COCO
    model = fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)
    WEIGHTS_FILE = "model.pth"
    num_classes = 10
    classes = {1: 'bus', 2: 'car', 3: 'truck', 4: 'motorbike'}

    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Load the trained weights
    model.load_state_dict(torch.load(WEIGHTS_FILE, map_location=torch.device('cpu')))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Lặp qua tất cả các hình ảnh trong thư mục
    pred_files = [os.path.join(pred_path, f) for f in os.listdir(pred_path)]
    num_images = min(20, len(pred_files))  # Limit the number of images to 20

    if num_images > 0:
        num_columns = int(np.ceil(np.sqrt(num_images)))  # Calculate the number of columns

        # Ensure the number of rows is a positive integer
        num_rows = int(np.ceil(num_images / num_columns))

        plt.figure(figsize=(20, 20))
        for i, images in enumerate(pred_files[:num_images]):
            plt.subplot(num_rows, num_columns, i + 1)
            try:
                names, boxes, sample = obj_detector(images)
            except ValueError as e:
                show_message(str(e))
                break

            if all(arr is not None for arr in (names, boxes, sample)):  # Check if all elements in the tuple are not None
                for i, box in enumerate(boxes):
                    cv2.rectangle(sample, (box[0], box[1]), (box[2], box[3]), (0, 220, 0), 2)
                    # Thay đổi đoạn này để hiển thị tên class thay vì chỉ số class
                    cv2.putText(sample, str(classes[names[i]]), (box[0], box[1] - 5), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                                (220, 0, 0), 1, cv2.LINE_AA)

                plt.axis('off')
                plt.imshow(sample)

        plt.show()
