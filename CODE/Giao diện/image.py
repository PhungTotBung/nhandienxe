import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import torch
import numpy as np
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.pyplot as plt
import os

def show_message(message, title="Thông báo"):
    tk.messagebox.showinfo(title, message)

# Tạo cửa sổ chọn file
root = tk.Tk()
root.withdraw()

# Hiển thị hộp thoại để chọn file
file_path = filedialog.askopenfilename(title="Chọn hình ảnh")

# Chuẩn hóa đường dẫn
normalized_path = os.path.normpath(file_path)

# Sử dụng normalized_path trong các phần code tiếp theo
print(normalized_path)

# Kiểm tra nếu người dùng không chọn file
if not file_path:
    show_message("Bạn chưa chọn hình ảnh.")
else:
    # Load a model pre-trained on COCO
    model = fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)
    WEIGHTS_FILE = "model.pth"
    num_classes = 10
    classes = {1:'bus', 2:'car', 6:'truck', 4:'motorbike'}

    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Load the trained weights
    model.load_state_dict(torch.load(WEIGHTS_FILE, map_location=torch.device('cpu')))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    def obj_detector(img):
        img = cv2.imread(img)
        if img is None:
            show_message(f"Không thể đọc hình ảnh từ đường dẫn: {img}")
            return None, None, None

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

        img /= 255.0
        img = torch.from_numpy(img)
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2)

        model.eval()
        detection_threshold = 0.50

        img = list(im.to(device) for im in img)
        output = model(img)

        for i, im in enumerate(img):
            boxes = output[i]['boxes'].data.cpu().numpy()
            scores = output[i]['scores'].data.cpu().numpy()
            labels = output[i]['labels'].data.cpu().numpy()

            labels = labels[scores >= detection_threshold]
            boxes = boxes[scores >= detection_threshold].astype(np.int32)
            scores = scores[scores >= detection_threshold]

            boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

        sample = img[0].permute(1, 2, 0).cpu().numpy()
        sample = np.array(sample)
        boxes = output[0]['boxes'].data.cpu().numpy()
        name = output[0]['labels'].data.cpu().numpy()
        scores = output[0]['scores'].data.cpu().numpy()
        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        names = name.tolist()

        return names, boxes, sample

    # Chọn hình ảnh cần hiển thị
    selected_image = normalized_path
    names, boxes, sample = obj_detector(selected_image)

    if names is not None:
        for i, box in enumerate(boxes):
            cv2.rectangle(sample, (box[0], box[1]), (box[2], box[3]), (0, 220, 0), 2)
            # Thay đổi đoạn này để hiển thị tên class thay vì chỉ số class
            class_index = names[i]
            class_name = classes.get(class_index, f"Unknown Class {class_index}")
            cv2.putText(sample, class_name, (box[0], box[1]-5), cv2.FONT_HERSHEY_COMPLEX, 0.7, (220, 0, 0), 1, cv2.LINE_AA)

        # Hiển thị hình ảnh đã chọn và gán nhãn
        plt.imshow(sample)
        plt.axis('off')
        plt.show()
