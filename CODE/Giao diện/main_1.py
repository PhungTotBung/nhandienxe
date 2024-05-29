import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os
from ttkthemes import ThemedStyle  

def run_app():
    os.system("python testfolder.py")

def run_zone():
    os.system("python testzone.py")

def run_image():
    os.system("python test.py")

# Tạo cửa sổ
window = tk.Tk()
window.title("Phụng - Tiến - Khánh")

# Đặt kích thước cửa sổ và giữ trung tâm màn hình
window_width = 400
window_height = 350
screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()
x = (screen_width - window_width) // 2
y = (screen_height - window_height) // 2
window.geometry(f"{window_width}x{window_height}+{x}+{y}")

# Thay đổi giao diện nút sử dụng ThemedStyle
style = ThemedStyle(window)
style.set_theme('clearlooks')  # Thay đổi theme nếu cần
    
# Tiêu đề
title_label = ttk.Label(window, text="Chọn chức năng bạn muốn nhận diện", font=("Arial", 16))
title_label.pack(pady=10)

# Hình ảnh
image_path = "avatar.jpg_.webp"  # Thay đổi đường dẫn tới hình ảnh của bạn
if os.path.exists(image_path):
    image = Image.open(image_path)
    image.thumbnail((150, 150))
    photo = ImageTk.PhotoImage(image)

    image_label = ttk.Label(window, image=photo)
    image_label.image = photo
    image_label.pack()

# Nút Load Folder
load_folder_button = ttk.Button(window, text="Folder", command=run_app)
load_folder_button.pack(pady=10)

# Nút Zone
zone_button = ttk.Button(window, text="Zone", command=run_zone)
zone_button.pack(pady=10)

# Nút Image
image_button = ttk.Button(window, text="Image", command=run_image)
image_button.pack(pady=10)

# Hiển thị cửa sổ
window.mainloop()
