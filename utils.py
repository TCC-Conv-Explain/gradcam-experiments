from torchvision.transforms.v2 import ToTensor, ToPILImage
from PIL import Image

import numpy as np
import json
from datetime import datetime
import os

def read_imagenet_classes(file_path: str) -> dict[int, str]:
    with open(file_path, "r") as f:
        class_list = json.load(f)
        index_to_class_name = {index:class_name.title() for index, class_name in enumerate(class_list)}
    
    return index_to_class_name

def load_image(path, resize=None):
    image = Image.open(path)
    if resize is not None:
        image = image.resize(resize)
    
    return image

def load_images(path_list, resize=None):
    return [load_image(path, resize=resize) for path in path_list]

def show_image_np(image_np):
    return Image.fromarray(image_np)

def pil_to_tensor(image):
    to_tensor = ToTensor()
    return to_tensor(image).unsqueeze(0)

def tensor_to_pil(image_tensor):
    to_pil = ToPILImage()
    return to_pil(image_tensor.squeeze(0))

def tensor_to_np(image_tensor):
    image = tensor_to_pil(image_tensor)
    image_np = np.array(image).astype(np.float32)
    return image_np / 255

def create_experiment_folder():
    # Get the current time and format it as a string
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Define the folder name with the timestamp
    folder_name = f"experiment_{timestamp}"
    
    current_folder_path = os.getcwd()
    folder_path = os.path.join(current_folder_path, folder_name)
    
    # Create the folder
    os.makedirs(folder_path, exist_ok=True)

    return folder_path