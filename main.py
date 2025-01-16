import torch


from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from torchvision.models import resnet18, ResNet18_Weights

from torch import Tensor
from torch.nn import Module
from torchvision.transforms.v2 import Transform

from torchvision.transforms.v2 import ColorJitter


from utils import *
from ploting import *

IMAGENET_CLASSES_PATH = "imagenet-simple-labels.json"
INDEX_TO_CLASS = read_imagenet_classes(IMAGENET_CLASSES_PATH)

@torch.no_grad()
def get_max_index(image_tensor, model):
    model.eval()
    result = model(image_tensor)
    return torch.argmax(result).item()

def get_max_indexes(images_tensor, model):
    return [get_max_index(image_tensor, model) for image_tensor in images_tensor]

def gradcam(image_tensor, model, index=None, target_layers=None):
    if target_layers is None:
        target_layers = [model.layer4[-1]]
    
    if index is None:
        index = get_max_index(image_tensor, model)
    targets = [ClassifierOutputTarget(index)] 

    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=image_tensor, targets=targets)
    
    grayscale_cam = grayscale_cam[0, :]
    return show_cam_on_image(tensor_to_np(image_tensor), grayscale_cam, use_rgb=True)

def gradcam_images(images_tensor, model, indexes=None, target_layers=None):
    if indexes is None:
        return [gradcam(image_tensor, model, index=None, target_layers=target_layers) for image_tensor in images_tensor]
    
    return [gradcam(image_tensor, model, index=index, target_layers=target_layers) for image_tensor, index in zip(images_tensor, indexes)]

def run_experiments(
        images_tensor: list[Tensor], 
        indexes: list[list[int]], 
        model: Module, 
        experiments: list[Transform]
    ):
    # returns all images ran in the experiment and their gradcams
    empty_experiment = lambda x: x
    experiments = [empty_experiment] + experiments
    transformed_images_tensor = [[exp(image) for exp in experiments] for image in images_tensor]

    # GradCam Indexes of orginal images
    original_indexes_experiments = [[index for _ in experiments] for index in indexes]
    # GradCam Indexes for new transformed images
    transformed_indexes_experiments = [get_max_indexes(image_exp_col, model) for image_exp_col in transformed_images_tensor]

    cam_images_original_index = [gradcam_images(images_exp, model, indexes=indexes_experiment) for images_exp, indexes_experiment in zip(transformed_images_tensor, original_indexes_experiments)]
    cam_images_transformed_index = [gradcam_images(images_exp, model, indexes=indexes_experiment) for images_exp, indexes_experiment in zip(transformed_images_tensor, transformed_indexes_experiments)]
    
    return transformed_images_tensor, cam_images_original_index, cam_images_transformed_index, transformed_indexes_experiments

def run_experiments_batch(
        images_tensor: list[Tensor], 
        model: Module, 
        experiment_map: dict[str, tuple[list[Transform], dict]]
    ):
    
    indexes = get_max_indexes(images_tensor, model)
    folder_path = create_experiment_folder()

    for experiments_name, experiments_config in experiment_map.items():
        experiments, experiments_params = experiments_config
        experiment_plotter = ExperimentPlotter(len(experiments_params) + 1, INDEX_TO_CLASS)
        all_images_tensor, cam_images_original, cam_images_transformed, transformed_indexes = run_experiments(images_tensor, indexes, model, experiments)
        
        all_images = [[tensor_to_np(image_tensor) for image_tensor in images_tensor] for images_tensor in all_images_tensor]
        
        for images_object, cam_images_original_object, cam_images_transformed_object, index, transformed_index in zip(all_images, cam_images_original, cam_images_transformed, indexes, transformed_indexes):
            experiment_plotter.plot(images_object, cam_images_original_object, cam_images_transformed_object)
            experiment_plotter.set_style(index, transformed_index, experiments_name, experiments_params)
            experiment_plotter.save(folder_path, experiments_name, experiments_params, index)

def main():
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    images_path_list = ["images/fox.jpg", "images/golden.jpeg", "images/puppy.png"]
    images = load_images(images_path_list)
    images_tensor = [pil_to_tensor(image) for image in images]

    blur_experiments_parameters = [{"kernel_size":k} for k in [3, 5, 7]]
    noise_experiments_parameters = [{"sigma":s} for s in [0.02, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.4]]
    jitter_brightness_experiments_parameters = [{"brightness":(b,b)} for b in [0.5, 2.5, 3.5, 4.5, 9]]
    jitter_contrast_experiments_parameters = [{"contrast":(b,b)} for b in [0.1, 0.2, 0.3, 0.4, 0.5, 1.1, 1.2, 1.3, 1.4, 1.5]]
    jitter_saturation_experiments_parameters = [{"saturation":(b,b)} for b in [0.1, 0.2, 0.3, 0.4, 0.5, 1.1, 1.2, 1.3, 1.4, 1.5]]

    experiment_map = {
        # Map from experiment name to a tuple of a list of experiments and their parameters used
        # "Blur":([GaussianBlur(**exp) for exp in blur_experiments_parameters], blur_experiments_parameters),
        # "Noise":([GaussianNoise(**exp) for exp in noise_experiments_parameters], noise_experiments_parameters),
        "Brightness":([ColorJitter(**exp) for exp in jitter_brightness_experiments_parameters], jitter_brightness_experiments_parameters),
        # "Contrast":([ColorJitter(**exp) for exp in jitter_contrast_experiments_parameters], jitter_contrast_experiments_parameters),
        # "Saturation":([ColorJitter(**exp) for exp in jitter_saturation_experiments_parameters], jitter_saturation_experiments_parameters),
    }

    run_experiments_batch(images_tensor, model, experiment_map)

if __name__ == "__main__":
    main()
