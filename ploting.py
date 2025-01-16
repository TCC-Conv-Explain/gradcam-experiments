from matplotlib import pyplot as plt
from utils import *

from typing import Any

class ExperimentPlotter:

    ROW_TITLES = ["Images", "Original Class", "Top Class"]

    def __init__(
            self,
            cols,
            index_to_class
        ):
        
        rows= 3
        self.fig, self.axes = plt.subplots(rows, cols, figsize=self._get_figsize(rows, cols))

        self.index_to_class = index_to_class

    def _get_figsize(self, rows, cols):
        return (2 * cols + 5, 7)

    def set_style(self, 
                  original_index: int, 
                  transformed_indexes: list[int],
                  experiments_name: str,
                  experiments_params: list[dict[str, Any]]
                ):
        
        self.fig.suptitle(f"{experiments_name}: {self.index_to_class[original_index]} [{original_index}]")
        
        for row_idx, title in enumerate(self.ROW_TITLES):
            self.axes[row_idx, 0].set_ylabel(title, rotation=90, size="large")
        
        for ax in self.axes.flat:
            ax.set_yticks([])
            ax.set_xticks([])

        self.axes[0, 0].set_title("Original")
        
        for i, experiment_params in enumerate(experiments_params, start=1):
            self.axes[0, i].set_title(self._format_experiments_title(experiment_params))

        for i, index in enumerate(transformed_indexes):
            self.axes[2, i].set_title(f"[{index}] {self.index_to_class[index]}")

    
    def _format_experiments_title(self, experiment_params):
        return ", ".join(f"{key}: {value}" for key, value in experiment_params.items())

    def plot(
            self, 
            images,
            cam_images_original,
            cam_images_transformed
        ):
        for i, image in enumerate(images):
            self.axes[0, i].imshow(image)

        for i, cam_image_original in enumerate(cam_images_original):
            self.axes[1, i].imshow(cam_image_original)

        for i, cam_image_transformed in enumerate(cam_images_transformed):
            self.axes[1, i].imshow(cam_image_transformed)

    def save(self, 
             folder_path,
             experiments_name, 
             experiment_params, 
             index
            ):
        plt.tight_layout()

        file_path = self._get_file_path(folder_path, experiments_name, experiment_params, index)
        self.fig.savefig(file_path)

    def _get_file_path(self, 
             folder_path,
             experiments_name, 
             experiment_params, 
             index
            ):
        
        experiments_params_used = [list(exp_params.keys()) for exp_params in experiment_params]
        experiments_params_used = [exp_param_name for exp_param_name for exp_params in experiments_params_used]
        file_name = f"{experiments_name}({','.join(experiments_params_used)}): {self.index_to_class[index]} [{index}].png"
        return os.path.join(folder_path, file_name)