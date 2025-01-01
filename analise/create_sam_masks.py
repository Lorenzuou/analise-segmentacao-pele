

import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import numpy as np
import cv2
import os 
from pathlib import Path

import pickle as pkl

SESSIONS_FOLDER = 'sessions'
MAIN_IMAGE_FOLDER = 'images'


def rmdir(directory):
    directory = Path(directory)
    for item in directory.iterdir():
        if item.is_dir():
            rmdir(item)
        else:
            item.unlink()
    directory.rmdir()


CHECKPOINT_PATH = "./sam_vit_h_4b8939.pth"
# CHECKPOINT_PATH = "./sam_vit_b_01ec64.pth"

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")
MODEL_TYPE = "vit_h"
# MODEL_TYPE = "vit_b"

model = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
model.to(device=DEVICE)
mask_generator = SamAutomaticMaskGenerator(model,
                                           crop_n_layers=1,
                                           crop_n_points_downscale_factor=2
                                           )

predictor = SamPredictor(model)


def get_embedding(image_name, image_folder):
    # load pkl file
    with open(f'{image_folder}/{image_name}.pkl', 'rb') as f:
        embedding = pkl.load(f)
    return embedding


def get_largest_area(result_dict):
    sorted_result = sorted(result_dict, key=(lambda x: x['area']),
                           reverse=True)
    return sorted_result[0]


def apply_mask(image, mask, color=None):
    print(image.shape, mask.shape)
    # Convert the mask to a 3 channel image
    if color is None:
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    else:
        mask_rgb = np.zeros_like(image)
        mask_rgb[mask > 0] = color

    return mask_rgb


def generate_image(image):
    # Generate segmentation mask
    output_mask = mask_generator.generate(image)
    # get second largest area
    largest_area = get_largest_area(output_mask)
    mask = largest_area['segmentation']

    return mask


def set_image(image):
    predictor.set_image(image)


def generate_images_with_box(image, box):
    box = np.array(box)
    output_mask, scores, logits = predictor.predict(
                            box=box,
                            multimask_output=True,
                        )
    
    mask_input = output_mask[np.argmax(scores), :, :]  # Choose the model's best mask

    image_rgb = np.zeros((image.shape[0], image.shape[1], 3), dtype='uint8')

    color = (255, 255, 255)
    mask = np.where(mask_input, 255, 0).astype('uint8')
    image = apply_mask(image_rgb, mask, color=color)
    return image


def generate_images_get_first(image):
    # Generate segmentation mask
    output_mask, scores, logits = predictor.predict(
                            multimask_output=True,
                        )
    
    mask_input = output_mask[np.argmax(scores), :, :]  # Choose the model's best mask
    image_rgb = np.zeros((image.shape[0], image.shape[1], 3), dtype='uint8')

    color = (255, 255, 255)
    mask = np.where(mask_input, 255, 0).astype('uint8')
    image = apply_mask(image_rgb, mask, color=color)

    return image


def generate_image_with_prompt(image, input_labels, input_points):
    input_points = np.array(input_points)
    
    output_mask, scores, logits = predictor.predict(
                            point_coords=input_points,
                            point_labels=input_labels,
                            multimask_output=True,
                        )
    
    mask_input = output_mask[np.argmax(scores), :, :]  # Choose the model's best mask

    image_rgb = np.zeros((image.shape[0], image.shape[1], 3), dtype='uint8')

    color = (255, 255, 255)
    mask = np.where(mask_input, 255, 0).astype('uint8')
    image = apply_mask(image_rgb, mask, color=color)

    return image


if __name__ == '__main__':

    groups = ["group_1", "group_2", "group_3", "group_4"]

    for group in groups:
        images_folder = f"./images/{group}"
        embedding_folder = images_folder

        output_folder = f"./sam_generated_{group}"
        os.makedirs(output_folder, exist_ok=True)

        for image_file in os.listdir(images_folder):
            if image_file.endswith('.jpg'):
                file_name = os.path.splitext(image_file)[0]
                image_cv2 = cv2.imread(f'{images_folder}/{image_file}')  

                predictor.reset_image()
                image_obj = get_embedding(file_name, embedding_folder)
                predictor.features = image_obj['embedd']
                predictor.original_size = image_obj['original_size']
                predictor.input_size = image_obj['input_size']
                predictor.is_image_set = True

                image_masked = generate_images_get_first(image_cv2)
                image_masked = cv2.bitwise_not(image_masked)

                cv2.imwrite(f'{output_folder}/{image_file}', image_masked)