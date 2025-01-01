import os
import pickle
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from typing import Dict, List, Tuple
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
import multiprocessing as mp


def pixel_accuracy(pred, target):
    correct = (pred == target).sum()
    total = pred.size
    return correct / total

def intersection_over_union(pred, target):
    intersection = (pred & target).sum()
    union = (pred | target).sum()
    return intersection / union

def dice_coefficient(pred, target):
    intersection = (pred & target).sum()
    return 2 * intersection / (pred.sum() + target.sum())


def nsd_coefficient(pred, ref, tau = 1):
    # Get the surface (edges) of both segmentations
    pred_surface = pred - distance_transform_edt(1 - pred) > 0
    ref_surface = ref - distance_transform_edt(1 - ref) > 0

    # Calculate the distance transform for each segmentation
    dist_pred_to_ref = distance_transform_edt(1 - ref)
    dist_ref_to_pred = distance_transform_edt(1 - pred)

    # Check distances for each surface
    within_tau_pred = dist_pred_to_ref[pred_surface] <= tau
    within_tau_ref = dist_ref_to_pred[ref_surface] <= tau

    # Compute the NSD
    nsd_pred = np.sum(within_tau_pred) / np.sum(pred_surface)
    nsd_ref = np.sum(within_tau_ref) / np.sum(ref_surface)
    
    nsd = 0.5 * (nsd_pred + nsd_ref)

    return nsd

def calculate_metrics(pred_list, target_list, metrics):
    if metrics is None:
        metrics = ['pixel_accuracy', 'iou', 'dice', 'nsd']
        
    results = {metric: 0 for metric in metrics}
    n = len(pred_list)  # Assuming pred_list and target_list are of the same length

    for pred, target in zip(pred_list, target_list):
        if 'pixel_accuracy' in metrics:
            results['pixel_accuracy'] += pixel_accuracy(pred, target)
        if 'iou' in metrics:
            results['iou'] += intersection_over_union(pred, target)
        if 'dice' in metrics:
            results['dice'] += dice_coefficient(pred, target)
        if 'nsd' in metrics:
            results['nsd'] += nsd_coefficient(pred, target)

    # Calculate the mean for each metric
    for metric in results:
        results[metric] /= n

    return results




# Define Erosion Function
def apply_erosion(image, kernel_size=(5, 5), iterations=1):
    """
    Apply erosion to an image using a specified kernel size and number of iterations.
    
    Parameters:
    - image: The input image on which erosion will be applied.
    - kernel_size: The size of the kernel to be used for erosion.
    - iterations: The number of times erosion is applied.
    
    Returns:
    - eroded_image: The eroded image.
    ""
    
    """
    est = cv2.getStructuringElement(cv2.MORPH_ERODE, kernel_size)
    eroded_image = cv2.erode(image, est, iterations=iterations)
    return eroded_image


def erode_all_masks(masks: List[np.ndarray], kernel_size=(5, 5), iterations=1) -> List[np.ndarray]:
    """Apply erosion to all masks."""
    return [apply_erosion(mask, kernel_size, iterations) for mask in masks]

def apply_dilation(image, kernel_size=(5, 5), iterations=1):
    """
    Apply dilation to an image using a specified kernel size and number of iterations.
    
    Parameters:
    - image: The input image on which dilation will be applied.
    - kernel_size: The size of the kernel to be used for dilation.
    - iterations: The number of times dilation is applied.
    
    Returns:
    - dilated_image: The dilated image.
    ""
    
    """
    est = cv2.getStructuringElement(cv2.MORPH_DILATE, kernel_size)
    dilated_image = cv2.dilate(image, est, iterations=iterations)
    return dilated_image

def dilate_all_masks(masks: List[np.ndarray], kernel_size=(5, 5), iterations=1) -> List[np.ndarray]:
    """Apply dilation to all masks."""
    return [apply_dilation(mask, kernel_size, iterations) for mask in masks]


def apply_opening(image, kernel_size=(5, 5), iterations=1):
    """
    Apply opening to an image using a specified kernel size and number of iterations.
    
    Parameters:
    - image: The input image on which opening will be applied.
    - kernel_size: The size of the kernel to be used for opening.
    - iterations: The number of times opening is applied.
    
    Returns:
    - opened_image: The opened image.
    ""
    
    """
    est = cv2.getStructuringElement(cv2.MORPH_OPEN, kernel_size)
    opened_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, est, iterations=iterations)
    return opened_image

def open_all_masks(masks: List[np.ndarray], kernel_size=(5, 5), iterations=1) -> List[np.ndarray]:
    """Apply opening to all masks."""
    return [apply_opening(mask, kernel_size, iterations) for mask in masks]

def apply_closing(image, kernel_size=(5, 5), iterations=1):
    """
    Apply closing to an image using a specified kernel size and number of iterations.
    
    Parameters:
    - image: The input image on which closing will be applied.
    - kernel_size: The size of the kernel to be used for closing.
    - iterations: The number of times closing is applied.
    
    Returns:
    - closed_image: The closed image.
    ""
    
    """
    est = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    closed_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, est, iterations=iterations)
    return closed_image

def close_all_masks(masks: List[np.ndarray], kernel_size=(5, 5), iterations=1) -> List[np.ndarray]:
    """Apply closing to all masks."""
    return [apply_closing(mask, kernel_size, iterations) for mask in masks]


def load_and_process_image(mask_path: str, original_mask_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load and preprocess mask images."""
    mask = cv2.imread(mask_path)
    original_mask_img = cv2.imread(original_mask_path)
    
    # Resize logic
    if original_mask_img.shape[:2] != mask.shape[:2]:
        target_shape = (
            min(original_mask_img.shape[1], mask.shape[1]),
            min(original_mask_img.shape[0], mask.shape[0])
        )
        original_mask_img = cv2.resize(original_mask_img, target_shape[::-1])
        mask = cv2.resize(mask, target_shape[::-1])
    
    return mask, original_mask_img


def get_segmentation_mask_from_image(img, shape=(1920, 1080)):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    masks = img > 0
    return masks
    

# def collect_metrics(sessoes_folder: str, original_mask_folder: str) -> Tuple[pd.DataFrame, Dict]:
#     """Collect metrics for all images and organize them into a DataFrame."""
#     # Initialize data storage
#     data_rows = []
#     groups_data = {
#         "sam": {f"group_{i}": {"metrics": [], "time": []} for i in range(1, 5)},
#         "manual": {f"group_{i}": {"metrics": [], "time": []} for i in range(1, 5)}
#     }
    
#     print(f"Processing folders:")
#     print(f"Total folders: {len(os.listdir(sessoes_folder))}")
#     print(f"SAM folders: {len([f for f in os.listdir(sessoes_folder) if 'sam' in f])}")
#     print(f"Manual folders: {len([f for f in os.listdir(sessoes_folder) if 'manual' in f])}\n")

#     progress = 0
#     total_folders = len(os.listdir(sessoes_folder))

#     for sessao in tqdm(os.listdir(sessoes_folder)):
#         # Get group name
#         group_name = next((g for g in ["group_1", "group_2", "group_3", "group_4"] if g in sessao), None)
#         if not group_name:
#             continue
            
#         # Skip small sessions
#         masked_path = os.path.join(sessoes_folder, sessao, "masked")
#         if len(os.listdir(masked_path)) < 50:
#             print(f"Skipping {sessao} - insufficient images")
#             continue
            
#         # Get processing type (sam or manual)
#         proc_type = "sam" if "sam" in sessao else "manual"
#         print(f"Processing {sessao} ({progress}/{total_folders})")
    
        
#         # Process each image
#         for image in tqdm(os.listdir(masked_path)):
#             original_mask_name = image.replace('.jpg', '_segmentation.png')
#             original_mask_path = os.path.join(original_mask_folder, original_mask_name)

            
            
#             if not os.path.exists(original_mask_path):
#                 continue
                
#             # Process images
#             mask, original_mask = load_and_process_image(
#                 os.path.join(masked_path, image),
#                 original_mask_path
#             )

            
#             # Calculate metrics
#             pixel_acc, iou, dice, nsd = calculate_metrics([get_segmentation_mask_from_image(mask)],
#                                                           [get_segmentation_mask_from_image(original_mask)],
#                                                           metrics=['pixel_accuracy', 'iou', 'dice', 'nsd']).values()


#             # Apply morphological operations
#             eroded_mask = apply_erosion(mask, kernel_size=(5, 5), iterations=1)
#             dilated_mask = apply_dilation(mask, kernel_size=(5, 5), iterations=1)
#             opened_mask = apply_opening(mask, kernel_size=(5, 5), iterations=1)
#             closed_mask = apply_closing(mask, kernel_size=(5, 5), iterations=1)
            
#             # Calculate metrics for morphological operations
#             eroded_metrics = calculate_metrics([eroded_mask], [original_mask], metrics=['pixel_accuracy', 'iou', 'dice', 'nsd'])
#             dilated_metrics = calculate_metrics([dilated_mask], [original_mask], metrics=['pixel_accuracy', 'iou', 'dice', 'nsd'])
#             opened_metrics = calculate_metrics([opened_mask], [original_mask], metrics=['pixel_accuracy', 'iou', 'dice', 'nsd'])
#             closed_metrics = calculate_metrics([closed_mask], [original_mask], metrics=['pixel_accuracy', 'iou', 'dice', 'nsd'])
            
            
            
#             # Get processing time
#             with open(os.path.join(sessoes_folder, sessao, "time.txt"), "r") as f:
#                 time = float(f.readlines()[0])
            
#             # Store data
#             data_rows.append({
#                 "group": group_name,
#                 "type": proc_type,
#                 "pixel_accuracy": pixel_acc,
#                 "iou": iou,
#                 "dice": dice,
#                 "nsd": nsd,
#                 "eroded_pixel_accuracy": eroded_metrics['pixel_accuracy'],
#                 "eroded_iou": eroded_metrics['iou'],
#                 "eroded_dice": eroded_metrics['dice'],
#                 "eroded_nsd": eroded_metrics['nsd'],
#                 "dilated_pixel_accuracy": dilated_metrics['pixel_accuracy'],
#                 "dilated_iou": dilated_metrics['iou'],
#                 "dilated_dice": dilated_metrics['dice'],
#                 "dilated_nsd": dilated_metrics['nsd'],
#                 "opened_pixel_accuracy": opened_metrics['pixel_accuracy'],
#                 "opened_iou": opened_metrics['iou'],
#                 "opened_dice": opened_metrics['dice'],
#                 "opened_nsd": opened_metrics['nsd'],
#                 "closed_pixel_accuracy": closed_metrics['pixel_accuracy'],
#                 "closed_iou": closed_metrics['iou'],
#                 "closed_dice": closed_metrics['dice'],
#                 "closed_nsd": closed_metrics['nsd'],
#                 "time": time,
#                 "image_name": image
#             })
            
#             # Store group-specific data
#             groups_data[proc_type][group_name]["metrics"].append([pixel_acc, iou, dice, nsd])
#             groups_data[proc_type][group_name]["time"].append(time)
#         progress += 1
#     return pd.DataFrame(data_rows), groups_data


def process_single_image(image_path: Path, original_mask_folder: str) -> Tuple[Dict, float]:
    """Process a single image and return metrics and processing time.
    
    Args:
        image_path: Path to the input image
        original_mask_folder: Path to the folder containing original masks
    
    Returns:
        Tuple containing metrics dictionary and processing time
    """
    image_name = image_path.name
    original_mask_name = image_name.replace('.jpg', '_segmentation.png')
    original_mask_path = Path(original_mask_folder) / original_mask_name

    if not original_mask_path.exists():
        return None, None
    
    # Load and preprocess images
    mask, original_mask = load_and_process_image(str(image_path), str(original_mask_path))

    # Calculate metrics
    metrics = calculate_metrics([get_segmentation_mask_from_image(mask)],
                                [get_segmentation_mask_from_image(original_mask)],
                                metrics=['pixel_accuracy', 'iou', 'dice', 'nsd'])
    
    # Apply morphological operations
    eroded_mask = apply_erosion(mask, kernel_size=(5, 5), iterations=1)
    dilated_mask = apply_dilation(mask, kernel_size=(5, 5), iterations=1)
    opened_mask = apply_opening(mask, kernel_size=(5, 5), iterations=1)
    closed_mask = apply_closing(mask, kernel_size=(5, 5), iterations=1)

    # Calculate metrics for morphological operations
    eroded_metrics = calculate_metrics([eroded_mask], [original_mask], metrics=['pixel_accuracy', 'iou', 'dice', 'nsd'])
    dilated_metrics = calculate_metrics([dilated_mask], [original_mask], metrics=['pixel_accuracy', 'iou', 'dice', 'nsd'])
    opened_metrics = calculate_metrics([opened_mask], [original_mask], metrics=['pixel_accuracy', 'iou', 'dice', 'nsd'])
    closed_metrics = calculate_metrics([closed_mask], [original_mask], metrics=['pixel_accuracy', 'iou', 'dice', 'nsd'])

    # Get processing time
    with open(image_path.parent / "time.txt", "r") as f:
        processing_time = float(f.readlines()[0])

    
    # Process image here
    # Add your image processing logic
    return metrics, processing_time



def collect_metrics_optimized(sessoes_folder: str, original_mask_folder: str) -> Tuple[pd.DataFrame, Dict]:
    """Optimized version of collect_metrics using parallel processing."""
    sessoes_path = Path(sessoes_folder)
    
    # Pre-filter valid directories
    valid_sessions = {
        "sam": [],
        "manual": []
    }
    
    # Initialize data storage with numpy arrays
    groups_data = {
        "sam": {f"group_{i}": {"metrics": [], "time": []} for i in range(1, 5)},
        "manual": {f"group_{i}": {"metrics": [], "time": []} for i in range(1, 5)}
    }
    
    # Pre-filter directories
    for sessao in sessoes_path.iterdir():
        if not sessao.is_dir():
            continue
            
        masked_path = sessao / "masked"
        if not masked_path.exists() or len(list(masked_path.glob("*.jpg"))) < 50:
            continue
            
        proc_type = "sam" if "sam" in sessao.name else "manual"
        group_name = next((g for g in ["group_1", "group_2", "group_3", "group_4"] 
                          if g in sessao.name), None)
        
        if group_name:
            valid_sessions[proc_type].append((sessao, group_name))

    image_paths = []

    # Process images in parallel
    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
                # Process batch of images
                results = list(executor.map(
                    lambda x: process_single_image(x, original_mask_folder),
                    image_paths,
                    chunksize=10
                ))

                # Process batch of images
                results = list(executor.map(process_func, 
                                         [(ip, original_mask_folder) for ip in image_paths], 
                                         chunksize=10))
                
                # Aggregate results
                metrics, times = zip(*results)
                groups_data[proc_type][group_name]["metrics"].extend(metrics)
                groups_data[proc_type][group_name]["time"].extend(times)

    return pd.DataFrame(groups_data), groups_data



def visualize_overall_metrics(df: pd.DataFrame):
    """Create visualizations for overall metrics comparison."""
    plt.figure(figsize=(15, 10))
    
    # Create boxplots for each metric
    metrics = ['pixel_accuracy', 'iou', 'dice']
    for i, metric in enumerate(metrics, 1):
        plt.subplot(2, 2, i)
        sns.boxplot(data=df, x='type', y=metric)
        plt.title(f'{metric.replace("_", " ").title()} Distribution')
        plt.xlabel('Processing Type')
        plt.ylabel(metric.replace("_", " ").title())
    
    # Create time comparison
    plt.subplot(2, 2, 4)
    sns.boxplot(data=df, x='type', y='time')
    plt.title('Processing Time Distribution')
    plt.xlabel('Processing Type')
    plt.ylabel('Time (seconds)')
    
    plt.tight_layout()
    plt.show()

def visualize_group_metrics(df: pd.DataFrame):
    """Create visualizations for group-specific metrics."""
    metrics = ['pixel_accuracy', 'iou', 'dice', 'time']
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Metrics by Group and Processing Type')
    
    for i, metric in enumerate(metrics):
        ax = axes[i // 2, i % 2]
        sns.boxplot(data=df, x='group', y=metric, hue='type', ax=ax)
        ax.set_title(f'{metric.replace("_", " ").title()}')
        ax.set_xlabel('Group')
        ax.set_ylabel(metric.replace("_", " ").title())
    
    plt.tight_layout()
    plt.show()

def display_summary_stats(df: pd.DataFrame):
    """Display summary statistics in a clean table format."""
    # Overall statistics
    overall_stats = df.groupby('type')[['pixel_accuracy', 'iou', 'dice', 'time']].agg(['mean', 'std', 'count'])
    
    # Group statistics
    group_stats = df.groupby(['type', 'group'])[['pixel_accuracy', 'iou', 'dice', 'time']].agg(['mean', 'std', 'count'])
    
    return overall_stats, group_stats


def main():
    sessoes_folder = "./sessoes"
    original_mask_folder = 'pad_segmentation_all/all-mask'

    # Collect and process data
    # df, groups_data = collect_metrics(sessoes_folder, original_mask_folder)

    df, groups_data = collect_metrics_optimized(sessoes_folder, original_mask_folder)

    df.to_csv("metrics.csv", index=False)
    with open("groups_data.pkl", "wb") as f:
        pickle.dump(groups_data, f)

    print("Metrics collected and saved to data/metrics.csv")
    print("Groups data saved to data/groups_data.pkl")


if __name__ == "__main__":
    main()
