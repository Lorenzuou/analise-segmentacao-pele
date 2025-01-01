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


def get_segmentation_mask_from_image(img, shape=(1920, 1080)):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    masks = img > 0
    return masks
    



import os
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

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
    
    return m.get_segmentation_mask_from_image(mask, original_mask_img.shape), \
           m.get_segmentation_mask_from_image(original_mask_img, original_mask_img.shape)

def collect_metrics(sessoes_folder: str, original_mask_folder: str) -> Tuple[pd.DataFrame, Dict]:
    """Collect metrics for all images and organize them into a DataFrame."""
    # Initialize data storage
    data_rows = []
    groups_data = {
        "sam": {f"group_{i}": {"metrics": [], "time": []} for i in range(1, 5)},
        "manual": {f"group_{i}": {"metrics": [], "time": []} for i in range(1, 5)}
    }
    
    print(f"Processing folders:")
    print(f"Total folders: {len(os.listdir(sessoes_folder))}")
    print(f"SAM folders: {len([f for f in os.listdir(sessoes_folder) if 'sam' in f])}")
    print(f"Manual folders: {len([f for f in os.listdir(sessoes_folder) if 'manual' in f])}\n")

    progress = 0
    total_folders = len(os.listdir(sessoes_folder))

    for sessao in os.listdir(sessoes_folder):
        # Get group name
        group_name = next((g for g in ["group_1", "group_2", "group_3", "group_4"] if g in sessao), None)
        if not group_name:
            continue
            
        # Skip small sessions
        masked_path = os.path.join(sessoes_folder, sessao, "masked")
        if len(os.listdir(masked_path)) < 50:
            print(f"Skipping {sessao} - insufficient images")
            continue
            
        # Get processing type (sam or manual)
        proc_type = "sam" if "sam" in sessao else "manual"
        print(f"Processing {sessao} ({progress}/{total_folders})")
    
        
        # Process each image
        for image in os.listdir(masked_path):
            original_mask_name = image.replace('.jpg', '_segmentation.png')
            original_mask_path = os.path.join(original_mask_folder, original_mask_name)
            
            if not os.path.exists(original_mask_path):
                continue
                
            # Process images
            mask, original_mask = load_and_process_image(
                os.path.join(masked_path, image),
                original_mask_path
            )
            
            # Calculate metrics
            pixel_acc, iou, dice, nsd = m.calculate_metrics([mask], [original_mask], metrics=['pixel_accuracy', 'iou', 'dice', 'nsd']).values()
            
            # Get processing time
            with open(os.path.join(sessoes_folder, sessao, "time.txt"), "r") as f:
                time = float(f.readlines()[0])
            
            # Store data
            data_rows.append({
                "group": group_name,
                "type": proc_type,
                "pixel_accuracy": pixel_acc,
                "iou": iou,
                "dice": dice,
                "nsd": nsd,
                "time": time
            })
            
            # Store group-specific data
            groups_data[proc_type][group_name]["metrics"].append([pixel_acc, iou, dice, nsd])
            groups_data[proc_type][group_name]["time"].append(time)
    
    return pd.DataFrame(data_rows), groups_data

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

