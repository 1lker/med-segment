import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import io, morphology, measure
import os

def load_image(image_path):
    """Load an RGB image from file."""
    return cv2.imread(image_path)

def load_mask(mask_path):
    """Load a mask from text file."""
    return np.loadtxt(mask_path, dtype=np.int32)

def load_cells(cells_path):
    """Load cell labels from text file."""
    return np.loadtxt(cells_path, dtype=np.int32)

def save_result(result, output_path):
    """Save a result to file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Handle different types of results
    if isinstance(result, np.ndarray):
        if len(result.shape) == 2:  # 2D array like a mask or segmentation
            np.savetxt(output_path, result, fmt='%d')
        else:  # Something else like cell locations
            np.savetxt(output_path, result, fmt='%d')
    else:
        # Convert to numpy array if it's a list
        result_array = np.array(result)
        np.savetxt(output_path, result_array, fmt='%d')

def calculate_precision_recall_f1(predicted, ground_truth):
    """Calculate precision, recall, and F1 score for pixel-level evaluation."""
    # Convert to binary if not already
    predicted_bin = (predicted > 0).astype(np.int32)
    ground_truth_bin = (ground_truth > 0).astype(np.int32)
    
    # Calculate true positives, false positives, and false negatives
    tp = np.sum((predicted_bin == 1) & (ground_truth_bin == 1))
    fp = np.sum((predicted_bin == 1) & (ground_truth_bin == 0))
    fn = np.sum((predicted_bin == 0) & (ground_truth_bin == 1))
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1

def visualize_mask_results(original, ground_truth, predicted, title, save_path=None):
    """Visualize original image, ground truth mask, and predicted mask."""
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(ground_truth, cmap='gray')
    plt.title('Ground Truth Mask')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(predicted, cmap='gray')
    plt.title(f'Predicted Mask: {title}')
    plt.axis('off')
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.show()

def visualize_cells_with_locations(image, regional_maxima, cell_locations, save_path=None):
    """Visualize cell locations on original image and regional maxima map."""
    plt.figure(figsize=(15, 5))
    
    # Original image
    plt.subplot(1, 3, 1)
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb_img)
    plt.title('Original Image')
    plt.axis('off')
    
    # Regional maxima map
    plt.subplot(1, 3, 2)
    plt.imshow(regional_maxima, cmap='hot')
    plt.title('Regional Maxima Map')
    plt.axis('off')
    
    # Original image with cell locations
    plt.subplot(1, 3, 3)
    marked_img = rgb_img.copy()
    
    # Check if cell_locations is not empty
    if cell_locations is not None and len(cell_locations) > 0:
        for x, y in cell_locations:
            if 0 <= y < marked_img.shape[0] and 0 <= x < marked_img.shape[1]:
                cv2.circle(marked_img, (x, y), 5, (0, 0, 255), -1)
    
    plt.imshow(marked_img)
    plt.title(f'Detected Cell Locations: {len(cell_locations) if cell_locations is not None else 0}')
    plt.axis('off')
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.show()

def visualize_segmentation(image, ground_truth, predicted, save_path=None):
    """Visualize segmentation results with random colors for each cell."""
    plt.figure(figsize=(15, 5))
    
    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    # Ground truth segmentation
    plt.subplot(1, 3, 2)
    # Create a colored visualization
    gt_colored = np.zeros((*ground_truth.shape, 3), dtype=np.uint8)
    for label in np.unique(ground_truth):
        if label == 0:  # Skip background
            continue
        # Generate a random color for this cell
        color = np.random.randint(0, 255, 3)
        gt_colored[ground_truth == label] = color
    plt.imshow(gt_colored)
    plt.title('Ground Truth Segmentation')
    plt.axis('off')
    
    # Predicted segmentation
    plt.subplot(1, 3, 3)
    # Create a colored visualization
    pred_colored = np.zeros((*predicted.shape, 3), dtype=np.uint8)
    for label in np.unique(predicted):
        if label == 0:  # Skip background
            continue
        # Generate a random color for this cell
        color = np.random.randint(0, 255, 3)
        pred_colored[predicted == label] = color
    plt.imshow(pred_colored)
    plt.title(f'Predicted Segmentation: {len(np.unique(predicted))-1} cells')
    plt.axis('off')
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.show()

def calculate_dice_iou(predicted, ground_truth, threshold):
    """Calculate Dice index and IoU for cell segmentation evaluation."""
    # Get unique labels (excluding background)
    pred_labels = np.unique(predicted)[1:] if len(np.unique(predicted)) > 1 else []
    gt_labels = np.unique(ground_truth)[1:] if len(np.unique(ground_truth)) > 1 else []
    
    # Initialize lists to store Dice and IoU scores
    dice_scores = []
    iou_scores = []
    
    # For each predicted cell, find the best matching ground truth cell
    for pred_label in pred_labels:
        pred_mask = (predicted == pred_label)
        
        best_dice = 0
        best_iou = 0
        
        # Find the ground truth cell with highest overlap
        for gt_label in gt_labels:
            gt_mask = (ground_truth == gt_label)
            
            # Calculate intersection and union
            intersection = np.sum(pred_mask & gt_mask)
            union = np.sum(pred_mask | gt_mask)
            
            # Calculate Dice and IoU
            dice = 2 * intersection / (np.sum(pred_mask) + np.sum(gt_mask)) if (np.sum(pred_mask) + np.sum(gt_mask)) > 0 else 0
            iou = intersection / union if union > 0 else 0
            
            # Update best scores
            if dice > best_dice:
                best_dice = dice
                best_iou = iou
        
        # If best match exceeds threshold, add to scores
        if best_dice >= threshold:
            dice_scores.append(best_dice)
            iou_scores.append(best_iou)
    
    # Calculate mean scores if any matches were found
    mean_dice = np.mean(dice_scores) if len(dice_scores) > 0 else 0
    mean_iou = np.mean(iou_scores) if len(iou_scores) > 0 else 0
    
    return mean_dice, mean_iou, len(dice_scores), len(gt_labels)