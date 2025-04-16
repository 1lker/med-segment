import numpy as np
import cv2
from scipy import ndimage
from skimage import feature, morphology, measure, filters
import matplotlib.pyplot as plt
import os
from src.util import save_result

def find_cell_locations(image, foreground_mask, method='combined', 
                      min_distance=8, validation=True):
    """
    Find approximate locations of cells in CAMA-1 cell microscopy images.
    
    This function implements the approach suggested in the assignment hint -
    detecting white boundaries between cells and calculating distances from
    these boundaries to find cell centers.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input RGB image
    foreground_mask : numpy.ndarray
        Binary mask where foreground pixels are 1 and background pixels are 0
    method : str
        Method to use ('combined', 'boundary_distance', 'intensity_based')
    min_distance : int
        Minimum distance between cell centers
    validation : bool
        Whether to validate detected cells
    
    Returns:
    --------
    tuple
        (cell_locations, regional_maxima_map)
        cell_locations: numpy.ndarray of shape (n, 2) with (x, y) coordinates
        regional_maxima_map: numpy.ndarray of the same size as input mask
    """
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Initialize regional maxima map
    regional_maxima = np.zeros_like(foreground_mask, dtype=np.float32)
    
    # Constants
    MAX_BINARY_VALUE = 255
    
    if method == 'combined':
        # This robust method combines both boundary-based and intensity-based approaches
        # to maximize detection in all types of cell images
        
        # Step 1: Enhance contrast to better detect features
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Step 2: Detect white boundaries between cells 
        # Using adaptive threshold to work with different image conditions
        mean_intensity = np.mean(enhanced[foreground_mask > 0])
        std_intensity = np.std(enhanced[foreground_mask > 0])
        white_threshold = min(mean_intensity + 1.5 * std_intensity, 220)
        
        _, white_boundaries = cv2.threshold(enhanced, white_threshold, MAX_BINARY_VALUE, cv2.THRESH_BINARY)
        
        # Step 3: Invert the white boundary mask (boundaries become 0)
        inv_boundaries = cv2.bitwise_not(white_boundaries)
        
        # Step 4: Restrict to foreground
        foreground = (foreground_mask * MAX_BINARY_VALUE).astype(np.uint8)
        dt_input = cv2.bitwise_and(foreground, inv_boundaries)
        
        # Step 5: Compute distance transform from boundaries
        dist_transform = cv2.distanceTransform(dt_input, cv2.DIST_L2, 3)
        
        # Step 6: Apply Gaussian smoothing to reduce noise
        # Use adaptive sigma based on the size of structures in the image
        mean_dist = np.mean(dist_transform[dist_transform > 0])
        gauss_sigma = max(1.0, mean_dist / 5.0)
        dist_smooth = cv2.GaussianBlur(dist_transform, (0, 0), gauss_sigma)
        
        # Step 7: Detect local maxima in the distance transform
        # Automatic parameter selection based on image statistics
        mean_dist = np.mean(dist_smooth[dist_smooth > 0])
        std_dist = np.std(dist_smooth[dist_smooth > 0])
        min_dist_threshold = max(1.0, mean_dist - std_dist)
        
        # Get boundary-based locations
        boundary_coordinates = feature.peak_local_max(
            dist_smooth, 
            min_distance=min_distance,
            threshold_abs=min_dist_threshold,
            exclude_border=False
        )
        
        boundary_locations = [(int(x), int(y)) for y, x in boundary_coordinates]
        
        # Step 8: Also use intensity-based approach for robustness
        # Invert the image to make darker regions (cell centers) become peaks
        inverted = 255 - enhanced
        
        # Mask out background
        masked_inverted = np.zeros_like(inverted, dtype=float)
        masked_inverted[foreground_mask > 0] = inverted[foreground_mask > 0]
        
        # Apply Gaussian blur to reduce noise
        blurred = ndimage.gaussian_filter(masked_inverted, sigma=1.0)
        
        # Detect local maxima with adaptive threshold
        mean_intensity = np.mean(blurred[blurred > 0])
        intensity_threshold = mean_intensity * 0.8
        
        intensity_coordinates = feature.peak_local_max(
            blurred,
            min_distance=min_distance,
            threshold_abs=intensity_threshold,
            exclude_border=False
        )
        
        intensity_locations = [(int(x), int(y)) for y, x in intensity_coordinates]
        
        # Step 9: Combine both approaches
        all_locations = set()
        
        # Add boundary-based locations
        for x, y in boundary_locations:
            all_locations.add((x, y))
        
        # Add intensity-based locations, avoiding duplicates
        for x, y in intensity_locations:
            too_close = False
            for px, py in all_locations:
                dist = np.sqrt((x - px)**2 + (y - py)**2)
                if dist < min_distance:
                    too_close = True
                    break
            
            if not too_close:
                all_locations.add((x, y))
        
        # Step 10: Final list of cell locations
        cell_locations = list(all_locations)
        
        # Create visualization map
        if np.max(dist_smooth) > 0:
            normalized_dist = dist_smooth / np.max(dist_smooth)
            regional_maxima[foreground_mask > 0] = normalized_dist[foreground_mask > 0] * MAX_BINARY_VALUE
    
    elif method == 'boundary_distance':
        # This implements the white boundary approach from the assignment hint
        
        # Step 1: Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Step 2: Detect white boundaries (adaptive threshold)
        mean_intensity = np.mean(enhanced[foreground_mask > 0])
        std_intensity = np.std(enhanced[foreground_mask > 0])
        white_threshold = min(mean_intensity + 1.5 * std_intensity, 220)
        
        _, white_boundaries = cv2.threshold(enhanced, white_threshold, MAX_BINARY_VALUE, cv2.THRESH_BINARY)
        
        # Step 3: Invert the white boundary mask
        inv_boundaries = cv2.bitwise_not(white_boundaries)
        
        # Step 4: Restrict to foreground
        foreground = (foreground_mask * MAX_BINARY_VALUE).astype(np.uint8)
        dt_input = cv2.bitwise_and(foreground, inv_boundaries)
        
        # Step 5: Compute distance transform
        dist_transform = cv2.distanceTransform(dt_input, cv2.DIST_L2, 3)
        
        # Step 6: Apply Gaussian smoothing (adaptive sigma)
        mean_dist = np.mean(dist_transform[dist_transform > 0])
        gauss_sigma = max(1.0, mean_dist / 5.0)
        dist_smooth = cv2.GaussianBlur(dist_transform, (0, 0), gauss_sigma)
        
        # Step 7: Detect local maxima (adaptive threshold)
        mean_dist = np.mean(dist_smooth[dist_smooth > 0])
        std_dist = np.std(dist_smooth[dist_smooth > 0])
        min_dist_threshold = max(1.0, mean_dist - std_dist)
        
        coordinates = feature.peak_local_max(
            dist_smooth, 
            min_distance=min_distance,
            threshold_abs=min_dist_threshold,
            exclude_border=False
        )
        
        # Convert to (x, y) format
        cell_locations = [(int(x), int(y)) for y, x in coordinates]
        
        # Create visualization map
        if np.max(dist_smooth) > 0:
            normalized_dist = dist_smooth / np.max(dist_smooth)
            regional_maxima[foreground_mask > 0] = normalized_dist[foreground_mask > 0] * MAX_BINARY_VALUE
    
    elif method == 'intensity_based':
        # Step 1: Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Step 2: Invert so darker regions (cell centers) become peaks
        inverted = 255 - enhanced
        
        # Step 3: Mask out background
        masked_inverted = np.zeros_like(inverted, dtype=float)
        masked_inverted[foreground_mask > 0] = inverted[foreground_mask > 0]
        
        # Step 4: Apply Gaussian blur to reduce noise
        blurred = ndimage.gaussian_filter(masked_inverted, sigma=1.0)
        
        # Step 5: Detect local maxima (adaptive threshold)
        mean_intensity = np.mean(blurred[blurred > 0])
        intensity_threshold = mean_intensity * 0.8
        
        coordinates = feature.peak_local_max(
            blurred,
            min_distance=min_distance,
            threshold_abs=intensity_threshold,
            exclude_border=False
        )
        
        # Convert to (x, y) format
        cell_locations = [(int(x), int(y)) for y, x in coordinates]
        
        # Create visualization map
        if np.max(blurred) > 0:
            normalized = blurred / np.max(blurred)
            regional_maxima[foreground_mask > 0] = normalized[foreground_mask > 0] * MAX_BINARY_VALUE
    
    else:
        # Default to combined method if unknown method is specified
        print(f"Warning: Unknown method '{method}'. Using combined method instead.")
        return find_cell_locations(image, foreground_mask, method='combined', 
                                  min_distance=min_distance, validation=validation)
    
    # If validation is enabled, filter out some cell locations
    if validation and len(cell_locations) > 0:
        validated_locations = []
        
        # Calculate cell size statistics from the foreground mask
        labeled_regions, num_regions = ndimage.label(foreground_mask)
        region_sizes = []
        for i in range(1, num_regions + 1):
            region_size = np.sum(labeled_regions == i)
            region_sizes.append(region_size)
        
        # Calculate average cell size
        avg_cell_size = 100  # Default value
        if len(region_sizes) > 0:
            avg_cell_size = np.median(region_sizes)  # Median is more robust than mean
        
        # Estimate typical cell radius
        estimated_radius = np.sqrt(avg_cell_size / np.pi)
        validation_radius = estimated_radius * 0.5  # Use 50% of radius for validation
        
        # For each cell location
        for x, y in cell_locations:
            # Make sure it's within the foreground mask
            if 0 <= y < foreground_mask.shape[0] and 0 <= x < foreground_mask.shape[1]:
                if foreground_mask[y, x] > 0:
                    # Check that this point is not too close to an already validated cell
                    too_close = False
                    for vx, vy in validated_locations:
                        # Calculate distance to other validated cells
                        dist = np.sqrt((x - vx) ** 2 + (y - vy) ** 2)
                        if dist < validation_radius:
                            too_close = True
                            break
                    
                    if not too_close:
                        validated_locations.append((x, y))
        
        # If we filtered out too many cells, try again with smaller radius
        if len(validated_locations) < 0.5 * len(cell_locations):
            validation_radius *= 0.7
            validated_locations = []
            
            for x, y in cell_locations:
                if 0 <= y < foreground_mask.shape[0] and 0 <= x < foreground_mask.shape[1]:
                    if foreground_mask[y, x] > 0:
                        too_close = False
                        for vx, vy in validated_locations:
                            dist = np.sqrt((x - vx) ** 2 + (y - vy) ** 2)
                            if dist < validation_radius:
                                too_close = True
                                break
                        
                        if not too_close:
                            validated_locations.append((x, y))
        
        # Update cell locations
        cell_locations = validated_locations
    
    print(f"Number of cells detected: {len(cell_locations)}")
    return np.array(cell_locations), regional_maxima.astype(np.uint8)

def evaluate_cell_locations(image_path, mask_path, cells_path, output_path, 
                           visualization_path=None, method='combined',
                           min_distance=8, validation=True):
    """
    Evaluate the cell location finding algorithm on a single image.
    
    Parameters:
    -----------
    image_path : str
        Path to input image
    mask_path : str
        Path to foreground mask (from part 1)
    cells_path : str
        Path to ground truth cell annotations
    output_path : str
        Path to save predicted cell locations
    visualization_path : str
        Path to save visualization
    method : str
        Method to use for cell detection
    min_distance : int
        Minimum distance between cell centers
    validation : bool
        Whether to validate detected cells
    
    Returns:
    --------
    dict
        Dictionary with evaluation metrics
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load data
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
        
    foreground_mask = np.loadtxt(mask_path, dtype=np.int32)
    gold_cells = np.loadtxt(cells_path, dtype=np.int32)
    
    # Find cell locations
    cell_locations, regional_maxima = find_cell_locations(
        image, foreground_mask, 
        method=method,
        min_distance=min_distance,
        validation=validation
    )
    
    # Save cell locations
    save_result(cell_locations, output_path)
    
    # Calculate metrics using the matched gold standard approach
    # Count the number of unique cells in gold standard (excluding background)
    unique_gold_cells = np.unique(gold_cells)
    unique_gold_cells = unique_gold_cells[unique_gold_cells > 0]  # Remove background (0)
    num_gold_cells = len(unique_gold_cells)
    
    # For each detected cell, find which gold standard cell it belongs to
    cell_matches = {}  # Maps gold cell ID to list of detected cells that match it
    unmatched = []  # Cells that don't match any gold cell
    
    for x, y in cell_locations:
        # Check if coordinates are within gold cells image boundaries
        if 0 <= y < gold_cells.shape[0] and 0 <= x < gold_cells.shape[1]:
            gold_id = gold_cells[y, x]
            if gold_id > 0:  # If this matches a gold cell
                if gold_id not in cell_matches:
                    cell_matches[gold_id] = []
                cell_matches[gold_id].append((x, y))
            else:
                unmatched.append((x, y))
        else:
            unmatched.append((x, y))
    
    # Count true positives: gold standard cells that receive at least one predicted centroid
    tp = len(cell_matches)
    
    # Count false positives (detected cells that don't match any gold cell)
    fp = len(unmatched)
    
    # Count false negatives (gold cells without any match)
    fn = num_gold_cells - tp
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Visualize results
    if visualization_path:
        # Create visualization directory if needed
        os.makedirs(os.path.dirname(visualization_path), exist_ok=True)
        
        # Create visualization showing cell locations on original image
        plt.figure(figsize=(15, 5))
        
        # Original image
        plt.subplot(1, 3, 1)
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(rgb_img)
        plt.title('Original Image')
        plt.axis('off')
        
        # Regional maxima map using jet colormap for better visibility
        plt.subplot(1, 3, 2)
        plt.imshow(regional_maxima, cmap='jet')
        plt.title('Distance Transform')
        plt.axis('off')
        
        # Original image with cell locations
        plt.subplot(1, 3, 3)
        marked_img = rgb_img.copy()
        
        # Check if cell_locations is not empty
        if cell_locations.size > 0:
            for x, y in cell_locations:
                if 0 <= y < marked_img.shape[0] and 0 <= x < marked_img.shape[1]:
                    # Color code: green for matched cells, red for unmatched
                    if 0 <= y < gold_cells.shape[0] and 0 <= x < gold_cells.shape[1] and gold_cells[y, x] > 0:
                        cv2.circle(marked_img, (x, y), 3, (0, 255, 0), -1)  # Green for matched
                    else:
                        cv2.circle(marked_img, (x, y), 3, (255, 0, 0), -1)  # Red for unmatched
        
        plt.imshow(marked_img)
        plt.title(f'Detected Cell Locations: {len(cell_locations)}')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(visualization_path)
        plt.close()
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'num_detected': len(cell_locations),
        'num_true': num_gold_cells,
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn
    }

# This allows running the script directly for testing
if __name__ == "__main__":
    # Example usage
    image_path = "data/images/im1.jpg"
    mask_path = "results/part1/im1_mask.txt"
    cells_path = "data/gold_cells/im1_gold_cells.txt"
    output_path = "results/part2/im1_cell_locations.txt"
    vis_path = "results/part2/im1_visualization.png"
    
    # Test different configurations
    methods = ['combined', 'boundary_distance', 'intensity_based']
    min_distances = [5, 8, 10]
    
    best_f1 = 0
    best_config = None
    
    for method in methods:
        for min_distance in min_distances:
            print(f"Testing method: {method}, min_distance: {min_distance}")
            try:
                result = evaluate_cell_locations(
                    image_path, mask_path, cells_path, output_path, 
                    f"results/part2/im1_{method}_{min_distance}.png",
                    method=method,
                    min_distance=min_distance,
                    validation=True
                )
                
                print(f"  Precision: {result['precision']:.3f}")
                print(f"  Recall: {result['recall']:.3f}")
                print(f"  F1 Score: {result['f1']:.3f}")
                print(f"  Detected cells: {result['num_detected']}")
                print(f"  Gold standard cells: {result['num_true']}")
                print(f"  True positives: {result['true_positives']}")
                
                if result['f1'] > best_f1:
                    best_f1 = result['f1']
                    best_config = (method, min_distance)
            except Exception as e:
                print(f"Error with configuration {method}, {min_distance}: {e}")
    
    if best_config:
        best_method, best_min_distance = best_config
        print(f"\nBest configuration: method={best_method}, min_distance={best_min_distance}")
        
        # Final evaluation with best configuration
        result = evaluate_cell_locations(
            image_path, mask_path, cells_path, output_path, vis_path,
            method=best_method,
            min_distance=best_min_distance,
            validation=True
        )
        
        print(f"Final results with best configuration:")
        print(f"  Precision: {result['precision']:.3f}")
        print(f"  Recall: {result['recall']:.3f}")
        print(f"  F1 Score: {result['f1']:.3f}")
        print(f"  Detected cells: {result['num_detected']}")
        print(f"  Gold standard cells: {result['num_true']}")
        print(f"  True positives: {result['true_positives']}")
    else:
        print("No valid configurations found")