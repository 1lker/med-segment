import numpy as np
import cv2
from scipy import ndimage
from skimage import feature, morphology, measure, filters
import matplotlib.pyplot as plt
from src.util import save_result, visualize_cells_with_locations

def find_cell_locations(image, foreground_mask, method='distance_transform', 
                      min_distance=10, validation=True):
    """
    Find approximate locations of cells in CAMA-1 cell images.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input RGB image
    foreground_mask : numpy.ndarray
        Binary mask where foreground pixels are 1 and background pixels are 0
    method : str
        Method to use ('distance_transform', 'h_maxima', 'intensity_based')
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
    # start here, convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if method == 'distance_transform':
        # Apply distance transform on the foreground mask
        dist_transform = ndimage.distance_transform_edt(foreground_mask)
        
        # Apply smoothing to avoid noisy peaks
        dist_transform = ndimage.gaussian_filter(dist_transform, sigma=2)
        
        # Use peak_local_max for more stable peak detection
        from skimage.feature import peak_local_max
        
        # choose and call peaks with a minimum distance between peaks
        # threshold for noise clearing
        min_distance_value = 3  # minimum distance value to consider as a valid peak
        # Removed 'indices' parameter to avoid TypeError
        coordinates = peak_local_max(dist_transform, min_distance=min_distance, 
                                   threshold_abs=min_distance_value,
                                   exclude_border=False)
     
        cell_locations = [(int(x), int(y)) for y, x in coordinates]
        
        # create a maxima map for visualization
        maxima = np.zeros_like(foreground_mask, dtype=bool)
        for y, x in coordinates:
            if 0 <= y < maxima.shape[0] and 0 <= x < maxima.shape[1]:
                maxima[y, x] = True
    
    elif method == 'h_maxima':
        # Apply distance transform
        dist_transform = ndimage.distance_transform_edt(foreground_mask)
        
        # Apply h-maxima transform to find significant local maxima
        from skimage.morphology import h_maxima
        
        # Adjust h value based on the maximum distance value
        max_dist = np.max(dist_transform)
        h_value = max_dist / 10 if max_dist > 0 else 1.0
        
        # Find h-maxima
        maxima = h_maxima(dist_transform, h_value)
        
        # Label maxima regions
        labeled_maxima, num_labels = ndimage.label(maxima)
        
        # Calculate centroids of maxima regions
        cell_locations = []
        for i in range(1, num_labels + 1):
            y_indices, x_indices = np.where(labeled_maxima == i)
            if len(y_indices) > 0:
                centroid_y = int(np.mean(y_indices))
                centroid_x = int(np.mean(x_indices))
                cell_locations.append((centroid_x, centroid_y))
    
    elif method == 'intensity_based':
        
        # we are implying contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # mask out background regions?
        masked_gray = np.zeros_like(enhanced)
        masked_gray[foreground_mask > 0] = enhanced[foreground_mask > 0]
        
        # Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(masked_gray, (5, 5), 0)
        
        # invert so cell centers (darker) become peaks idea here
        inverted = 255 - blurred
        
        # here we find local maxima
        from skimage.feature import peak_local_max
        # Removed 'indices' parameter
        coordinates = peak_local_max(inverted, min_distance=min_distance, 
                                   threshold_abs=50,  # minimum intensity to consider we pick
                                   exclude_border=False)
        
        cell_locations = [(int(x), int(y)) for y, x in coordinates]
        
        # Create maxima map
        maxima = np.zeros_like(foreground_mask, dtype=bool)
        for y, x in coordinates:
            if 0 <= y < maxima.shape[0] and 0 <= x < maxima.shape[1]:
                maxima[y, x] = True
    
    # If validation is enabled, filter out some cell locations
    if validation and len(cell_locations) > 0:
        validated_locations = []
        
        # Calculate average cell size from the foreground mask
        # This helps determine how many cells should be in each region
        labeled_regions, num_regions = ndimage.label(foreground_mask)
        region_sizes = []
        for i in range(1, num_regions + 1):
            region_size = np.sum(labeled_regions == i)
            region_sizes.append(region_size)
        
        avg_cell_size = 100  # Default value
        if len(region_sizes) > 0:
            avg_cell_size = np.median(region_sizes)  # Median is more robust than mean
        
        # Estimate typical cell radius
        estimated_radius = np.sqrt(avg_cell_size / np.pi)
        

        for x, y in cell_locations:
            # check if  it's within the foreground mask
            if 0 <= y < foreground_mask.shape[0] and 0 <= x < foreground_mask.shape[1]:
                if foreground_mask[y, x] > 0:
                    # point is not too close to an already validated cell?
                    too_close = False
                    for vx, vy in validated_locations:
                        # we did calculate distance to other validated cells
                        dist = np.sqrt((x - vx) ** 2 + (y - vy) ** 2)
                        if dist < estimated_radius * 0.5:  # we did use half the radius for stricter filtering
                            too_close = True
                            break
                    
                    if not too_close:
                        validated_locations.append((x, y))
        
        # here we are making updates
        cell_locations = validated_locations
        
        # recreate maxima map
        maxima = np.zeros_like(foreground_mask, dtype=bool)
        for x, y in cell_locations:
            if 0 <= y < maxima.shape[0] and 0 <= x < maxima.shape[1]:
                maxima[y, x] = True
    
    print(f"Number of cells detected: {len(cell_locations)}")
    return np.array(cell_locations), maxima.astype(np.uint8)

def evaluate_cell_locations(image_path, mask_path, cells_path, output_path, 
                           visualization_path=None, method='distance_transform',
                           min_distance=10, validation=True):
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

    image = cv2.imread(image_path)
    foreground_mask = np.loadtxt(mask_path, dtype=np.int32)
    gold_cells = np.loadtxt(cells_path, dtype=np.int32)
    
    # find cell locations
    cell_locations, regional_maxima = find_cell_locations(
        image, foreground_mask, 
        method=method,
        min_distance=min_distance,
        validation=validation
    )
    
   
    save_result(cell_locations, output_path)
    

    # A true positive means that a detected cell location corresponds to a gold standard cell
    tp = 0
    for x, y in cell_locations:
 
        if 0 <= y < gold_cells.shape[0] and 0 <= x < gold_cells.shape[1]:
            # point corresponds to a foreground cell in gold standard?
            if gold_cells[y, x] > 0:
                tp += 1
    
    # Count the number of unique cells in gold standard (excluding background)
    num_gold_cells = len(np.unique(gold_cells)) - 1  # Subtract 1 for background
    
    # Calculate metrics
    precision = tp / len(cell_locations) if len(cell_locations) > 0 else 0
    recall = tp / num_gold_cells if num_gold_cells > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Visualize results
    if visualization_path:
        visualize_cells_with_locations(image, regional_maxima, cell_locations, visualization_path)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'num_detected': len(cell_locations),
        'num_true': num_gold_cells,
        'true_positives': tp

if __name__ == "__main__":
  
    image_path = "data/images/im1.jpg"
    mask_path = "results/part1/im1_mask.txt"
    cells_path = "data/gold_cells/im1_gold_cells.txt"
    output_path = "results/part2/im1_cell_locations.txt"
    vis_path = "results/part2/im1_visualization.png"
    
    # Test configs
    methods = ['distance_transform', 'h_maxima', 'intensity_based']
    min_distances = [8, 10, 15]
    
    best_f1 = 0
    best_config = None
    
    for method in methods:
        for min_distance in min_distances:
            print(f"Testing method: {method}, min_distance: {min_distance}")
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
    
    if best_config:
        best_method, best_min_distance = best_config
        print(f"\nBest configuration: method={best_method}, min_distance={best_min_distance}")
        
        # Final result is here as the best option we have
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
