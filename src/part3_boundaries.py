import numpy as np
import cv2
from scipy import ndimage
from skimage import segmentation, morphology, measure, feature, filters
import matplotlib.pyplot as plt
import os
import time
from src.util import save_result, visualize_segmentation, calculate_dice_iou

def find_cell_boundaries(image, foreground_mask, cell_locations, method='membrane_enhanced', refine=True):
    """
    Segment individual cells in the CAMA-1 microscopy images using a region growing approach.
    Uses cell locations from Part 2 as initial seeds.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input RGB image
    foreground_mask : numpy.ndarray
        Binary mask where foreground (cell) pixels are 1 and background pixels are 0
    cell_locations : numpy.ndarray
        Array of shape (n, 2) with (x, y) coordinates of cell centers
    method : str
        Segmentation method to use ('membrane_enhanced', 'watershed', 'marker_controlled')
    refine : bool
        Whether to apply post-processing to refine segmentation
    
    Returns:
    --------
    numpy.ndarray
        Segmentation map where cells are labeled from 1 to N and background is 0
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Create marker image for region growing
    markers = np.zeros_like(gray, dtype=np.int32)
    
    # Mark background as -1 (watershed convention)
    markers[foreground_mask == 0] = -1
    
    # Clean up cell locations to remove any that are outside the mask or too close to each other
    processed_locations = preprocess_seeds(cell_locations, foreground_mask, gray)
    
    # Mark each cell location with a unique ID (starting from 1)
    for i, (x, y) in enumerate(processed_locations):
        # Ensure coordinates are within image bounds
        if 0 <= y < markers.shape[0] and 0 <= x < markers.shape[1]:
            # Only place seed if it's in the foreground
            if foreground_mask[y, x] > 0:
                # Create a small seed area for more stable watershed
                rr, cc = create_seed_area(y, x, 2, markers.shape)
                markers[rr, cc] = i + 1  # Labels start from 1
    
    if method == 'membrane_enhanced':
        """
        This enhanced method is specifically designed for CAMA-1 cells that have
        obvious white membranes (boundaries) between adjacent cells.
        """
        # Step 1: Enhance contrast to better see cell membranes
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Step 2: Apply edge enhancement to emphasize boundaries
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharp_enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        # Step 3: Detect white boundaries between cells
        # Use adaptive threshold to handle varying brightness
        adapt_thresh = cv2.adaptiveThreshold(
            sharp_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 25, -3
        )
        
        # Step 4: Apply top-hat transformation to isolate bright structures (membranes)
        kernel = np.ones((9, 9), np.uint8)
        tophat = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, kernel)
        
        # Step 5: Combine enhanced edges with top-hat result
        combined = cv2.addWeighted(adapt_thresh, 0.5, tophat, 0.5, 0)
        
        # Step 6: Use combined edge map to create a boundary-enhanced image
        membrane_enhanced = cv2.addWeighted(enhanced, 0.7, combined, 0.3, 0)
        
        # Step 7: Calculate gradient for watershed
        sobel_x = cv2.Sobel(membrane_enhanced, cv2.CV_64F, 1, 0, ksize=5)  # Larger kernel for smoother gradients
        sobel_y = cv2.Sobel(membrane_enhanced, cv2.CV_64F, 0, 1, ksize=5)
        gradient = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Add extra weight to membrane areas
        gradient = gradient + 10 * combined.astype(float)
        
        # Normalize to 0-255 range
        gradient = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        
        # Step 8: Apply Gaussian blur to smooth the gradient
        gradient = cv2.GaussianBlur(gradient, (5, 5), 1.0)
        
        # Step 9: Apply watershed segmentation with distance from boundaries
        try:
            segmentation_map = segmentation.watershed(gradient, markers, watershed_line=True)
            
            # Set watershed lines to background
            segmentation_map[segmentation_map == -1] = 0
        except Exception as e:
            print(f"Error in watershed segmentation: {e}")
            # Fallback to a simpler approach if watershed fails
            segmentation_map = markers.copy()
            segmentation_map[segmentation_map == -1] = 0
    
    elif method == 'watershed':
        # Standard watershed approach using image gradient
        
        # Create a marking function based on image gradient
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Normalize gradient to 0-255 range
        gradient = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        
        # Apply watershed algorithm
        try:
            segmentation_map = segmentation.watershed(gradient, markers)
            
            # Set watershed lines to background
            segmentation_map[segmentation_map == -1] = 0
        except Exception as e:
            print(f"Error in watershed segmentation: {e}")
            # Fallback to a simpler approach
            segmentation_map = markers.copy()
            segmentation_map[segmentation_map == -1] = 0
    
    elif method == 'marker_controlled':
        # Marker-controlled watershed with distance transform
        
        # Create distance transform from foreground mask
        dist_transform = ndimage.distance_transform_edt(foreground_mask)
        
        # Smooth the distance transform
        dist_transform = cv2.GaussianBlur(dist_transform, (5, 5), 0)
        
        # Invert distance transform to use as marking function
        # This makes peaks (cell centers) become valleys for watershed
        dist_transform_max = np.max(dist_transform)
        if dist_transform_max > 0:
            inv_dist = dist_transform_max - dist_transform
        else:
            inv_dist = dist_transform
        
        # Apply watershed algorithm
        try:
            segmentation_map = segmentation.watershed(inv_dist, markers)
            
            # Set watershed lines to background
            segmentation_map[segmentation_map == -1] = 0
        except Exception as e:
            print(f"Error in marker-controlled segmentation: {e}")
            # Fallback to a simpler approach
            segmentation_map = markers.copy()
            segmentation_map[segmentation_map == -1] = 0
    
    else:
        # Default to membrane_enhanced if unknown method is specified
        print(f"Unknown segmentation method: {method}. Using membrane_enhanced instead.")
        return find_cell_boundaries(image, foreground_mask, cell_locations, method='membrane_enhanced', refine=refine)
    
    # Apply post-processing to refine the segmentation
    if refine:
        segmentation_map = refine_segmentation(segmentation_map, foreground_mask, image)
    
    return segmentation_map

def preprocess_seeds(cell_locations, foreground_mask, gray_image):
    """
    Preprocess seed locations to ensure they are well-positioned for region growing.
    Tries to place seeds at optimal positions within cells (away from boundaries).
    
    Parameters:
    -----------
    cell_locations : numpy.ndarray
        Array of shape (n, 2) with (x, y) coordinates of cell centers
    foreground_mask : numpy.ndarray
        Binary mask where foreground pixels are 1 and background pixels are 0
    gray_image : numpy.ndarray
        Grayscale image for detecting optimal seed position
    
    Returns:
    --------
    list
        List of (x, y) coordinates of processed seed locations
    """
    # Validate input cell_locations
    if cell_locations is None or len(cell_locations) == 0:
        return []
    
    processed_seeds = []
    
    # Create a gradient image to detect cell boundaries
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Normalize gradient
    gradient = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    # Enhance contrast to better see cell features
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_image)
    
    # Create a mask of likely cell centers (darker regions)
    inverted = 255 - enhanced
    
    # For each seed location
    for x, y in cell_locations:
        # Skip if out of bounds
        if not (0 <= y < foreground_mask.shape[0] and 0 <= x < foreground_mask.shape[1]):
            continue
            
        # Skip if not in foreground
        if foreground_mask[y, x] == 0:
            continue
        
        # Search for better position within a window
        window_size = 11  # Larger window for better repositioning
        half = window_size // 2
        
        # Extract windows around current position
        y_min = max(0, y - half)
        y_max = min(foreground_mask.shape[0], y + half + 1)
        x_min = max(0, x - half)
        x_max = min(foreground_mask.shape[1], x + half + 1)
        
        # Get gradient window (high values = boundaries)
        gradient_window = gradient[y_min:y_max, x_min:x_max].copy()
        
        # Get inverted intensity window (high values = likely cell centers)
        intensity_window = inverted[y_min:y_max, x_min:x_max].copy()
        
        # Get foreground mask window
        mask_window = foreground_mask[y_min:y_max, x_min:x_max].copy()
        
        # Set non-foreground areas to bad values
        gradient_window[mask_window == 0] = 255  # Highest gradient (avoid)
        intensity_window[mask_window == 0] = 0   # Lowest intensity (avoid)
        
        # Create combined score (high intensity, low gradient = good)
        score_window = intensity_window.astype(float) - gradient_window.astype(float)/2
        
        # Find position with best score
        if np.any(mask_window > 0):  # Only if there's any foreground
            best_y, best_x = np.unravel_index(np.argmax(score_window), score_window.shape)
            new_y = y_min + best_y
            new_x = x_min + best_x
            
            # Only add if position is within foreground
            if foreground_mask[new_y, new_x] > 0:
                processed_seeds.append((new_x, new_y))
            else:
                # Keep original if new position is outside foreground
                processed_seeds.append((x, y))
        else:
            # Keep original if no foreground in window
            processed_seeds.append((x, y))
    
    # Check for duplicates or very close seeds
    final_seeds = []
    min_distance = 5  # Minimum distance between seeds
    
    for seed in processed_seeds:
        # Check if this seed is too close to any already accepted seed
        too_close = False
        for accepted_seed in final_seeds:
            dist = np.sqrt((seed[0] - accepted_seed[0])**2 + (seed[1] - accepted_seed[1])**2)
            if dist < min_distance:
                too_close = True
                break
        
        # Add seed if not too close to existing seeds
        if not too_close:
            final_seeds.append(seed)
    
    return final_seeds

def create_seed_area(y, x, radius, shape):
    """
    Create a small circular seed area around the center point.
    
    Parameters:
    -----------
    y, x : int
        Center coordinates
    radius : int
        Radius of the circular area
    shape : tuple
        Shape of the image (height, width)
    
    Returns:
    --------
    tuple
        (row_indices, col_indices) for the circular area
    """
    # Create a grid of coordinates
    rr, cc = np.ogrid[-radius:radius+1, -radius:radius+1]
    
    # Calculate distance from center
    distance = rr**2 + cc**2
    
    # Select points within radius
    mask = distance <= radius**2
    
    # Get coordinates
    rr_indices, cc_indices = np.where(mask)
    
    # Shift to center at (y, x)
    rr_indices = rr_indices + y - radius
    cc_indices = cc_indices + x - radius
    
    # Ensure coordinates are within image bounds
    valid = (rr_indices >= 0) & (rr_indices < shape[0]) & (cc_indices >= 0) & (cc_indices < shape[1])
    
    return rr_indices[valid], cc_indices[valid]

def refine_segmentation(segmentation, foreground_mask, image=None):
    """
    Post-process segmentation to refine cell boundaries and remove small artifacts.
    Creates smoother, more cell-like boundaries.
    
    Parameters:
    -----------
    segmentation : numpy.ndarray
        Initial segmentation map
    foreground_mask : numpy.ndarray
        Binary mask where foreground pixels are 1 and background pixels are 0
    image : numpy.ndarray or None
        Original image (optional, used for additional refinement)
    
    Returns:
    --------
    numpy.ndarray
        Refined segmentation map
    """
    # Convert foreground mask to boolean
    foreground = foreground_mask > 0
    
    # Get unique labels (excluding background)
    unique_labels = np.unique(segmentation)
    unique_labels = unique_labels[unique_labels > 0]
    
    # Create refined segmentation
    refined = np.zeros_like(segmentation)
    
    # Process each cell
    for label in unique_labels:
        # Create binary mask for this cell
        mask = (segmentation == label)
        
        # Skip very small regions (likely noise)
        if np.sum(mask) < 30:
            continue
        
        # Apply a series of morphological operations to smooth boundaries
        # First, close small holes and smooth edges
        smooth_mask = morphology.binary_closing(mask, morphology.disk(3))
        
        # Remove small isolated pixels
        smooth_mask = morphology.remove_small_objects(smooth_mask, min_size=30)
        
        # Fill small holes
        smooth_mask = morphology.remove_small_holes(smooth_mask, area_threshold=100)
        
        # Apply additional closing for smoother boundaries
        smooth_mask = morphology.binary_closing(smooth_mask, morphology.disk(2))
        
        # Ensure mask stays within foreground
        smooth_mask = np.logical_and(smooth_mask, foreground)
        
        # Add to refined segmentation
        refined[smooth_mask] = label
    
    # Perform a final pass to eliminate small gaps between cells
    # Create a binary mask of all segmented cells
    all_cells = refined > 0
    
    # Apply morphological closing to fill small gaps between cells
    closed_all = morphology.binary_closing(all_cells, morphology.disk(1))
    
    # Find pixels that were added by the closing operation
    added_pixels = np.logical_and(closed_all, ~all_cells)
    
    # For each added pixel, assign it to the nearest cell
    if np.any(added_pixels):
        y_indices, x_indices = np.where(added_pixels)
        
        for y, x in zip(y_indices, x_indices):
            # Skip if not in foreground
            if not foreground[y, x]:
                continue
                
            # Check 8-connected neighborhood for cell labels
            neighbor_labels = []
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                        
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < refined.shape[0] and 0 <= nx < refined.shape[1]:
                        label = refined[ny, nx]
                        if label > 0:
                            neighbor_labels.append(label)
            
            # Assign pixel to most common neighboring label
            if neighbor_labels:
                from collections import Counter
                most_common = Counter(neighbor_labels).most_common(1)[0][0]
                refined[y, x] = most_common
    
    return refined

def evaluate_cell_boundaries(image_path, mask_path, cell_locations_path, 
                            cells_path, output_path, visualization_path=None,
                            method='membrane_enhanced', timeout=None):
    """
    Evaluate the cell boundary segmentation algorithm on a single image.
    
    Parameters:
    -----------
    image_path : str
        Path to input image
    mask_path : str
        Path to foreground mask (from part 1)
    cell_locations_path : str
        Path to cell locations (from part 2)
    cells_path : str
        Path to ground truth cell annotations
    output_path : str
        Path to save predicted segmentation
    visualization_path : str
        Path to save visualization
    method : str
        Segmentation method to use
    timeout : int or None
        Maximum time (in seconds) to allow for segmentation
    
    Returns:
    --------
    dict
        Dictionary with evaluation metrics at different thresholds
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load data
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    foreground_mask = np.loadtxt(mask_path, dtype=np.int32)
    gold_cells = np.loadtxt(cells_path, dtype=np.int32)
    
    # Load cell locations
    try:
        cell_locations = np.loadtxt(cell_locations_path, dtype=np.int32)
        # Handle single location case
        if len(cell_locations.shape) == 1:
            if len(cell_locations) >= 2:  # At least x, y coordinates
                cell_locations = cell_locations.reshape(1, -1)
            else:
                raise ValueError("Invalid cell locations format")
    except Exception as e:
        print(f"Error loading cell locations: {e}")
        # Create dummy cell locations in foreground regions as fallback
        print("Creating fallback cell locations")
        # Find connected components in foreground
        labels, num = ndimage.label(foreground_mask)
        # Get centroids
        cell_locations = []
        for i in range(1, num + 1):
            y, x = np.where(labels == i)
            if len(y) > 0:
                cell_locations.append([int(np.mean(x)), int(np.mean(y))])
        cell_locations = np.array(cell_locations)
    
    # Print information about data
    print(f"Image shape: {image.shape}")
    print(f"Foreground mask shape: {foreground_mask.shape}")
    print(f"Number of cell locations: {len(cell_locations)}")
    
    # Apply timeout if specified
    if timeout:
        start_time = time.time()
        
        # Find cell boundaries with timeout
        if hasattr(time, 'pthread_getcpuclockid'):  # Unix-based systems
            # Use signal-based timeout on Unix
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Segmentation timed out after {timeout} seconds")
            
            # Set timeout
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)
            
            try:
                segmentation_map = find_cell_boundaries(image, foreground_mask, cell_locations, method)
                signal.alarm(0)  # Cancel alarm
            except TimeoutError as e:
                print(str(e))
                # Fallback to simpler method
                print("Using fallback segmentation method due to timeout")
                segmentation_map = find_cell_boundaries(image, foreground_mask, cell_locations, 'watershed')
        else:
            # Simple time-based check for Windows
            segmentation_map = find_cell_boundaries(image, foreground_mask, cell_locations, method)
            if time.time() - start_time > timeout:
                print(f"Segmentation took too long ({time.time() - start_time:.2f}s)")
    else:
        # No timeout, just run the segmentation
        segmentation_map = find_cell_boundaries(image, foreground_mask, cell_locations, method)
    
    # Save segmentation map
    save_result(segmentation_map, output_path)
    
    # Calculate metrics at different thresholds
    thresholds = [0.5, 0.75, 0.9]
    results = {}
    
    for threshold in thresholds:
        # Calculate Dice and IoU
        dice, iou, matches, total = calculate_dice_iou(segmentation_map, gold_cells, threshold)
        
        # Store results
        results[f'dice_{threshold}'] = dice
        results[f'iou_{threshold}'] = iou
        results[f'matches_{threshold}'] = matches
        results['total'] = total
        
        print(f"Threshold {threshold}: Dice={dice:.3f}, IoU={iou:.3f}, Matches={matches}/{total}")
    
    # Visualize segmentation
    if visualization_path:
        visualize_segmentation(image, gold_cells, segmentation_map, visualization_path)
    
    return results

# This allows running the script directly for testing
if __name__ == "__main__":
    # Example usage
    image_path = "data/images/im1.jpg"
    mask_path = "results/part1/im1_mask.txt"
    cell_locations_path = "results/part2/im1_cell_locations.txt"
    cells_path = "data/gold_cells/im1_gold_cells.txt"
    output_path = "results/part3/im1_segmentation.txt"
    vis_path = "results/part3/im1_visualization.png"
    
    # Test different methods
    methods = ['membrane_enhanced', 'watershed', 'marker_controlled']
    best_method = None
    best_dice = 0
    
    for method in methods:
        print(f"\nTesting method: {method}")
        try:
            result = evaluate_cell_boundaries(
                image_path, mask_path, cell_locations_path, cells_path, 
                output_path, vis_path, method=method
            )
            
            # Using Dice at threshold 0.75 for comparison
            dice_75 = result['dice_0.75']
            print(f"Method: {method}, Dice (0.75): {dice_75:.3f}")
            
            if dice_75 > best_dice:
                best_dice = dice_75
                best_method = method
        except Exception as e:
            print(f"Error with method {method}: {e}")
    
    if best_method:
        print(f"\nBest method: {best_method} with Dice (0.75): {best_dice:.3f}")
    else:
        print("No method succeeded")