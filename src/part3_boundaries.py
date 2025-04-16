import numpy as np
import cv2
from scipy import ndimage
from skimage import segmentation, color, filters, morphology, measure, feature
import matplotlib.pyplot as plt
from src.util import save_result, visualize_segmentation, calculate_dice_iou
import time

def find_cell_boundaries(image, foreground_mask, cell_locations, method='membrane_enhanced'):
    """
    Segment individual cells using region growing from seed points.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input RGB image
    foreground_mask : numpy.ndarray
        Binary mask where foreground pixels are 1 and background pixels are 0
    cell_locations : numpy.ndarray
        Array of shape (n, 2) with (x, y) coordinates of cell centers
    method : str
        Segmentation method ('membrane_enhanced', 'watershed', 'marker_controlled')
    
    Returns:
    --------
    numpy.ndarray
        Segmentation map where cells are labeled from 1 to N and background is 0
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create a marker image for watershed/region growing
    markers = np.zeros_like(gray, dtype=np.int32)
    
    # Mark background as -1
    markers[foreground_mask == 0] = -1
    
    # Mark each cell location with a unique ID
    # Limit to a reasonable number to avoid performance issues
    max_cells = min(len(cell_locations), 500)  # Cap at 500 cells maximum
    
    # For better seed placement, check if we're too close to a membrane
    if method == 'membrane_enhanced':
        # Detect cell membranes (white boundaries between cells)
        # First enhance contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Apply threshold to highlight bright membranes
        _, membrane_binary = cv2.threshold(enhanced, 180, 255, cv2.THRESH_BINARY)
        
        # Dilate slightly to ensure complete membrane detection
        kernel_small = np.ones((3, 3), np.uint8)
        membrane_dilated = cv2.dilate(membrane_binary, kernel_small, iterations=1)
        
        # Create a mask of areas where seeds should NOT be placed (near membranes)
        membrane_distance = ndimage.distance_transform_edt(1 - membrane_dilated/255)
        invalid_seed_areas = membrane_distance < 5  # Keep seeds at least 5 pixels from membranes
    else:
        invalid_seed_areas = np.zeros_like(gray, dtype=bool)
    
    # Place seeds more carefully, avoiding membrane areas
    for i, (x, y) in enumerate(cell_locations[:max_cells]):
        # Make sure coordinates are within image boundaries
        if 0 <= y < markers.shape[0] and 0 <= x < markers.shape[1]:
            # Only place seed if this is not too close to a membrane and is in foreground
            if foreground_mask[y, x] > 0 and not invalid_seed_areas[y, x]:
                # Mark a small region around the center to serve as a seed
                rr, cc = draw_circle(y, x, 3, markers.shape)
                markers[rr, cc] = i + 1
    
    if method == 'membrane_enhanced':
        """
        This method specifically enhances the cell membranes to guide watershed segmentation,
        which should produce cell shapes more similar to the ground truth.
        """
        # Step 1: Enhance membrane detection in the original image
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Step 2: Apply top-hat transformation to isolate bright structures (membranes)
        kernel = np.ones((11, 11), np.uint8)
        tophat = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, kernel)
        
        # Step 3: Apply threshold to highlight membranes
        _, membrane_binary = cv2.threshold(tophat, 30, 255, cv2.THRESH_BINARY)
        
        # Step 4: Enhance membrane contrast
        membrane_enhanced = cv2.addWeighted(enhanced, 0.7, membrane_binary, 0.3, 0)
        
        # Step 5: Calculate gradient for watershed
        # Create gradients with extra weight on detected membranes
        sobelx = cv2.Sobel(membrane_enhanced, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(membrane_enhanced, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Add extra weight to membrane areas in gradient
        gradient_magnitude = gradient_magnitude + 5 * membrane_binary
        
        # Normalize to 0-255 range
        gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        
        try:
            # Step 6: Apply watershed segmentation
            segmentation_map = segmentation.watershed(gradient_magnitude, markers, watershed_line=True)
            
            # Step 7: Post-process to regularize cell shapes
            # First, set watershed lines to background
            segmentation_map[segmentation_map == -1] = 0
            
            # Label each segment
            labeled_segments, num_segments = ndimage.label(segmentation_map > 0)
            
            # Process each segment to make more regular
            for i in range(1, num_segments + 1):
                # Create mask for this segment
                segment_mask = labeled_segments == i
                
                # Apply closing to smooth boundaries
                segment_mask = morphology.binary_closing(segment_mask, morphology.disk(2))
                
                # Fill small holes
                segment_mask = morphology.remove_small_holes(segment_mask, area_threshold=50)
                
                # Update segmentation map
                # Get the most common non-zero label in this area
                segment_labels = segmentation_map[segment_mask]
                segment_labels = segment_labels[segment_labels > 0]
                if len(segment_labels) > 0:
                    most_common_label = np.bincount(segment_labels).argmax()
                    segmentation_map[segment_mask] = most_common_label
        except Exception as e:
            print(f"Error in membrane-enhanced segmentation: {e}")
            # Fallback to simpler segmentation
            segmentation_map = markers.copy()
            segmentation_map[segmentation_map == -1] = 0
    
    elif method == 'watershed':
        # Standard watershed approach
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Normalize to 0-255 range
        gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        
        try:
            # Apply watershed
            segmentation_map = segmentation.watershed(gradient_magnitude, markers)
        except Exception as e:
            print(f"Error in watershed segmentation: {e}")
            # Fallback to simpler segmentation
            segmentation_map = markers.copy()
            segmentation_map[segmentation_map == -1] = 0
    
    elif method == 'marker_controlled':
        # Marker-controlled watershed with distance transform
        dist_transform = ndimage.distance_transform_edt(foreground_mask)
        elevation_map = -dist_transform
        
        try:
            segmentation_map = segmentation.watershed(elevation_map, markers)
        except Exception as e:
            print(f"Error in marker-controlled segmentation: {e}")
            # Fallback to simpler segmentation
            segmentation_map = markers.copy()
            segmentation_map[segmentation_map == -1] = 0
    
    # Final post-processing
    
    # Set background to 0 (watershed uses -1 for background)
    segmentation_map[segmentation_map == -1] = 0
    
    # Remove very small regions (noise)
    for label in np.unique(segmentation_map):
        if label == 0:  # Skip background
            continue
        # Count pixels with this label
        size = np.sum(segmentation_map == label)
        if size < 50:  # Minimum size threshold (increased from 30)
            segmentation_map[segmentation_map == label] = 0
    
    return segmentation_map

def draw_circle(y, x, radius, shape):
    """Helper function to draw a circle of given radius at (y, x)"""
    # Create a grid of coordinates
    yy, xx = np.ogrid[-radius:radius+1, -radius:radius+1]
    
    # The distance from the center
    distance = xx**2 + yy**2
    
    # Select points within radius
    mask = distance <= radius**2
    
    # Get coordinates in the mask
    cy, cx = np.where(mask)
    
    # Shift to center at (y, x)
    cy = cy + y - radius
    cx = cx + x - radius
    
    # Clip to ensure within image boundaries
    valid = (cy >= 0) & (cy < shape[0]) & (cx >= 0) & (cx < shape[1])
    
    return cy[valid], cx[valid]

def evaluate_cell_boundaries(image_path, mask_path, cell_locations_path, 
                            cells_path, output_path, visualization_path=None,
                            method='membrane_enhanced'):
    """
    Evaluate the cell boundary segmentation algorithm on a single image.
    
    Parameters:
    -----------
    image_path : str
        Path to input image
    mask_path : str
        Path to foreground mask
    cell_locations_path : str
        Path to cell locations
    cells_path : str
        Path to ground truth cell annotations
    output_path : str
        Path to save predicted segmentation
    visualization_path : str
        Path to save visualization
    method : str
        Segmentation method
    
    Returns:
    --------
    dict
        Dictionary with evaluation metrics
    """
    # Load data
    image = cv2.imread(image_path)
    foreground_mask = np.loadtxt(mask_path, dtype=np.int32)
    gold_cells = np.loadtxt(cells_path, dtype=np.int32)
    
    # Handle empty or invalid cell locations file
    try:
        cell_locations = np.loadtxt(cell_locations_path, dtype=np.int32)
        # Check if empty or only one row
        if cell_locations.size == 0:
            print("No cell locations found. Generating placeholder locations.")
            # Run the cell location detection directly
            from src.part2_locations import find_cell_locations
            cell_locations, _ = find_cell_locations(image, foreground_mask, validation=True)
        elif len(cell_locations.shape) == 1:
            # Handle single row (reshape to correct format)
            cell_locations = cell_locations.reshape(1, -1)
            if cell_locations.shape[1] != 2:
                # Invalid format, regenerate
                print("Invalid cell locations format. Generating new locations.")
                from src.part2_locations import find_cell_locations
                cell_locations, _ = find_cell_locations(image, foreground_mask, validation=True)
    except Exception as e:
        print(f"Error loading cell locations: {e}")
        # Generate placeholder cell locations
        print("Generating cell locations directly.")
        from src.part2_locations import find_cell_locations
        cell_locations, _ = find_cell_locations(image, foreground_mask, validation=True)
    
    # Print shape of cell_locations for debugging
    print(f"Cell locations shape: {cell_locations.shape}")
    print(f"Number of cell locations: {len(cell_locations)}")
    
    # Find cell boundaries
    segmentation_map = find_cell_boundaries(
        image, foreground_mask, cell_locations, 
        method=method
    )
    
    # Save segmentation map
    save_result(segmentation_map, output_path)
    
    # Calculate metrics for different thresholds
    thresholds = [0.5, 0.75, 0.9]
    metrics = {}
    
    for threshold in thresholds:
        dice, iou, matches, total = calculate_dice_iou(segmentation_map, gold_cells, threshold)
        metrics[f'dice_{threshold}'] = dice
        metrics[f'iou_{threshold}'] = iou
        metrics[f'matches_{threshold}'] = matches
        metrics[f'total'] = total
    
    # Visualize results
    if visualization_path:
        visualize_segmentation(image, gold_cells, segmentation_map, visualization_path)
    
    return metrics

# This allows running the script directly for testing
if __name__ == "__main__":
    # Example usage
    image_path = "data/images/im1.jpg"
    mask_path = "results/part1/im1_mask.txt"
    cell_locations_path = "results/part2/im1_cell_locations.txt"
    cells_path = "data/gold_cells/im1_gold_cells.txt"
    output_path = "results/part3/im1_segmentation.txt"
    vis_path = "results/part3/im1_visualization.png"
    
    # Use the improved membrane-enhanced segmentation
    print("Testing membrane_enhanced segmentation method...")
    result = evaluate_cell_boundaries(
        image_path, mask_path, cell_locations_path, cells_path, 
        output_path, vis_path,
        method='membrane_enhanced'
    )
    
    # Print results for each threshold
    for threshold in [0.5, 0.75, 0.9]:
        print(f"  Threshold {threshold}:")
        print(f"    Dice index: {result[f'dice_{threshold}']:.3f}")
        print(f"    IoU: {result[f'iou_{threshold}']:.3f}")
        print(f"    Matches: {result[f'matches_{threshold}']} / {result['total']}")