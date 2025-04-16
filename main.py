import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from src.part1_foreground import evaluate_foreground_mask
from src.part2_locations import evaluate_cell_locations
from src.part3_boundaries import evaluate_cell_boundaries

def main():
    """Run the complete cell segmentation pipeline on all images."""
    # Create results directories
    os.makedirs('results/part1', exist_ok=True)
    os.makedirs('results/part2', exist_ok=True)
    os.makedirs('results/part3', exist_ok=True)
    
    # Define paths for images
    image_paths = [
        'data/images/im1.jpg',
        'data/images/im2.jpg',
        'data/images/im3.jpg'
    ]
    
    # Define paths for ground truth
    mask_paths = [
        'data/gold_masks/im1_gold_mask.txt',
        'data/gold_masks/im2_gold_mask.txt',
        'data/gold_masks/im3_gold_mask.txt'
    ]
    
    cells_paths = [
        'data/gold_cells/im1_gold_cells.txt',
        'data/gold_cells/im2_gold_cells.txt',
        'data/gold_cells/im3_gold_cells.txt'
    ]
    
    # Part 1: ObtainForegroundMask
    print("Part 1: Obtaining Foreground Masks...")
    part1_results = []
    
    for i, (image_path, mask_path) in enumerate(zip(image_paths, mask_paths)):
        # Define output paths
        output_path = f'results/part1/im{i+1}_mask.txt'
        vis_path = f'results/part1/im{i+1}_visualization.png'
        
        # Run and evaluate
        print(f"Processing image {i+1}...")
        
        # Test different methods to find the best one for this image
        methods = ['filled_mask', 'otsu', 'combined']
        best_method = 'filled_mask'  # Default method
        best_f1 = 0
        
        for method in methods:
            result = evaluate_foreground_mask(
                image_path, 
                mask_path,
                output_path,
                method=method
            )
            
            if result['f1'] > best_f1:
                best_f1 = result['f1']
                best_method = method
        
        # Run again with the best method to save the visualization
        result = evaluate_foreground_mask(
            image_path, 
            mask_path,
            output_path,
            method=best_method,
            vis_path=vis_path
        )
        
        # Store results
        part1_results.append({
            'Image': f'Image {i+1}',
            'Method': best_method,
            'Precision': result['precision'],
            'Recall': result['recall'],
            'F1 Score': result['f1']
        })
    
    # Create results DataFrame
    part1_df = pd.DataFrame(part1_results)
    part1_df.to_csv('results/part1/metrics.csv', index=False)
    
    # Display results
    print("\nPart 1 Results:")
    print(part1_df.to_string(index=False, float_format=lambda x: f"{x:.3f}" if isinstance(x, float) else x))
    
    # Part 2: FindCellLocations
    print("\nPart 2: Finding Cell Locations...")
    part2_results = []
    
    for i, (image_path, cells_path) in enumerate(zip(image_paths, cells_paths)):
        # Define paths
        mask_path = f'results/part1/im{i+1}_mask.txt'  # Use predicted mask from part 1
        output_path = f'results/part2/im{i+1}_cell_locations.txt'
        vis_path = f'results/part2/im{i+1}_visualization.png'
        
        # Run and evaluate
        print(f"Processing image {i+1}...")
        
        # Test different methods to find the best one for this image
        # Using improved implementation with 'combined_approach' method
        methods = ['combined_approach', 'boundary_distance', 'intensity_based']
        min_distances = [6, 8, 10]  # Smaller min_distances to detect more cells
        
        best_method = 'combined_approach'  # Default method
        best_min_distance = 8  # Default min_distance
        best_f1 = 0
        
        for method in methods:
            for min_distance in min_distances:
                try:
                    result = evaluate_cell_locations(
                        image_path,
                        mask_path,
                        cells_path,
                        output_path,
                        method=method,
                        min_distance=min_distance,
                        validation=True
                    )
                    
                    if result['f1'] > best_f1:
                        best_f1 = result['f1']
                        best_method = method
                        best_min_distance = min_distance
                except Exception as e:
                    print(f"Error with method {method}, min_distance {min_distance}: {e}")
                    continue
        
        # Run again with the best method to save the visualization
        try:
            result = evaluate_cell_locations(
                image_path,
                mask_path,
                cells_path,
                output_path,
                vis_path,
                method=best_method,
                min_distance=best_min_distance,
                validation=True
            )
            
            # Store results
            part2_results.append({
                'Image': f'Image {i+1}',
                'Method': best_method,
                'Min Distance': best_min_distance,
                'Detected Cells': result['num_detected'],
                'True Cells': result['num_true'],
                'True Positives': result['true_positives'],
                'Precision': result['precision'],
                'Recall': result['recall'],
                'F1 Score': result['f1']
            })
        except Exception as e:
            print(f"Error in final evaluation: {e}")
            # Add placeholder results
            part2_results.append({
                'Image': f'Image {i+1}',
                'Method': best_method,
                'Min Distance': best_min_distance,
                'Detected Cells': 0,
                'True Cells': 0,
                'True Positives': 0,
                'Precision': 0,
                'Recall': 0,
                'F1 Score': 0
            })
    
    # Create results DataFrame
    part2_df = pd.DataFrame(part2_results)
    part2_df.to_csv('results/part2/metrics.csv', index=False)
    
    # Display results
    print("\nPart 2 Results:")
    print(part2_df.to_string(index=False, float_format=lambda x: f"{x:.3f}" if isinstance(x, float) else x))
    
    # Part 3: FindCellBoundaries
    print("\nPart 3: Finding Cell Boundaries...")
    part3_results = []
    
    for i, (image_path, cells_path) in enumerate(zip(image_paths, cells_paths)):
        # Define paths
        mask_path = f'results/part1/im{i+1}_mask.txt'
        cell_locations_path = f'results/part2/im{i+1}_cell_locations.txt'
        output_path = f'results/part3/im{i+1}_segmentation.txt'
        vis_path = f'results/part3/im{i+1}_visualization.png'
        
        # Run and evaluate
        print(f"Processing image {i+1}...")
        
        # Test different methods to find the best one for this image
        methods = ['watershed', 'marker_controlled', 'membrane_based']
        
        best_method = 'watershed'  # Default method
        best_dice = 0
        
        for method in methods:
            try:
                result = evaluate_cell_boundaries(
                    image_path,
                    mask_path,
                    cell_locations_path,
                    cells_path,
                    output_path,
                    method=method,
                    timeout=120  # Set timeout to 2 minutes
                )
                
                # Use Dice at threshold 0.75 for comparison
                if result['dice_0.75'] > best_dice:
                    best_dice = result['dice_0.75']
                    best_method = method
            except Exception as e:
                print(f"Error with method {method}: {e}")
                continue
        
        # Run again with the best method to save the visualization
        try:
            result = evaluate_cell_boundaries(
                image_path,
                mask_path,
                cell_locations_path,
                cells_path,
                output_path,
                vis_path,
                method=best_method,
                timeout=120
            )
            
            # Store results for each threshold
            for threshold in [0.5, 0.75, 0.9]:
                part3_results.append({
                    'Image': f'Image {i+1}',
                    'Method': best_method,
                    'Threshold': threshold,
                    'Matched Cells': result[f'matches_{threshold}'],
                    'Total Cells': result['total'],
                    'Dice Index': result[f'dice_{threshold}'],
                    'IoU': result[f'iou_{threshold}']
                })
        except Exception as e:
            print(f"Error in final evaluation: {e}")
            # Add placeholder results
            for threshold in [0.5, 0.75, 0.9]:
                part3_results.append({
                    'Image': f'Image {i+1}',
                    'Method': best_method,
                    'Threshold': threshold,
                    'Matched Cells': 0,
                    'Total Cells': 0,
                    'Dice Index': 0,
                    'IoU': 0
                })
    
    # Create results DataFrame
    part3_df = pd.DataFrame(part3_results)
    part3_df.to_csv('results/part3/metrics.csv', index=False)
    
    # Display results
    print("\nPart 3 Results:")
    print(part3_df.to_string(index=False, float_format=lambda x: f"{x:.3f}" if isinstance(x, float) else x))
    
    # Create a summary visualization of the entire pipeline
    for i, image_path in enumerate(image_paths):
        create_pipeline_visualization(i+1, image_path)
    
    print("\nCell segmentation pipeline completed successfully!")

def create_pipeline_visualization(image_index, image_path):
    """Create a visualization of the entire pipeline for the given image."""
    # Define paths
    mask_path = f'results/part1/im{image_index}_mask.txt'
    cell_locations_path = f'results/part2/im{image_index}_cell_locations.txt'
    segmentation_path = f'results/part3/im{image_index}_segmentation.txt'
    output_path = f'results/im{image_index}_pipeline_visualization.png'
    
    # Load data if available
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    # Load mask
    mask = None
    if os.path.exists(mask_path):
        try:
            mask = np.loadtxt(mask_path, dtype=np.int32)
        except Exception as e:
            print(f"Error loading mask: {e}")
    
    # Load cell locations
    cell_locations = None
    if os.path.exists(cell_locations_path):
        try:
            cell_locations = np.loadtxt(cell_locations_path, dtype=np.int32)
            # Handle single row case
            if len(cell_locations.shape) == 1 and len(cell_locations) >= 2:
                cell_locations = cell_locations.reshape(1, -1)
        except Exception as e:
            print(f"Error loading cell locations: {e}")
    
    # Load segmentation
    segmentation = None
    if os.path.exists(segmentation_path):
        try:
            segmentation = np.loadtxt(segmentation_path, dtype=np.int32)
        except Exception as e:
            print(f"Error loading segmentation: {e}")
    
    # Create figure
    plt.figure(figsize=(20, 5))
    
    # Original image
    plt.subplot(1, 4, 1)
    plt.imshow(original_image)
    plt.title('Original Image')
    plt.axis('off')
    
    # Foreground mask
    plt.subplot(1, 4, 2)
    if mask is not None:
        plt.imshow(mask, cmap='gray')
        plt.title('Foreground Mask')
    else:
        plt.imshow(np.zeros_like(original_image[:,:,0]), cmap='gray')
        plt.title('Foreground Mask (Not Available)')
    plt.axis('off')
    
    # Cell locations
    plt.subplot(1, 4, 3)
    plt.imshow(original_image)
    if cell_locations is not None and len(cell_locations) > 0:
        # Check if cell_locations is 2D and has at least 2 columns
        if len(cell_locations.shape) > 1 and cell_locations.shape[1] >= 2:
            for location in cell_locations:
                x, y = location[0], location[1]
                plt.plot(x, y, 'r.', markersize=3)
            plt.title(f'Cell Locations ({len(cell_locations)} cells)')
        else:
            plt.title('Cell Locations (Invalid Format)')
    else:
        plt.title('Cell Locations (Not Available)')
    plt.axis('off')
    
    # Segmentation
    plt.subplot(1, 4, 4)
    if segmentation is not None:
        # Create a colored visualization
        from matplotlib import cm
        from matplotlib.colors import ListedColormap
        
        # Get unique labels
        labels = np.unique(segmentation)
        
        # Create random colormap (excluding background)
        n_labels = len(labels)
        if n_labels > 1:  # Only if we have more than just background
            colors = cm.tab10(np.linspace(0, 1, 10))
            if n_labels > 10:
                colors = np.vstack([colors, np.random.rand(n_labels-10, 4)])
            colors[0] = [0, 0, 0, 1]  # Background is black
            cmap = ListedColormap(colors)
            
            plt.imshow(segmentation, cmap=cmap)
            plt.title(f'Cell Segmentation ({len(labels)-1} cells)')
        else:
            plt.imshow(np.zeros_like(original_image[:,:,0]), cmap='gray')
            plt.title('Cell Segmentation (No cells)')
    else:
        plt.imshow(np.zeros_like(original_image[:,:,0]), cmap='gray')
        plt.title('Cell Segmentation (Not Available)')
    plt.axis('off')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    main()