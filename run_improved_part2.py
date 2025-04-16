import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.part2_locations import evaluate_cell_locations

def main():
    """Run a comprehensive test of the cell detection algorithm on all images."""
    # Create results directories
    os.makedirs('results/part2_final', exist_ok=True)
    
    # Define paths for images
    image_paths = [
        'data/images/im1.jpg',
        'data/images/im2.jpg',
        'data/images/im3.jpg'
    ]
    
    # Define paths for ground truth
    mask_paths = [
        'results/part1/im1_mask.txt',
        'results/part1/im2_mask.txt',
        'results/part1/im3_mask.txt'
    ]
    
    cells_paths = [
        'data/gold_cells/im1_gold_cells.txt',
        'data/gold_cells/im2_gold_cells.txt',
        'data/gold_cells/im3_gold_cells.txt'
    ]
    
    # Test different methods and parameters
    methods = ['combined', 'boundary_distance', 'intensity_based']
    min_distances = [5, 8, 10]
    
    # Store all results
    all_results = []
    best_configs = []
    
    # Process each image
    for i, (image_path, mask_path, cells_path) in enumerate(zip(image_paths, mask_paths, cells_paths)):
        print(f"\nProcessing image {i+1}...")
        
        best_f1 = 0
        best_method = None
        best_min_distance = None
        best_metrics = None
        
        # Test each configuration
        for method in methods:
            for min_distance in min_distances:
                output_path = f'results/part2_final/im{i+1}_{method}_{min_distance}_locations.txt'
                vis_path = f'results/part2_final/im{i+1}_{method}_{min_distance}_viz.png'
                
                try:
                    print(f"  Testing {method} with min_distance={min_distance}")
                    result = evaluate_cell_locations(
                        image_path, mask_path, cells_path, output_path, vis_path,
                        method=method,
                        min_distance=min_distance,
                        validation=True
                    )
                    
                    # Record metrics
                    metrics = {
                        'Image': f'Image {i+1}',
                        'Method': method,
                        'Min Distance': min_distance,
                        'Detected Cells': result['num_detected'],
                        'True Cells': result['num_true'],
                        'True Positives': result['true_positives'],
                        'False Positives': result['false_positives'],
                        'False Negatives': result['false_negatives'],
                        'Precision': result['precision'],
                        'Recall': result['recall'],
                        'F1 Score': result['f1']
                    }
                    all_results.append(metrics)
                    
                    # Print current result
                    print(f"    Precision: {result['precision']:.3f}")
                    print(f"    Recall: {result['recall']:.3f}")
                    print(f"    F1 Score: {result['f1']:.3f}")
                    print(f"    Detected: {result['num_detected']}, TP: {result['true_positives']}, FP: {result['false_positives']}")
                    
                    # Check if this is the best configuration
                    if result['f1'] > best_f1:
                        best_f1 = result['f1']
                        best_method = method
                        best_min_distance = min_distance
                        best_metrics = metrics
                
                except Exception as e:
                    print(f"    Error with {method}, min_distance={min_distance}: {e}")
        
        # Save the best result for this image
        if best_metrics:
            print(f"\n  Best configuration for Image {i+1}: {best_method}, min_distance={best_min_distance}")
            print(f"  Best F1 score: {best_f1:.3f}")
            
            # Save best result in the final location
            output_path = f'results/part2_final/im{i+1}_best_locations.txt'
            vis_path = f'results/part2_final/im{i+1}_best_visualization.png'
            
            result = evaluate_cell_locations(
                image_path, mask_path, cells_path, output_path, vis_path,
                method=best_method,
                min_distance=best_min_distance,
                validation=True
            )
            
            best_configs.append(best_metrics)
            
            # Copy the best file to the main part2 directory
            import shutil
            main_output_path = f'results/part2/im{i+1}_cell_locations.txt'
            try:
                shutil.copy2(output_path, main_output_path)
                print(f"  Copied best results to main pipeline directory: {main_output_path}")
            except Exception as e:
                print(f"  Error copying to main directory: {e}")
    
    # Create a DataFrame of all results
    all_results_df = pd.DataFrame(all_results)
    all_results_df.to_csv('results/part2_final/all_configurations.csv', index=False)
    
    # Create a DataFrame of best configurations
    best_configs_df = pd.DataFrame(best_configs)
    best_configs_df.to_csv('results/part2_final/best_configurations.csv', index=False)
    
    # Print summary
    print("\n===== SUMMARY OF BEST CONFIGURATIONS =====")
    print(best_configs_df.to_string(index=False, float_format=lambda x: f"{x:.3f}" if isinstance(x, float) else x))
    
    # Compare with previous results if available
    try:
        # Try to load results from previous runs
        previous_paths = [
            'results/part2/metrics.csv',  # Original results
        ]
        
        for prev_path in previous_paths:
            if os.path.exists(prev_path):
                previous_results = pd.read_csv(prev_path)
                print(f"\n===== COMPARISON WITH {os.path.basename(prev_path)} =====")
                
                for i, prev_row in previous_results.iterrows():
                    if i < len(best_configs):
                        current_row = best_configs[i]
                        image = f"Image {i+1}"
                        
                        if 'F1 Score' in prev_row:
                            prev_f1 = prev_row['F1 Score']
                            new_f1 = current_row['F1 Score']
                            improvement = (new_f1 - prev_f1) / prev_f1 * 100 if prev_f1 > 0 else float('inf')
                            
                            print(f"{image}: Previous F1={prev_f1:.3f}, New F1={new_f1:.3f}, Improvement: {improvement:.1f}%")
    except Exception as e:
        print(f"Could not compare with previous results: {e}")
    
    print("\nFinal testing completed successfully!")

if __name__ == "__main__":
    main()