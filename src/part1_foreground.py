import numpy as np
import cv2
from scipy import ndimage
from skimage import filters, morphology, measure
import matplotlib.pyplot as plt
from src.util import calculate_precision_recall_f1, visualize_mask_results, save_result

def obtain_foreground_mask(image, method='filled_mask'):
    """
    Extract foreground (cells) from background in CAMA-1 cell microscopy images.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input RGB image
    method : str
        Segmentation method ('filled_mask', 'otsu', 'combined')
    
    Returns:
    --------
    numpy.ndarray
        Binary mask where foreground (cell) pixels are 1 and background pixels are 0
    """
    # start. Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if method == 'filled_mask':
        # This method specifically targets filling the entire cell areas
        
        # a lets start with apply contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        #  set threshold to get initial cell areas and boundaries
        # handle to appear brighter than background in CAMA-1 images
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # use morphological operations to fill cell interiors
        kernel_close = np.ones((15, 15), np.uint8)  # Large kernel to close gaps within cells
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
        
        # Apply hole filling
        filled = ndimage.binary_fill_holes(closed).astype(np.uint8)
        
    
        # Remove small objects (noise)
        mask_cleaned = morphology.remove_small_objects(
            filled.astype(bool), min_size=500
        ).astype(np.uint8)
        
        # Apply additional closing to smooth boundaries
        kernel_smooth = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel_smooth)
        
    elif method == 'otsu':
        # Standard Otsu thresholding
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Otsu's thresholding
        _, binary = cv2.threshold(blurred, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # we are using morphological operations to fill holes and smooth
        #this is allowed as stated in pdf.
        kernel = np.ones((11, 11), np.uint8)
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        mask = ndimage.binary_fill_holes(closed).astype(np.uint8)
        
        # Remove small objects
        mask = morphology.remove_small_objects(mask.astype(bool), min_size=400).astype(np.uint8)
    
    elif method == 'combined':
        # Combination of methods
        
        # Apply CLAHE enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Apply Otsu's thresholding
        _, otsu_mask = cv2.threshold(enhanced, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Apply adaptive thresholding
        adaptive_mask = cv2.adaptiveThreshold(
            enhanced, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 111, 2  # Large block size to capture cell regions
        )
        
        # Combine masks (union rather than intersection)
        combined_mask = np.maximum(otsu_mask, adaptive_mask)
        
        # Fill holes, incerase smoothnesss
        kernel = np.ones((15, 15), np.uint8)
        closed = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        filled = ndimage.binary_fill_holes(closed).astype(np.uint8)
        
        # Remove small objects because of noise
        mask = morphology.remove_small_objects(filled.astype(bool), min_size=500).astype(np.uint8)
    
    else:
        raise ValueError(f"Unknown segmentation method: {method}")
    
    return mask

def evaluate_foreground_mask(image_path, mask_path, output_path, method='filled_mask', 
                          save_visualization=True, vis_path=None):
    """
    Evaluate the foreground mask algorithm on a single image.
    
    Parameters:
    -----------
    image_path : str
        Path to input image
    mask_path : str
        Path to ground truth mask
    output_path : str
        Path to save predicted mask
    method : str
        Segmentation method
    save_visualization : bool
        Whether to save visualization
    vis_path : str
        Path to save visualization
    
    Returns:
    --------
    dict
        Dictionary with evaluation metrics
    """
    # Load data
    image = cv2.imread(image_path)
    ground_truth = np.loadtxt(mask_path, dtype=np.int32)
    
    # Obtain foreground mask
    predicted_mask = obtain_foreground_mask(image, method=method)
    
    # Save result
    save_result(predicted_mask, output_path)
    
    # Calculate metrics
    precision, recall, f1 = calculate_precision_recall_f1(predicted_mask, ground_truth)
    
    # Visualize results
    if save_visualization:
        visualize_mask_results(image, ground_truth, predicted_mask, 
                              f"{method.capitalize()} (F1={f1:.3f})", vis_path)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# lets run the script directly for testing
if __name__ == "__main__":
    # Example usage
    image_path = "data/images/im1.jpg"
    mask_path = "data/gold_masks/im1_gold_mask.txt"
    output_path = "results/part1/im1_mask.txt"
    vis_path = "results/part1/im1_visualization.png"
    
    # Testing
    methods = ['filled_mask', 'otsu', 'combined']
    
    for method in methods:
        print(f"Testing method: {method}")
        result = evaluate_foreground_mask(
            image_path, mask_path, output_path, 
            method=method, 
            vis_path=f"results/part1/im1_{method}.png"
        )
        
        print(f"  Precision: {result['precision']:.3f}")
        print(f"  Recall: {result['recall']:.3f}")
        print(f"  F1 Score: {result['f1']:.3f}")
    
    # Final 
    result = evaluate_foreground_mask(image_path, mask_path, output_path, 
                                    method='filled_mask', vis_path=vis_path)
    print(f"Final results with filled_mask method:")
    print(f"  Precision: {result['precision']:.3f}")
    print(f"  Recall: {result['recall']:.3f}")
    print(f"  F1 Score: {result['f1']:.3f}")
