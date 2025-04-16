# Cell Detection and Segmentation Project

This project implements an algorithm for detecting and segmenting cells in microscopy images of the CAMA-1 cell line. The implementation follows a three-stage pipeline approach:

1. **Foreground Mask Extraction**: Separating cells from background
2. **Cell Location Detection**: Finding approximate locations of individual cells
3. **Cell Boundary Segmentation**: Segmenting individual cells using region growing

## Project Structure

```
cell_segmentation/
├── data/
│   ├── images/            # Input RGB images (im1.jpg, im2.jpg, im3.jpg)
│   ├── gold_masks/        # Ground truth foreground masks (im*_gold_mask.txt)
│   └── gold_cells/        # Ground truth cell annotations (im*_gold_cells.txt)
├── src/
│   ├── part1_foreground.py
│   ├── part2_locations.py
│   ├── part3_boundaries.py
│   └── utils.py
├── results/
│   ├── part1/
│   ├── part2/
│   └── part3/
├── main.py
└── requirements.txt
```

## Installation

1. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Place your data files in the appropriate directories:
   - Images in `data/images/`
   - Foreground masks in `data/gold_masks/`
   - Cell annotations in `data/gold_cells/`

2. Run the complete pipeline:
   ```
   python main.py
   ```

3. Check the results in the `results/` directory:
   - Foreground masks in `results/part1/`
   - Cell locations in `results/part2/`
   - Cell segmentations in `results/part3/`

## Algorithm Details

### Part 1: ObtainForegroundMask

This part separates foreground (cells) from background using:
- Preprocessing: Gaussian blur and contrast enhancement
- Combined segmentation approach using Otsu's and adaptive thresholding
- Postprocessing with morphological operations

### Part 2: FindCellLocations

This part finds approximate locations of cells using:
- Boundary detection to identify white borders between cells
- Distance transform to measure distance from each cell pixel to nearest boundary
- Regional maxima detection to identify cell centers

### Part 3: FindCellBoundaries

This part segments individual cells using:
- Watershed-based region growing algorithm
- Gradient magnitude as the marking function
- Cell locations from Part 2 as initial seeds

## Evaluation Metrics

- **Part 1**: Pixel-level precision, recall, and F1 score
- **Part 2**: Cell-level precision, recall, and F1 score
- **Part 3**: Cell-level Dice index and IoU at thresholds of 0.5, 0.75, and 0.9

## Example Results

After running the pipeline, you'll find visualizations for each part in the results directory:
- Part 1: Original image, ground truth mask, and predicted mask
- Part 2: Original image, regional maxima map, and cell locations
- Part 3: Original image, ground truth segmentation, and predicted segmentation

## Notes on Implementation

- This implementation uses traditional computer vision techniques without deep learning
- The algorithms can be adjusted by modifying parameters in each file
- The evaluation functions are designed to match the requirements in the assignment