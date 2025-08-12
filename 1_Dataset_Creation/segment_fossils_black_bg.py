#!/usr/bin/env python3
"""
Fossil Image Segmentation Script
Segments fossil slices by removing background noise and keeping only the fossil structure.
"""

import cv2
import numpy as np
import os
import sys
from pathlib import Path
from skimage import filters, morphology, measure
from tqdm import tqdm
import argparse

def segment_fossil_slice(image_path, output_path):
    """
    Segment a single fossil slice image by removing background.
    
    Args:
        image_path (str): Path to input image
        output_path (str): Path to save segmented image
    
    Returns:
        bool: True if segmentation successful, False otherwise
    """
    try:
        # Load image in grayscale
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Could not load image {image_path}")
            return False
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(img, (3, 3), 0)
        
        # Apply Otsu thresholding to separate fossil from background
        threshold = filters.threshold_otsu(blurred)
        binary = blurred > threshold
        
        # Clean up with morphological operations
        # Remove small objects (noise)
        cleaned = morphology.remove_small_objects(binary, min_size=200)
        
        # Fill small holes within the fossil
        cleaned = morphology.remove_small_holes(cleaned, area_threshold=100)
        
        # Apply closing operation to connect nearby fossil parts
        kernel = morphology.disk(2)
        cleaned = morphology.binary_closing(cleaned, kernel)
        
        # Find the largest connected component (main fossil)
        labeled = measure.label(cleaned)
        if labeled.max() == 0:  # No objects found
            print(f"Warning: No fossil structure detected in {image_path}")
            return False
        
        # Keep only the largest connected component
        props = measure.regionprops(labeled)
        largest_area = max(props, key=lambda x: x.area)
        main_fossil = (labeled == largest_area.label)
        
        # Create final mask
        mask = main_fossil.astype(np.uint8) * 255
        
        # Apply mask to original image
        segmented = cv2.bitwise_and(img, img, mask=mask)
        
        # Keep background as black (0) for better transfer learning performance
        # segmented[mask == 0] = 255  # Commented out - keeping black background
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save segmented image
        success = cv2.imwrite(str(output_path), segmented)
        if not success:
            print(f"Warning: Could not save image to {output_path}")
            return False
        
        return True
        
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return False

def process_folder(input_folder, output_folder):
    """
    Process all images in a folder and its subfolders.
    
    Args:
        input_folder (str): Path to input folder
        output_folder (str): Path to output folder
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    
    # Get all image files
    image_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(input_path.rglob(f'*{ext}'))
        image_files.extend(input_path.rglob(f'*{ext.upper()}'))
    
    print(f"Found {len(image_files)} images to process in {input_folder}")
    
    successful = 0
    failed = 0
    
    # Process each image with progress bar
    for img_file in tqdm(image_files, desc=f"Processing {input_path.name}"):
        # Create relative path structure
        relative_path = img_file.relative_to(input_path)
        output_file = output_path / relative_path
        
        # Skip if output already exists
        if output_file.exists():
            continue
        
        # Segment the image
        if segment_fossil_slice(img_file, output_file):
            successful += 1
        else:
            failed += 1
    
    print(f"Completed processing {input_path.name}: {successful} successful, {failed} failed")

def main():
    parser = argparse.ArgumentParser(description='Segment fossil images')
    parser.add_argument('--input_dir', type=str, 
                       default='3d_fossil_dataset_clean',
                       help='Input directory containing original images')
    parser.add_argument('--output_dir', type=str,
                       default='3d_fossil_dataset_segmented_final', 
                       help='Output directory for segmented images')
    parser.add_argument('--splits', nargs='+', 
                       default=['test', 'train_split', 'val_split'],
                       help='Data splits to process')
    parser.add_argument('--species', nargs='+',
                       default=['Alveolina', 'Arumella', 'Ataxophragmium', 'Baculogypsina', 
                               'Chrysalidina', 'Coskinolina', 'Elphidiella', 'Fallotia',
                               'Lockhartia', 'Minoxia', 'Orbitoides', 'Rhapydionina'],
                       help='Species to process')
    
    args = parser.parse_args()
    
    input_base = Path(args.input_dir)
    output_base = Path(args.output_dir)
    
    print("Starting fossil image segmentation...")
    print(f"Input directory: {input_base.absolute()}")
    print(f"Output directory: {output_base.absolute()}")
    print(f"Processing splits: {args.splits}")
    print(f"Processing species: {args.species}")
    print("-" * 60)
    
    # Process each split and species combination
    for split in args.splits:
        print(f"\nProcessing split: {split}")
        for species in args.species:
            input_folder = input_base / split / species
            output_folder = output_base / split / species
            
            if not input_folder.exists():
                print(f"Warning: Input folder {input_folder} does not exist, skipping...")
                continue
            
            print(f"Processing {species}...")
            process_folder(input_folder, output_folder)
    
    print("\n" + "="*60)
    print("Fossil segmentation completed!")
    print(f"Check the output directory: {output_base.absolute()}")

if __name__ == "__main__":
    main()
