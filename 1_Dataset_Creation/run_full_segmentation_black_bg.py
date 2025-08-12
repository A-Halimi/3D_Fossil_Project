#!/usr/bin/env python3
"""
Run full segmentation for all fossil species
"""

import subprocess
import sys
import time
from datetime import datetime

def run_segmentation():
    """Run the full segmentation process for all species and splits"""
    
    species_list = [
        'Alveolina', 'Arumella', 'Ataxophragmium', 'Baculogypsina', 
        'Chrysalidina', 'Coskinolina', 'Elphidiella', 'Fallotia',
        'Lockhartia', 'Minoxia', 'Orbitoides', 'Rhapydionina'
    ]
    
    splits = ['test', 'train_split', 'val_split']
    
    # python_exe = "H:/3D_Fossil_projects/Dataset_v11_segmented/.venv/Scripts/python.exe"
    python_exe = sys.executable

    
    print(f"Starting full fossil segmentation at {datetime.now()}")
    print(f"Processing {len(species_list)} species across {len(splits)} splits")
    print("=" * 60)
    
    total_start_time = time.time()
    
    for split in splits:
        print(f"\n{'='*20} Processing {split.upper()} {'='*20}")
        split_start_time = time.time()
        
        for i, species in enumerate(species_list, 1):
            print(f"\n[{i}/{len(species_list)}] Processing {species} in {split}...")
            species_start_time = time.time()
            
            # Run segmentation for this species and split
            cmd = [
                python_exe, 
                "segment_fossils_black_bg.py", 
                "--splits", split,
                "--species", species
            ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                species_time = time.time() - species_start_time
                print(f"✓ Completed {species} in {species_time:.1f} seconds")
                
                # Extract summary from output
                if "successful" in result.stdout:
                    for line in result.stdout.split('\n'):
                        if "successful" in line and "failed" in line:
                            print(f"  {line.strip()}")
                            break
                
            except subprocess.CalledProcessError as e:
                print(f"✗ Error processing {species}: {e}")
                print(f"  stdout: {e.stdout}")
                print(f"  stderr: {e.stderr}")
        
        split_time = time.time() - split_start_time
        print(f"\n{'='*20} Completed {split} in {split_time/60:.1f} minutes {'='*20}")
    
    total_time = time.time() - total_start_time
    print(f"\n{'='*60}")
    print(f"SEGMENTATION COMPLETED!")
    print(f"Total processing time: {total_time/3600:.1f} hours ({total_time/60:.1f} minutes)")
    print(f"Finished at: {datetime.now()}")
    print(f"{'='*60}")

if __name__ == "__main__":
    run_segmentation()
