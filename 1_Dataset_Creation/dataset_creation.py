#!/usr/bin/env python
# dataset_creation_v9_verbose.py
#
# v9  (noise-slice clean-up)  +  v8-style console logging
# ─────────────────────────────────────────────────────────────
# •   mask-guided slice sampling  (no near-empty frames)
# •   area threshold per slice   (>= 2 % fossil pixels)
# •   all progress / config prints preserved from v8
# ------------------------------------------------------------

import os, warnings, random, multiprocessing as mp
from collections import Counter

import numpy as np
import nibabel as nib
from skimage.filters import threshold_otsu
from skimage.transform import resize
from scipy.ndimage import rotate, zoom, label
from sklearn.model_selection import train_test_split
from skimage.io import imsave

warnings.filterwarnings("ignore", category=UserWarning, module="skimage")

# ─────────────────────────────────────────────────────────────
# USER SETTINGS
# ─────────────────────────────────────────────────────────────
NIFTI_DIR  = "models"
OUTPUT_DIR = "3d_fossil_dataset_clean"

TARGET_IMAGES_PER_SPECIES = 10000
TEST_SPLIT_FRAC           = 0.20
SAVE_FORMAT               = "png"           # "png" or "npy"
#WORKERS                   = max(mp.cpu_count() - 1, 1)
WORKERS                   = 16
SEED                      = 42

# Augmentation
JITTER_AUGMENTATION = True
JITTER_ANGLE_RANGE  = 5        # ± degrees
ROTATION_ANGLES     = [20, 40, 60, 80, 100, 120, 140, 160]
ROT_AXES            = [(0, 2), (0, 1), (1, 2)]

# Threshold / bbox
OVERRIDE_BBOX_THRESH = None
BOUNDING_PAD         = 2

# Slice acceptance
MIN_AREA_RATIO       = 0.02     # 2 % of 224×224

# Label map (fill the full list in real use)
SPECIES_LABEL_MAP = {
    # ... PASTE YOUR FINAL, CLEANED SPECIES_LABEL_MAP HERE ...
    "Chrysalidina_ST-1_3680_85_specimen_2_recon.nii": "Chrysalidina_specimen_2a",
    "Chrysalidina_ST_1_3675-80_specimen_1_recon.nii": "Chrysalidina_specimen_1",
    "Chrysalidina_ST_1_3675-80_specimen_9_recon.nii": "Chrysalidina_specimen_9",
    "Chrysalidina_ST_1_3675_80_8.nii": "Chrysalidina_specimen_12",
    "avizo_Chrysalidina_194_metal_stem_Chrysalidina_194_Recon_April_30_2025.nii": "Chrysalidina_specimen_11",
    "avizo_Chrysalidina_ST-1_3670_75_specimen_1_Chrysalidina_ST-1_3670_75_Specimen_1_Recon_May_1_2025.nii": "Chrysalidina_specimen_1",
    "avizo_Chrysalidina_ST-1_3675_80_specimen_2_Chrysalidina_ST-1_3675_80_Specimen_2_Recon_May_1_2025.nii": "Chrysalidina_specimen_2b",
    "avizo_Chrysalidina_ST-1_3675_80_specimen_3_Chrysalidina_ST-1_3675_80_Specimen_3_Recon_May_1_2025.nii": "Chrysalidina_specimen_3",
    "avizo_Chrysalidina_ST-1_3675_80_specimen_4_Chrysalidina_ST-1_3675_80_Specimen_4_Recon_May_1_2025.nii": "Chrysalidina_specimen_4",
    "avizo_Chrysalidina_ST-1_3675_80_specimen_5_Chrysalidina_ST-1_3675_80_Specimen_5_Recon_May_1_2025.nii": "Chrysalidina_specimen_5",
    "avizo_Chrysalidina_ST-1_3675_80_specimen_6_Chrysalidina_ST-1_3675_3680_specimen_6_Recon_May_1_2025.nii": "Chrysalidina_specimen_6",
    "avizo_Chrysalidina_ST-1_3675_80_specimen_7_Chrysalidina_St_1_3675_80_specimen_7_Recon_May_4_2025.nii": "Chrysalidina_specimen_7",
    "avizo_Chrysalidina_ST-1_3680_85_specimen_1_Chrysalidina_ST_1_3680-85_Recon_May_4_2025.nii": "Chrysalidina_specimen_1",
    "avizo_Chrysalidina_ST_1_3675-80_specimen_10_Chrysalidina_ST_1_3675_80_specimen_10_Recon_May_1_2025.nii": "Chrysalidina_specimen_10",
    "avizo_Chrysalidina_ST_1_3675-80_specimen_1_Chrysalidina_ST_1_3675_specimen_1_Recon_May_1_2025.nii": "Chrysalidina_specimen_1",
    "avizo_Chrysalidina_ST_1_3675-80_specimen_9_Chrysalidina_ST_1_3675_80_specimen_9_Recon_May_1_2025.nii": "Chrysalidina_specimen_9",
    "avizo_Ataxophragmium_782_a_below_line_specimen_2_Ataxophragmium_782_a_Recon_April_30_2025.nii": "Ataxophragmium_specimen_2",
    "avizo_Ataxophragmium_782_a_below_line_specimen_2_repeat_Ataxophragmium_782_a_Recon_April_30_2025.nii": "Ataxophragmium_specimen_2",
    "avizo_Ataxophragmium_782_a_below_line_specimen_3_recon.nii": "Ataxophragmium_specimen_3",
    "avizo_Ataxophragmium_784_a_above_line_specimen_1_Ataxophragmium_784_a_specimen_1_Recon_April_30_2025.nii": "Ataxophragmium_specimen_1",
    "avizo_Ataxophragmium_784_a_above_line_specimen_2_Ataxophragmium_784_a_specimen_2_Recon_April_30_2025.nii": "Ataxophragmium_specimen_2",
    "avizo_Ataxophragmium_784_a_above_line_specimen_3_Ataxophragmium_784_a_specimen_3_Recon_April_30_2025.nii": "Ataxophragmium_specimen_3",
    "avizo_Ataxophragmium_784_a_below_line_specimen_4_Ataxophragmium_784_a_specimen_4_Recon_April_30_2025.nii": "Ataxophragmium_specimen_4",
    "avizo_Baculogypsain_test_new_stem_regular_thickness_2_Baculogypsina_2_test_Recon_May_5_2025.nii": "Baculogypsina_specimen_1",
    "avizo_Baculogypsain_test_new_stem_regular_thickness_3_Baculogypsina_test_3_Recon_May_5_2025.nii": "Baculogypsina_specimen_2",
    "avizo_Baculogypsain_test_new_stem_regular_thickness_4_Baculogypsina_4_test_Recon_May_5_2025.nii": "Baculogypsina_specimen_3",
    "avizo_Baculogypsain_test_new_stem_regular_thickness_5_Baculogypsina_5_test_Recon_May_5_2025.nii": "Baculogypsina_specimen_4",
    "avizo_Baculogypsina_New_stem_regular_thickness_1_Baculogypsina_test_1_Recon_May_5_2025.nii": "Baculogypsina_specimen_5",
    "avizo_Baculogypsina_metal_stem_higher_projection_count_test_Baculogypsina_metal_test_May_5_2025.nii": "Baculogypsina_specimen_6",
    "avizo_Baculogypsina_stainless_steel_test_Baculogypsina_stainless_Recon_May_5_2025.nii": "Baculogypsina_specimen_7",
    "Minoxia_ST_3675-80_stainless_specimen_1_recon.nii": "Minoxia_specimen_1",
    "Minoxia_ST_3675-80_stainless_specimen_2_recon.nii": "Minoxia_specimen_2",
    "Minoxia_ST_3675-80_stainless_specimen_3_recon.nii": "Minoxia_specimen_3",
    "avizo_Minoxia_ST_3675-80_stainless_specimen_1_Minoxia_ST_3675_80_1_May_5_2025.nii": "Minoxia_specimen_1",
    "avizo_Minoxia_ST_3675-80_stainless_specimen_2_Minoxia_ST_3675_80_2_Recon_May_2025.nii": "Minoxia_specimen_2",
    "avizo_Minoxia_ST_3675-80_stainless_specimen_3_Minoxia_ST_3675_80_3_Recon_May_5_2025.nii": "Minoxia_specimen_3",
    "avizo_Elphidiella_type_stainless_specimen_1_Elphidiella_stainless_1_Recon_May_5_2025.nii": "Elphidiella_specimen_1",
    "avizo_Elphidiella_type_stainless_specimen_1_recon.nii": "Elphidiella_specimen_1",
    "avizo_Elphidiella_type_stainless_specimen_2_Elphidiella_stainless_2_Recon_May_5_2025.nii": "Elphidiella_specimen_2",
    "avizo_Elphidiella_type_stainless_specimen_3_Elphidiella_stainless_3_Recon_May_5_2025.nii": "Elphidiella_specimen_3",
    "avizo_Elphidiella_type_stainless_specimen_4_Elphidiella_stainless_4_Recon_May_5_2025.nii": "Elphidiella_specimen_4",
    "avizo_Elphidiella_type_stainless_specimen_5_Elphidiella_stainless_5_Recon_May_5_2025.nii": "Elphidiella_specimen_5",
    "avizo_Fallotia_756_a_stainless_steel_specimen_1_Fallotia_756_specimen_1_Recon_May_1_2025.nii": "Fallotia_specimen_1",
    "avizo_Fallotia_756_a_stainless_steel_specimen_2_Fallotia_756_a_specimen_2_May_1_2025.nii": "Fallotia_specimen_2",
    "avizo_Fallotia_756_a_stainless_steel_specimen_3_Fallotia_756_a_specimen_3_Recon_May_1_2025.nii": "Fallotia_specimen_3",
    "avizo_Fallotia_756_a_stainless_steel_specimen_4_Fallotia_756_a_spcimen_4_Recon_May_1_2025.nii": "Fallotia_specimen_4",
    "avizo_Fallotia_756_a_stainless_steel_specimen_5_Fallotia_756_a_specimen_5_Recon_May_1_2025.nii": "Fallotia_specimen_5",
    "avizo_Fallotia_756_a_stainless_steel_specimen_6_Fallotia_756_a_specimen_6_Recon_May_1_2025.nii": "Fallotia_specimen_6",
    "Arumella_213_specimen_6_Aug_5.nii": "Arumella_specimen_6",
    "avizo_Arumella_213_metal_stem_specimen_1_Arumella_213_Specimen_1_Recon_April_30_2025.nii": "Arumella_specimen_1",
    "avizo_Arumella_213_metal_stem_specimen_2_Arumella_213_specimen_2_April_30_2025.nii": "Arumella_specimen_2",
    "avizo_Arumella_213_metal_stem_specimen_3_Arumella_213_specimen_3_Recon_April_30_2025.nii": "Arumella_specimen_3",
    "avizo_Arumella_213_metal_stem_specimen_5_repeat_Arumella_213_specimen_5_Recon_April_30_2025.nii": "Arumella_specimen_5",
    "Lockhartia_90_a_specimen_1_recon.nii": "Lockhartia_specimen_1",
    "Lockhartia_90_a_specimen_2_recon.nii": "Lockhartia_specimen_2",
    "Lockhartia_90_a_specimen_3_recon.nii": "Lockhartia_specimen_3",
    "avizo_Lockhartia_90_a_specimen_2_Lockhartia_90_a_specimen_2_Recon_May_1_2025.nii": "Lockhartia_specimen_2",
    "avizo_Lockhartia_90_a_specimen_3_Lockhartia_90_a_specimen_3_Recon_May_1_2025.nii": "Lockhartia_specimen_3",
    "avizo_Orbitoides_554_a_specimen_1_metal_stem_Orbitoides_554_a_specimen_1_REcon_April_30_2025.nii": "Orbitoides_specimen_1",
    "avizo_Orbitoides_554_a_specimen_2_metal_stem_Orbitoides_554_a_specimen_2_Recon_April_30_2025.nii": "Orbitoides_specimen_2",
    "avizo_Orbitoides_554_a_specimen_3_metal_stem_Orbitoides_554_specimen_3_Recon_April_30_2025.nii": "Orbitoides_specimen_3",
    "avizo_Orbitoides_554_a_specimen_4_metal_stem_Orbitoides_554_a_specimen_4_Recon_April_30_2025.nii": "Orbitoides_specimen_4",
    "avizo_Questionable_Orbitoides_554_metal_stem_Omphalocyclus_18_Orbitoides_554_Recon_April_24_2025.nii": "Orbitoides",
    "avizo_Rhapydionina_845_c_specimen_1_Rhapydionina_845_c_Specimen_1_Recon_April_30_2025.nii": "Rhapydionina_specimen_1",
    "avizo_Rhapydionina_845_c_specimen_2_Rhapydionina_845_c_spcimen_2_Recon_April_30_2025.nii": "Rhapydionina_specimen_2",
    "avizo_Rhapydionina_845_c_specimen_3_Rhapydionina_845_c_spcimen_3_Recon_April_30_2025.nii": "Rhapydionina_specimen_3",
    "avizo_Rhapydionina_845_c_specimen_4_Rhapydionina_845_c_specimen_4_Recon_April_30_2025.nii": "Rhapydionina_specimen_4",
    "avizo_Rhapydionina_845_c_specimen_5_Rhapydionina_845_c_specimen_5_Recon_April_30_2025.nii": "Rhapydionina_specimen_5",
    "avizo_Alveolina_1_stainless_steel_specimen_1_Alveolina_1_specimen_1_Recon_April_30_2025.nii": "Alveolina_specimen_1",
    "avizo_Alveolina_2_stainless_steel_specimen_2_Alveolina_2_specimen_2_Recon_ApriL-30_2025.nii": "Alveolina_specimen_2",
    "avizo_Alveolina_2_stainless_steel_specimen_3_Alveolina_2_specimen_3_Recon_May_1_2025.nii": "Alveolina_specimen_3",
    "avizo_Alveolina_2_stainless_steel_specimen_4_Alveolina_2_specimen_4_Recon_May_1_2025.nii": "Alveolina_specimen_4",
    "avizo_Coskinolina_metal_stem_1_Coskinolina_1_metal_Recon_May_4_2025.nii": "Coskinolina_specimen_1",
    "avizo_Coskinolina_metal_stem_2_Coskinolina_2_metal_Recon_May_4_2025.nii": "Coskinolina_specimen_2",
    "avizo_Coskinolina_metal_stem_4_Coskinolina_metal_4_Recon_May_5_2025.nii": "Coskinolina_specimen_3",
    "avizo_Coskinolina_metal_stem_5_Coskinolina_metal_stem_5_Recon_May_5_2025.nii": "Coskinolina_specimen_4",
}

IMAGE_SIZE = (224, 224)
random.seed(SEED); np.random.seed(SEED)

# ─────────────────────────────────────────────────────────────
# HELPERS  (unchanged core logic from v9)
# ─────────────────────────────────────────────────────────────
def ensure_dir(p): os.makedirs(p, exist_ok=True)
def clean_label(fn): return SPECIES_LABEL_MAP[fn].split("_specimen_")[0]

def smart_threshold(vol):
    mx = vol.max()
    if OVERRIDE_BBOX_THRESH is not None: return OVERRIDE_BBOX_THRESH
    nz = vol[vol > vol.min()]
    if nz.size < 10: return 0.4 * mx
    try: thr = threshold_otsu(nz)
    except ValueError: thr = 0.4 * mx
    return 0.4 * mx if thr > 0.8 * mx else thr

def resample_isotropic(vol, hdr):
    zf = np.array(hdr.get_zooms()) / min(hdr.get_zooms())
    return zoom(vol, zf, order=1) if zf.ptp() > 1e-3 else vol

def build_fossil_mask(vol, thr):
    lbl, n = label(vol > thr)
    if n == 0: return np.ones_like(vol, bool), (0,)*6
    sizes = np.bincount(lbl.ravel())[1:]
    mask = lbl == (np.argmax(sizes)+1)
    loc  = np.where(mask)
    x0,x1 = loc[0].min(), loc[0].max()
    y0,y1 = loc[1].min(), loc[1].max()
    z0,z1 = loc[2].min(), loc[2].max()
    return mask, (max(0,x0-BOUNDING_PAD), min(vol.shape[0]-1,x1+BOUNDING_PAD),
                  max(0,y0-BOUNDING_PAD), min(vol.shape[1]-1,y1+BOUNDING_PAD),
                  max(0,z0-BOUNDING_PAD), min(vol.shape[2]-1,z1+BOUNDING_PAD))

def volume_minmax(vol):
    p1,p99 = np.percentile(vol,[1,99]); v = np.clip(vol,p1,p99)
    return v.min(), v.max()

def prepare_slice(a,vmin,vmax,to_uint8):
    a = np.clip(a.astype(np.float32),vmin,vmax)
    if vmax>vmin: a=(a-vmin)/(vmax-vmin)
    a = resize(a, IMAGE_SIZE, mode="constant", anti_aliasing=True)
    if to_uint8:
        a=(a*255).astype(np.uint8); a=np.stack([a]*3,-1)
    return a

def slice_has_fossil(mask2d): return mask2d.sum()/mask2d.size >= MIN_AREA_RATIO
def valid_idx(mask,axis):
    if axis==0: return np.where(mask.any(axis=(1,2)))[0]
    if axis==1: return np.where(mask.any(axis=(0,2)))[0]
    return np.where(mask.any(axis=(0,1)))[0]
def choose(cand,k):
    if len(cand)==0: return np.array([],int)
    return np.random.choice(cand,size=min(k,len(cand)),replace=False)

# rotation worker
def _rot_worker(args):
    v,a,ax=args; return rotate(v,a,ax,reshape=False,order=1,cval=v.min())

def rotated_slices(vol, n_per, thr, vmin, vmax, to_uint8):
    imgs=[]; n_rot=max(1,n_per//2)
    ctx=mp.get_context("spawn")
    tasks=[(vol,a,ax) for a in ROTATION_ANGLES for ax in ROT_AXES]
    with ctx.Pool(WORKERS,initializer=np.random.seed,initargs=(SEED,)) as pool:
        for rvol in pool.imap_unordered(_rot_worker,tasks):
            m,_=build_fossil_mask(rvol,thr)
            for z in choose(valid_idx(m,2),n_rot):
                if slice_has_fossil(m[:,:,z]):
                    imgs.append(prepare_slice(rvol[:,:,z],vmin,vmax,to_uint8))
    return imgs

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    print("--- 3D Fossil Slice Dataset Generator v9 (verbose) ---")
    print(f"[CONFIG] Output Dir: {OUTPUT_DIR}")
    print(f"[CONFIG] Target Slices/Species: {TARGET_IMAGES_PER_SPECIES}")
    print(f"[CONFIG] Workers: {WORKERS}, Format: {SAVE_FORMAT}\n")

    ensure_dir(OUTPUT_DIR)
    files=[f for f in os.listdir(NIFTI_DIR)
           if f.endswith((".nii",".nii.gz")) and f in SPECIES_LABEL_MAP]
    if not files:
        print("[ERROR] No NIfTI files match the label map. Exiting."); return

    print(f"[INFO] Found {len(files)} total 3D models with valid labels.")
    labels=[clean_label(f) for f in files]
    c=Counter(labels)
    single=[f for f,l in zip(files,labels) if c[l]==1]
    multi =[f for f,l in zip(files,labels) if c[l]>1]
    train,test=single,[]
    if multi:
        tr,te=train_test_split(multi,test_size=TEST_SPLIT_FRAC,
                               stratify=[clean_label(f) for f in multi],
                               random_state=SEED)
        train+=tr; test+=te
    print(f"[INFO] Split complete: {len(train)} models for training, {len(test)} for testing.")

    for split,fset in [("train",train),("test",test)]:
        if not fset: continue
        sp_counts=Counter([clean_label(f) for f in fset])
        print(f"\n── Generating {split.upper()} set ({len(fset)} models) ──")
        set_total=0

        quota={sp:TARGET_IMAGES_PER_SPECIES for sp in sp_counts}
        # quick upper-bound pass
        print("  [QUOTA] Estimating maximum achievable slices per species...")
        for sp,nm in sp_counts.items():
            possible=0
            for f in [x for x in fset if clean_label(x)==sp]:
                v = resample_isotropic(nib.load(os.path.join(NIFTI_DIR,f))
                                       .get_fdata(dtype=np.float32),
                                       nib.load(os.path.join(NIFTI_DIR,f)).header)
                thr=smart_threshold(v)
                _,bb=build_fossil_mask(v,thr)
                dx,dy,dz=(bb[1]-bb[0]+1,bb[3]-bb[2]+1,bb[5]-bb[4]+1)
                possible += (dx+dy+dz)*(1+len(ROTATION_ANGLES)*len(ROT_AXES)//2)
            if possible < quota[sp]:
                quota[sp]=possible
                print(f"    ⚠️  Species '{sp}' quota reduced to {possible}")

        # ------------- loop over specimens --------------------------
        for idx,nii in enumerate(fset,1):
            spec=clean_label(nii); n_mod=sp_counts[spec]
            print(f"\n  Processing model {idx}/{len(fset)}: {nii} ➜ '{spec}'")
            print("    [LOAD] Reading & resampling volume...")
            vnii=nib.load(os.path.join(NIFTI_DIR,nii))
            vol =resample_isotropic(vnii.get_fdata(dtype=np.float32),vnii.header)
            thr =smart_threshold(vol)
            vmin,vmax=volume_minmax(vol)
            mask,bb =build_fossil_mask(vol,thr)
            orient_tot=3+len(ROTATION_ANGLES)*len(ROT_AXES)
            per_orient=max(1, quota[spec]//(n_mod*orient_tot))
            imgs=[]

            print("    [PROC] Axis-aligned views (jitter={})...".format(JITTER_AUGMENTATION))
            base_vol=vol
            if JITTER_AUGMENTATION:
                a=[random.uniform(-JITTER_ANGLE_RANGE,JITTER_ANGLE_RANGE) for _ in range(3)]
                base_vol=rotate(base_vol,a[0],axes=(1,2),reshape=False,order=1,cval=vol.min())
                base_vol=rotate(base_vol,a[1],axes=(0,2),reshape=False,order=1,cval=vol.min())
                base_vol=rotate(base_vol,a[2],axes=(0,1),reshape=False,order=1,cval=vol.min())
                mask,_=build_fossil_mask(base_vol,thr*0.9)

            for axis,slicer in [(0,lambda i:base_vol[i,:,:]),
                                (1,lambda i:base_vol[:,i,:]),
                                (2,lambda i:base_vol[:,:,i])]:
                for i in choose(valid_idx(mask,axis),per_orient):
                    if slice_has_fossil(mask.take(i,axis=axis)):
                        imgs.append(prepare_slice(slicer(i),vmin,vmax,SAVE_FORMAT=="png"))

            print("    [PROC] Heavy rotated views (parallel)...")
            imgs.extend(rotated_slices(vol, per_orient, thr, vmin, vmax,
                                       SAVE_FORMAT=="png"))

            print(f"    [SAVE] Writing {len(imgs)} slices...")
            random.shuffle(imgs)
            out_dir=os.path.join(OUTPUT_DIR,split,spec); ensure_dir(out_dir)
            for k,im in enumerate(imgs):
                fn=f"{os.path.splitext(nii)[0]}_s{k}.{SAVE_FORMAT}"
                p=os.path.join(out_dir,fn)
                if SAVE_FORMAT=="png": imsave(p,im)
                else: np.save(p,im.astype(np.float32))
            set_total += len(imgs)

        print(f"\n[DONE] Finished {split} set. Total slices generated: {set_total}.")

    print("\n[SUCCESS] Dataset creation complete ✔")
    print("Output saved to:", os.path.abspath(OUTPUT_DIR))

if __name__ == "__main__":
    main()
