import os
import cv2
import numpy as np
from glob import glob
from config.config import Config

class DataLoader:
    def __init__(self, config):
        self.config = config
        self.mask_paths = self._create_mask_paths()
    
    def _create_mask_paths(self):
        return {
            year: os.path.join(
                self.config.GROUNDTRUTH_PATH,
                f"Forest_vs_NoForest_{year}.tif"
            )
            for year in range(self.config.START_YEAR, self.config.END_YEAR)
        }
    
    def load_bands(self, year_path):
        images = []
        band_files = {
            'B02': glob(f"{year_path}/**/*_B02_10m.jp2", recursive=True),
            'B03': glob(f"{year_path}/**/*_B03_10m.jp2", recursive=True),
            'B04': glob(f"{year_path}/**/*_B04_10m.jp2", recursive=True),
            'B08': glob(f"{year_path}/**/*_B08_10m.jp2", recursive=True)
        }
        
        for b2, b3, b4, b8 in zip(band_files['B02'], band_files['B03'], 
                                 band_files['B04'], band_files['B08']):
            try:
                images.append(self._process_band_set(b2, b3, b4, b8))
            except Exception as e:
                print(f"Error processing bands: {e}")
        
        return images
    
    def _process_band_set(self, b2_path, b3_path, b4_path, b8_path):
        bands = []
        for band_path in [b2_path, b3_path, b4_path, b8_path]:
            img = cv2.imread(band_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"Failed to load band: {band_path}")
            
            img = img.astype(np.float32)
            img = cv2.resize(img, (self.config.IMG_WIDTH, self.config.IMG_HEIGHT))
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
            bands.append(img)
        
        return np.stack(bands, axis=-1)
    
    def load_masks(self):
        masks = []
        for year in range(self.config.START_YEAR, self.config.END_YEAR):
            if year in self.mask_paths:
                mask = cv2.imread(self.mask_paths[year], cv2.IMREAD_UNCHANGED)
                if mask is not None:
                    mask = cv2.resize(mask, (self.config.IMG_WIDTH, self.config.IMG_HEIGHT))
                    mask = (mask > 0).astype(np.uint8)
                    masks.append(mask)
                else:
                    print(f"Error loading mask for {year}")
        
        return np.array(masks, dtype=np.uint8) if masks else None
