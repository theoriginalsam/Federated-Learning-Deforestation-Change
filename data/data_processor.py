import numpy as np
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self, config):
        self.config = config
    
    def prepare_change_pairs(self, all_images, all_masks, years):
        X, y = [], []
        
        for i in range(len(years)-1):
            year1, year2 = years[i], years[i+1]
            
            if all(k in all_images and k in all_masks for k in [year1, year2]):
                images1, images2 = all_images[year1], all_images[year2]
                mask1, mask2 = all_masks[year1], all_masks[year2]
                
                for img1 in images1:
                    for img2 in images2:
                        X.append(np.concatenate([img1, img2], axis=-1))
                        y.append(np.logical_xor(mask1, mask2).astype(np.float32))
        
        if not X:
            raise ValueError("No valid image pairs were created!")
        
        return np.array(X), np.array(y)
    
    def create_dense_patches(self, X, y):
        n_samples, height, width, channels = X.shape
        patches_X, patches_y = [], []
        
        for i in range(n_samples):
            for h in range(0, height - self.config.PATCH_SIZE + 1, self.config.STRIDE):
                for w in range(0, width - self.config.PATCH_SIZE + 1, self.config.STRIDE):
                    patch_X = X[i, h:h+self.config.PATCH_SIZE, w:w+self.config.PATCH_SIZE, :]
                    patch_y = y[i, h:h+self.config.PATCH_SIZE, w:w+self.config.PATCH_SIZE]
                    
                    if np.std(patch_y) > 0:
                        patches_X.append(patch_X)
                        patches_y.append(patch_y)
        
        return np.array(patches_X), np.array(patches_y)
