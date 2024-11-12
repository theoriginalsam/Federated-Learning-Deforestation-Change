import tensorflow as tf
import numpy as np
from models.unet import build_unet
from models.losses import weighted_binary_crossentropy
from data.data_loader import DataLoader
from data.data_processor import DataProcessor
from sklearn.model_selection import train_test_split


class FederatedClient:
    def __init__(self, client_id, config):
        self.client_id = client_id
        self.config = config
        self.model = None
        self.data_loader = DataLoader(config)
        self.data_processor = DataProcessor(config)
        
    def initialize_model(self):
        """Initialize the model with current architecture"""
        self.model = build_unet()
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.LEARNING_RATE),
            loss=weighted_binary_crossentropy(beta=self.config.BETA),
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
    
    def load_and_prepare_data(self):
        """Load and prepare data for this client"""
        # Load data specific to this client's region
        all_images = {}
        all_masks = {}
        years = range(self.config.START_YEAR, self.config.END_YEAR)
        
        for year in years:
            year_path = f"{self.config.BASE_PATH}/Region_{self.client_id}/Year_{year}"
            year_images = self.data_loader.load_bands(year_path)
            if year_images:
                all_images[year] = year_images
        
        # Load masks
        masks = self.data_loader.load_masks()
        if masks is not None:
            for idx, year in enumerate(years):
                all_masks[year] = masks[idx]
        
        return self.data_processor.prepare_change_pairs(all_images, all_masks, sorted(years))
    
    def train_local_model(self, global_weights=None):
        """Train the model on local data"""
        if global_weights is not None:
            self.model.set_weights(global_weights)
        
        # Load and prepare local data
        X, y = self.load_and_prepare_data()
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
        
        # Create patches
        X_patches, y_patches = self.data_processor.create_dense_patches(X_train, y_train)
        X_val_patches, y_val_patches = self.data_processor.create_dense_patches(X_val, y_val)
        
        # Train the model
        history = self.model.fit(
            X_patches, y_patches,
            batch_size=self.config.BATCH_SIZE,
            epochs=self.config.LOCAL_EPOCHS,
            validation_data=(X_val_patches, y_val_patches),
            verbose=1
        )
        
        return self.model.get_weights(), history.history
