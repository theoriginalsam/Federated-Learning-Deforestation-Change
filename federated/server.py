import numpy as np
from federated.aggregator import FederatedAggregator
from models.losses import weighted_binary_crossentropy
from models.unet import build_unet
import tensorflow as tf

class FederatedServer:
    def __init__(self, config):
        self.config = config
        self.global_model = None
        self.aggregator = FederatedAggregator()
        self.clients = {}
        
    def initialize_global_model(self):
        """Initialize the global model"""
        self.global_model = build_unet()
        self.global_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.LEARNING_RATE),
            loss=weighted_binary_crossentropy(beta=self.config.BETA),
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
    
    def register_client(self, client_id):
        """Register a new client"""
        self.clients[client_id] = None  # Will store client weights
    
    def distribute_global_model(self):
        """Get the current global model weights"""
        return self.global_model.get_weights()
    
    def receive_client_update(self, client_id, weights):
        """Receive and store weights from a client"""
        self.clients[client_id] = weights
    
    def aggregate_models(self):
        """Aggregate all client models"""
        all_weights = list(self.clients.values())
        if not all_weights:
            return
        
        # Use the aggregator to compute new global weights
        global_weights = self.aggregator.aggregate(all_weights)
        self.global_model.set_weights(global_weights)
        
        # Clear client updates
        self.clients = {client_id: None for client_id in self.clients}
