import numpy as np

class FederatedAggregator:
    def __init__(self):
        pass
    
    def aggregate(self, weights_list):
        """
        Implement FedAvg algorithm
        Args:
            weights_list: List of client model weights
        Returns:
            Aggregated weights
        """
        # Simple average aggregation (FedAvg)
        averaged_weights = []
        for weights_per_layer in zip(*weights_list):
            averaged_weights.append(
                np.array([np.array(w).mean(axis=0) for w in zip(*weights_per_layer)])
            )
        return averaged_weights
