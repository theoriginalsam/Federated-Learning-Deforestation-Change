import argparse
import logging
import numpy as np
import requests
from config.config import Config
from federated.client import FederatedClient
from sklearn.model_selection import train_test_split
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--client_id', type=int, required=True)
    parser.add_argument('--server_address', type=str, default='http://localhost:5001')
    args = parser.parse_args()
    
    try:
        # Initialize client
        logger.info(f"Initializing client {args.client_id}")
        config = Config()
        client = FederatedClient(args.client_id, config)
        client.initialize_model()
        
        # Register with server
        logger.info("Registering with server...")
        try:
            response = requests.post(
                f"{args.server_address}/register_client",
                json={'client_id': args.client_id}
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect to server: {e}")
            logger.error("Make sure the server is running first!")
            return
        
        # Training loop
        for round_num in range(config.NUM_ROUNDS):
            logger.info(f"Starting round {round_num + 1}/{config.NUM_ROUNDS}")
            
            try:
                # Get global model from server
                logger.info("Fetching global model...")
                response = requests.get(f"{args.server_address}/global_model")
                response.raise_for_status()
                global_weights = [np.array(w) for w in response.json()['weights']]
                
                # Train local model
                logger.info("Training local model...")
                local_weights, metrics = client.train_local_model(global_weights)
                
                # Convert numpy arrays to lists for JSON serialization
                serializable_weights = [w.tolist() for w in local_weights]
                
                # Send updated weights to server
                logger.info("Sending updates to server...")
                response = requests.post(
                    f"{args.server_address}/client_update",
                    json={
                        'client_id': args.client_id,
                        'weights': serializable_weights,
                        'metrics': metrics
                    }
                )
                response.raise_for_status()
                
                logger.info(f"Round {round_num + 1} completed successfully")
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Communication error in round {round_num + 1}: {e}")
                time.sleep(5)  # Wait before retrying
                continue
            except Exception as e:
                logger.error(f"Error in round {round_num + 1}: {e}")
                break
        
        logger.info("Training completed")
        
    except Exception as e:
        logger.error(f"Critical error: {e}")

if __name__ == '__main__':
    main()