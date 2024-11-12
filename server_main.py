import tensorflow as tf
import numpy as np
from flask import Flask, jsonify, request
from config.config import Config
from federated.server import FederatedServer
import logging
from models.unet import build_unet
from models.losses import weighted_binary_crossentropy

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
config = Config()
server = FederatedServer(config)

@app.route('/register_client', methods=['POST'])
def register_client():
    client_id = request.json['client_id']
    logger.info(f"Registering client {client_id}")
    server.register_client(client_id)
    return jsonify({'status': 'success'})

@app.route('/global_model', methods=['GET'])
def get_global_model():
    logger.info("Distributing global model")
    weights = server.distribute_global_model()
    # Convert numpy arrays to lists for JSON serialization
    serializable_weights = [w.tolist() for w in weights]
    return jsonify({'weights': serializable_weights})

@app.route('/client_update', methods=['POST'])
def receive_client_update():
    data = request.json
    client_id = data['client_id']
    logger.info(f"Receiving update from client {client_id}")
    
    # Convert lists back to numpy arrays
    weights = [np.array(w) for w in data['weights']]
    server.receive_client_update(client_id, weights)
    
    # Check if we should aggregate
    if all(w is not None for w in server.clients.values()):
        logger.info("All clients reported. Performing aggregation.")
        server.aggregate_models()
    
    return jsonify({'status': 'success'})

def main():
    logger.info("Initializing server...")
    server.initialize_global_model()
    logger.info("Server initialized. Starting Flask application...")
    app.run(host='0.0.0.0', port=5001)

if __name__ == '__main__':
    main()