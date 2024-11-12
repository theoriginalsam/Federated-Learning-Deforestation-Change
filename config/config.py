class Config:
    # Data paths
    BASE_PATH = "/Users/samir/Desktop/MTSU/MTSU-3rd Sem/Datas/Project_Dataset"
    GROUNDTRUTH_PATH = "/Users/samir/Desktop/MTSU/MTSU-3rd Sem/Datas/Project_Dataset/GroundTruth"
    
    # Image parameters
    IMG_HEIGHT = 256
    IMG_WIDTH = 256
    PATCH_SIZE = 64
    STRIDE = 16
    
    # Training parameters
    BATCH_SIZE = 32
    EPOCHS = 20
    LOCAL_EPOCHS = 5  # Number of epochs for each client
    LEARNING_RATE = 1e-4
    BETA = 0.9
    
    # Federated learning parameters
    NUM_ROUNDS = 10
    MIN_CLIENTS = 2
    MAX_CLIENTS = 5
    
    # Years range
    START_YEAR = 2015
    END_YEAR = 2024
