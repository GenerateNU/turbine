import json

class TrainConfig:
    def __init__(self, config_file:str="config.json"):
        with open(config_file, "r") as f:
            config_data = json.load(f)

        # MongoDB & Dataloader Configuration
        self.mongo_uri = config_data.get("mongo_uri", "mongodb://localhost:27017/")
        self.database_name = config_data.get("database_name", "generate")
        self.collection_names = config_data.get("collection_names", ["remo", "salutemp", "caitscurates", "voxeti"])
        self.max_length = config_data.get("max_length", 512)
        self.batch_size = config_data.get("batch_size", 4)

        # Model Configuration
        self.hidden_dim = config_data.get("hidden_dim", 256)
        self.num_heads = config_data.get("num_heads", 4)
        self.num_layers = config_data.get("num_layers", 6)
        self.dropout_rate = config_data.get("dropout_rate", 0.1)
