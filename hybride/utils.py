import json

def load_config(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config

def normalize_data(data, mean, std):
    return (data - mean) / std