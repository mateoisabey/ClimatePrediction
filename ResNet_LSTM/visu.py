import torch
from model import ClimateTransformer
from extraction import create_dataloader, VARIABLE_STATS
from evaluate import evaluate_model, visualize_predictions

# Configurations
INPUT_DIM = len(VARIABLE_STATS)
NUM_LABELS = 3
SEQUENCE_LENGTH = 768 * 1152
BATCH_SIZE = 1

# Chargement des données et du modèle
val_dir = "data/raw/Test"
val_dataloader = create_dataloader(val_dir, SEQUENCE_LENGTH, BATCH_SIZE)

model = ClimateTransformer(INPUT_DIM, NUM_LABELS)
model.load_state_dict(torch.load("climate_transformer.pth"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Évaluation
evaluate_model(model, val_dataloader, device)

# Visualisation
for features, labels in val_dataloader:
    features, labels = features.to(device), labels.to(device)
    with torch.no_grad():
        outputs = model(features)
        predictions = torch.argmax(outputs, dim=-1).cpu().numpy()
        labels = labels.cpu().numpy()

    visualize_predictions(predictions[0], labels[0], lat_size=768, lon_size=1152)
    break