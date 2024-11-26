import torch
import torch.nn as nn
import torch.optim as optim
from model import ClimateTransformer
from extraction import create_dataloader, VARIABLE_STATS

# Configurations
INPUT_DIM = len(VARIABLE_STATS)
NUM_LABELS = 3
SEQUENCE_LENGTH = 768 * 1152
BATCH_SIZE = 2
EPOCHS = 10
LEARNING_RATE = 1e-4

# Initialisation
netcdf_dir = "data/raw/Train"
train_dataloader = create_dataloader(netcdf_dir, SEQUENCE_LENGTH, BATCH_SIZE)
model = ClimateTransformer(INPUT_DIM, NUM_LABELS)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Entraînement
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for features, labels in train_dataloader:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs.view(-1, NUM_LABELS), labels.view(-1))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss / len(train_dataloader)}")

# Sauvegarder le modèle
torch.save(model.state_dict(), "climate_transformer.pth")