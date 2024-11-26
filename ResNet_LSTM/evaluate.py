import torch
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            outputs = model(features)
            preds = torch.argmax(outputs, dim=-1).cpu().numpy()
            all_preds.extend(preds.flatten())
            all_labels.extend(labels.flatten().numpy())

    print(classification_report(all_labels, all_preds, target_names=["Background", "Cyclone", "River"]))

def visualize_predictions(predictions, labels, lat_size=768, lon_size=1152):
    predictions = predictions.reshape(lat_size, lon_size)
    labels = labels.reshape(lat_size, lon_size)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Predictions")
    plt.imshow(predictions, cmap="viridis")
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.title("Ground Truth")
    plt.imshow(labels, cmap="viridis")
    plt.colorbar()

    plt.show()