import torch

if torch.backends.mps.is_available():
    device = torch.device("mps")  # Utilise le GPU M1/M2 via MPS
elif torch.cuda.is_available():
    device = torch.device("cuda")  # GPU CUDA si disponible
else:
    device = torch.device("cpu")  # Par défaut, utilise le CPU

print(f"Appareil utilisé : {device}")