import pandas as pd
import matplotlib.pyplot as plt

# Charger les données
file_path = 'modele_CNN_LSTM/training_metrics.csv'
data = pd.read_csv(file_path)

# Graphique 1 : Évolution de la perte (Loss)
plt.figure(figsize=(10, 6))
plt.plot(data['Epoch'], data['Loss'], label='Loss', marker='o')
plt.title('Évolution de la Perte')
plt.xlabel('Époque')
plt.ylabel('Perte')
plt.legend()
plt.grid()
plt.show()

# Graphique 2 : Évolution des métriques principales
plt.figure(figsize=(10, 6))
for metric in ['Accuracy', 'Precision', 'Recall', 'F1']:
    plt.plot(data['Epoch'], data[metric], label=metric, marker='o')
plt.title('Évolution des Métriques')
plt.xlabel('Époque')
plt.ylabel('Valeur')
plt.legend()
plt.grid()
plt.show()

# Graphique 3 : Évolution des IoU par classe
plt.figure(figsize=(10, 6))
for iou_class in ['IoU_Class_0', 'IoU_Class_1', 'IoU_Class_2']:
    plt.plot(data['Epoch'], data[iou_class], label=iou_class, marker='o')
plt.title('Évolution des IoU par Classe')
plt.xlabel('Époque')
plt.ylabel('IoU')
plt.legend()
plt.grid()
plt.show()