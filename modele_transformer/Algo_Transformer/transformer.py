import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import numpy as np
import os
import matplotlib.pyplot as plt

# Configuration GPU
print("Configuration du GPU...")
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU disponible")
    except RuntimeError as e:
        print(e)
else:
    print("Pas de GPU disponible")

class DataGenerator(tf.keras.utils.Sequence):
    """Générateur de données personnalisé pour le chargement et le prétraitement"""
    def __init__(self, data_dir, file_list, batch_size=8):
        self.data_dir = data_dir
        self.file_list = file_list
        self.batch_size = batch_size
        print(f"Premier fichier : {file_list[0]}")
        self._check_first_file()
    
    def _check_first_file(self):
        """Vérifie le premier fichier pour valider la structure des données"""
        first_file = self.file_list[0]
        feature_path = os.path.join(self.data_dir, first_file)
        label_path = os.path.join(self.data_dir, f'labels_{first_file[9:]}')
        
        try:
            features = np.load(feature_path)
            labels = np.load(label_path)
            
            print("\nVérification des données:")
            print(f"Shape features: {features.shape}")
            print(f"Shape labels: {labels.shape}")
            print(f"Valeurs uniques dans labels: {np.unique(labels)}")
            
            # Affichage de la distribution des classes
            if len(labels.shape) > 2:
                total_pixels = labels.shape[0] * labels.shape[1]
                for i in range(labels.shape[-1]):
                    count = np.sum(labels[..., i] == 1)
                    print(f"Classe {i}: {count} pixels ({count/total_pixels*100:.2f}%)")
            else:
                unique, counts = np.unique(labels, return_counts=True)
                total = np.sum(counts)
                for val, count in zip(unique, counts):
                    print(f"Classe {val}: {count} pixels ({count/total*100:.2f}%)")
        
        except Exception as e:
            print(f"Erreur lors de la vérification: {str(e)}")
    
    def __len__(self):
        return int(np.ceil(len(self.file_list) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_files = self.file_list[idx * self.batch_size:(idx + 1) * self.batch_size]
        X_batch = []
        y_batch = []
        
        for file in batch_files:
            try:
                # Chargement des features
                feature_path = os.path.join(self.data_dir, file)
                features = np.load(feature_path)
                features = np.squeeze(features)
                
                # Chargement des labels
                label_path = os.path.join(self.data_dir, f'labels_{file[9:]}')
                labels = np.load(label_path)
                labels = np.squeeze(labels)
                
                # Validation et conversion des données
                if len(labels.shape) == 2:
                    # Conversion en one-hot si nécessaire
                    labels_one_hot = np.zeros((*labels.shape, 3), dtype=np.float32)
                    labels_one_hot[..., 0] = (labels == 0).astype(np.float32)
                    labels_one_hot[..., 1] = (labels == 1).astype(np.float32)
                    labels_one_hot[..., 2] = (labels == 2).astype(np.float32)
                    labels = labels_one_hot
                
                X_batch.append(features)
                y_batch.append(labels)
                
            except Exception as e:
                print(f"Erreur de chargement pour {file}: {str(e)}")
                continue
        
        if len(X_batch) == 0:
            raise ValueError(f"Aucune donnée chargée pour le batch {idx}")
        
        X = np.array(X_batch)
        y = np.array(y_batch)
        
        if idx == 0:
            print(f"Forme du batch - X: {X.shape}, y: {y.shape}")
        
        return X, y

class IoUMetric(tf.keras.metrics.Metric):
    """Métrique IoU personnalisée pour l'évaluation par classe"""
    def __init__(self, class_id, name=None):
        super().__init__(name=name or f'iou_class_{class_id}')
        self.class_id = class_id
        self.total_iou = self.add_weight(name='total_iou', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Calcul de l'IoU pour la classe spécifique
        y_true_class = tf.cast(y_true[..., self.class_id], tf.float32)
        y_pred_class = tf.cast(
            tf.equal(tf.argmax(y_pred, axis=-1), self.class_id),
            tf.float32
        )
        
        intersection = tf.reduce_sum(y_true_class * y_pred_class)
        union = tf.reduce_sum(y_true_class) + tf.reduce_sum(y_pred_class) - intersection
        
        iou = tf.math.divide_no_nan(intersection, union)
        self.total_iou.assign_add(iou)
        self.count.assign_add(1.0)
    
    def result(self):
        return tf.math.divide_no_nan(self.total_iou, self.count)
    
    def reset_state(self):
        self.total_iou.assign(0.0)
        self.count.assign(0.0)

def build_unet(input_shape):
    """Construction du modèle U-Net avec une architecture améliorée"""
    print(f"Construction du modèle avec input_shape: {input_shape}")
    inputs = Input(input_shape)
    
    # Encoder
    conv1 = Conv2D(32, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(64, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # Bridge
    conv3 = Conv2D(128, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, 3, activation='relu', padding='same')(conv3)
    
    # Decoder
    up4 = concatenate([Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv3), conv2])
    conv4 = Conv2D(64, 3, activation='relu', padding='same')(up4)
    conv4 = Conv2D(64, 3, activation='relu', padding='same')(conv4)
    
    up5 = concatenate([Conv2DTranspose(32, 2, strides=(2, 2), padding='same')(conv4), conv1])
    conv5 = Conv2D(32, 3, activation='relu', padding='same')(up5)
    conv5 = Conv2D(32, 3, activation='relu', padding='same')(conv5)
    
    # Couche de sortie
    outputs = Conv2D(3, 1, activation='softmax')(conv5)
    
    return Model(inputs=inputs, outputs=outputs)

def train_model(model, train_generator, val_generator, epochs=20):
    """Fonction d'entraînement avec des améliorations pour les classes minoritaires"""
    print("Configuration de l'entraînement...")
    
    # Métriques
    metrics = [
        'accuracy',
        IoUMetric(class_id=0, name='iou_background'),
        IoUMetric(class_id=1, name='iou_cyclone'),
        IoUMetric(class_id=2, name='iou_riviere')
    ]
    
    # Fonction de perte focale personnalisée avec poids de classe
    def weighted_focal_loss(class_weights, gamma=2.0):
        weights = tf.constant(class_weights, dtype=tf.float32)
        
        def loss(y_true, y_pred):
            # Clip pour éviter les valeurs numériques instables
            y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
            
            # Calcul de la focal loss
            cross_entropy = -y_true * tf.math.log(y_pred)
            focal_weight = tf.pow(1 - y_pred, gamma) * y_true
            
            # Appliquer les poids des classes
            weighted_focal = cross_entropy * focal_weight
            weighted_focal = weighted_focal * weights
            
            # Moyenne sur toutes les classes et exemples
            return tf.reduce_mean(tf.reduce_sum(weighted_focal, axis=-1))
        
        return loss

    # Poids ajustés en fonction des résultats précédents
    class_weights = [1.0, 300.0, 25.0]  # [background, cyclone, riviere]
    
    # Compilation avec la nouvelle fonction de perte
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001,
            clipnorm=1.0  # Ajout de gradient clipping
        ),
        loss=weighted_focal_loss(class_weights),
        metrics=metrics
    )
    
    class DetailedMetricsCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            print(f"\nÉpoque {epoch+1}/{epochs}")
            classes = ['Background', 'Cyclone tropical', 'Rivière atmosphérique']
            
            print("\nMétriques d'entraînement:")
            print(f"Loss: {logs.get('loss', 0):.4f}")
            print(f"Accuracy: {logs.get('accuracy', 0):.4f}")
            for i, name in enumerate(classes):
                iou = logs.get(f'iou_class_{i}', 0)
                print(f"{name} IoU: {iou:.4f}")
            
            if logs.get('val_loss') is not None:
                print("\nMétriques de validation:")
                print(f"Val Loss: {logs.get('val_loss', 0):.4f}")
                print(f"Val Accuracy: {logs.get('val_accuracy', 0):.4f}")
                for i, name in enumerate(classes):
                    val_iou = logs.get(f'val_iou_class_{i}', 0)
                    print(f"{name} Val IoU: {val_iou:.4f}")
    
    # Configuration des callbacks
    callbacks = [
        DetailedMetricsCallback(),
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.keras',
            save_best_only=True,
            monitor='val_iou_cyclone',  # Priorité à la classe minoritaire
            mode='max'
        ),
        tf.keras.callbacks.EarlyStopping(
            patience=10,  # Augmenté pour donner plus de temps
            monitor='val_iou_cyclone',
            mode='max',
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.5,
            patience=5,
            monitor='val_loss',
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Augmenter le nombre d'époques
    print("\nDébut de l'entraînement...")
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Sauvegarde des courbes d'apprentissage avec plus de détails
    plt.figure(figsize=(20, 5))
    
    # Plot de la loss
    plt.subplot(141)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot de l'accuracy
    plt.subplot(142)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot des IoU d'entraînement
    plt.subplot(143)
    classes = ['background', 'cyclone', 'riviere']
    for i, class_name in enumerate(classes):
        plt.plot(history.history[f'iou_class_{i}'], 
                label=f'IoU {class_name}')
    plt.title('IoU par classe (Train)')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    
    # Plot des IoU de validation
    plt.subplot(144)
    for i, class_name in enumerate(classes):
        plt.plot(history.history[f'val_iou_class_{i}'], 
                label=f'IoU {class_name}')
    plt.title('IoU par classe (Validation)')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
    
    return history

if __name__ == "__main__":
    print("Préparation des données...")
    data_dir = "D:/Bureau/DATA/Data_restructure"
    
    # Liste des fichiers
    files = sorted([f for f in os.listdir(data_dir) if f.startswith('features_')])
    train_size = int(0.9 * len(files))
    train_files = files[:train_size]
    val_files = files[train_size:]
    
    print(f"Nombre de fichiers d'entraînement : {len(train_files)}")
    print(f"Nombre de fichiers de validation : {len(val_files)}")
    
    # Création des générateurs
    batch_size = 8
    train_generator = DataGenerator(data_dir, train_files, batch_size)
    val_generator = DataGenerator(data_dir, val_files, batch_size)
    
    # Chargement d'un échantillon pour obtenir la forme
    first_batch = train_generator[0]
    if first_batch is None or len(first_batch) != 2:
        raise ValueError("Erreur lors du chargement du premier batch")
    
    X_sample, y_sample = first_batch
    input_shape = X_sample.shape[1:]
    print(f"\nForme d'entrée : {input_shape}")
    
    # Construction et entraînement du modèle
    model = build_unet(input_shape)
    model.summary()
    
    history = train_model(model, train_generator, val_generator)
    
    # Sauvegarde du modèle final
    model.save('model_final.keras')
    print("Entraînement terminé !")