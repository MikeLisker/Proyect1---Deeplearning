import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import zipfile
import requests
from io import BytesIO
from PIL import Image
import random
import matplotlib.pyplot as plt

# Diccionario de etiquetas basado en GitHub
label_mapping = {
    "bart_simpson": 0,
    "charles_montgomery_burns": 1,
    "homer_simpson": 2,
    "krusty_the_clown": 3,
    "lisa_simpson": 4,
    "marge_simpson": 5,
    "milhouse_van_houten": 6,
    "moe_szyslak": 7,
    "ned_flanders": 8,
    "principal_skinner": 9
}

# Función para cargar imágenes y etiquetas con depuración
def load_images_labels(data_path):
    images, labels = [], []
    for label in os.listdir(data_path):
        label_path = os.path.join(data_path, label)
        if os.path.isdir(label_path) and label in label_mapping:
            print(f"Cargando imágenes de la clase: {label}")
            for img_name in os.listdir(label_path):
                img_path = os.path.join(label_path, img_name)
                try:
                    img = Image.open(img_path).convert("L")  # Convertir a escala de grises
                    img = img.resize((28, 28))  # Redimensionar a 28x28
                    images.append(np.array(img))
                    labels.append(label_mapping[label])
                except Exception as e:
                    print(f"Error cargando imagen {img_path}: {e}")
    return np.array(images), np.array(labels)

# Directorios donde están los datos extraídos
train_data_path = "C:/Users/mclis/Downloads/grayscale-train/train"  # Reemplazar con la ruta real
test_data_path = "C:/Users/mclis/Downloads/grayscale-test/test"  # Reemplazar con la ruta real

# Cargar datos
X_train, y_train = load_images_labels(train_data_path)
X_test, y_test = load_images_labels(test_data_path)

# Normalizar imágenes
X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
X_test = X_test.reshape(-1, 28, 28, 1) / 255.0

# Dividir Train en Train/Validation (80%-20%)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

# Convertir etiquetas a One-Hot Encoding
num_classes = len(label_mapping)
y_train = to_categorical(y_train, num_classes)
y_val = to_categorical(y_val, num_classes)
y_test = to_categorical(y_test, num_classes)

# Crear el modelo MLP
model = Sequential([
    Flatten(input_shape=(28, 28, 1)),
    Dense(512, activation='relu'),
    Dropout(0.2),
    Dense(256, activation='relu'),
    Dropout(0.2),
    Dense(num_classes, activation='softmax')
])

# Compilar el modelo
model.compile(optimizer=Adam(learning_rate=0.005), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Entrenar el modelo
epochs = 30
batch_size = 64
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size)

# Evaluar el modelo en test
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_acc * 100:.2f}%')

# Graficar pérdida y precisión
def plot_history(history):
    plt.figure(figsize=(12, 5))
    
    # Gráfica de precisión
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Gráfica de pérdida
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.show()

plot_history(history)

# Seleccionar una imagen aleatoria del conjunto de test
random_index = random.randint(0, len(X_test) - 1)
test_image = X_test[random_index]
true_label = np.argmax(y_test[random_index])  # Obtener la etiqueta verdadera

# Hacer la predicción
predicted_probs = model.predict(test_image.reshape(1, 28, 28, 1))  # Redimensionar para que sea un batch
predicted_label = np.argmax(predicted_probs)  # Obtener la clase con mayor probabilidad

# Mapeo de etiquetas inverso (para mostrar nombres en lugar de números)
label_mapping_inv = {v: k for k, v in label_mapping.items()}

# Mostrar la imagen con la predicción
plt.imshow(test_image.reshape(28, 28), cmap="gray")
plt.title(f"Predicción: {label_mapping_inv[predicted_label]}\nEtiqueta real: {label_mapping_inv[true_label]}")
plt.axis("off")
plt.show()