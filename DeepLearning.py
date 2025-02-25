#PROGRAMA (HIPERPARAMETROS AÑADIDOS)(PRUEBA N4)
#DAVID ALEJANDRO PARDO CONTRERAS
#MICHAEL LISKER

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from keras_tuner import RandomSearch
from tensorflow.keras.regularizers import l2
import zipfile
import pathlib
from sklearn.model_selection import train_test_split

# 1️⃣ Descomprimir los datasets
with zipfile.ZipFile('/content/train.zip', 'r') as zip_ref:
    zip_ref.extractall('/Simpsons/DATOS')

# 2️⃣ Definir directorio de datos
data_dir = pathlib.Path("/Simpsons/DATOS/train")

# 3️⃣ Definir nombres de clases manualmente
class_names = [
    "bart_simpson", "charles_montgomery_burns", "homer_simpson",
    "krusty_the_clown", "lisa_simpson", "marge_simpson",
    "milhouse_van_houten", "moe_szyslak", "ned_flanders", "principal_skinner"
]

# 4️⃣ Cargar dataset utilizando image_dataset_from_directory
batch_size = 256             #<----- Incrementar el tamaño del batch
img_size = (28, 28)            #<----- Mantener el tamaño de las imágenes 

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=img_size,       
    batch_size=batch_size,    
    color_mode="grayscale",
    label_mode="int",
    class_names=class_names  # Especificar class_names para asegurar solo 10 clases
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=img_size,
    batch_size=batch_size,
    color_mode="grayscale",
    label_mode="int",
    class_names=class_names  # Especificar class_names para asegurar solo 10 clases
)

# Verificar etiquetas asignadas
print("Clases detectadas:", train_ds.class_names)

# Normalización de los valores de píxeles (0-255 → 0-1)
def normalize(image, label):
    return image / 255.0, label

train_ds = train_ds.map(normalize)
val_ds = val_ds.map(normalize)

# Verificar el contenido de los datasets
print(f"Tamaño del dataset de entrenamiento: {len(train_ds)}")
print(f"Tamaño del dataset de validación: {len(val_ds)}")

# Definir el modelo MLP
model = Sequential([
    Input(shape=(28, 28, 1)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu', kernel_regularizer=l2(0.001)),  # Incrementar el número de unidades
    Dropout(0.2),  # Ajustar la tasa de dropout
    Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.2),
    Dense(10, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.0003),  # Ajustar la tasa de aprendizaje
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar el modelo con manejo de errores y verbose para ver progreso
epochs = 70
try:
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=1)
except Exception as e:
    print(f"Error durante el entrenamiento: {e}")

# Graficar resultados si el entrenamiento fue exitoso
if 'history' in locals():
    def plot_history(history):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    plot_history(history)

# 3️⃣ Herramienta de selección de hiperparámetros
def build_model(hp):
    model = Sequential()
    model.add(Input(shape=(28, 28, 1)))
    model.add(Flatten())
    for i in range(hp.Int('num_layers', 1, 3)):
        model.add(Dense(units=hp.Int('units_' + str(i), 64, 2048, step=64),  # Ajustar el rango de unidades
                        activation='relu',
                        kernel_regularizer=l2(0.001)))
        model.add(Dropout(rate=hp.Float('dropout_' + str(i), 0.1, 0.4, step=0.1)))  # Ajustar el rango de dropout
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=hp.Float('learning_rate', 1e-5, 1e-3, sampling='LOG')),  # Ajustar el rango de tasa de aprendizaje
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

tuner = RandomSearch(build_model,
                     objective='val_accuracy',
                     max_trials=10,  # Aumentar el número de pruebas
                     executions_per_trial=3,  # Aumentar el número de ejecuciones por prueba
                     directory='my_dir',
                     project_name='simpsons_mlp')

tuner.search_space_summary()

tuner.search(train_ds, epochs=20, validation_data=val_ds)  # Aumentar el número de épocas

tuner.results_summary()

best_model = tuner.get_best_models(num_models=1)[0]

# Entrenar el mejor modelo con los datos de entrenamiento y validación y verbose
best_model.summary()
history = best_model.fit(train_ds, validation_data=val_ds, epochs=70, verbose=1)

# Evaluar el modelo en el conjunto de prueba (test)
test_loss, test_acc = best_model.evaluate(val_ds)
print(f'\nTest Accuracy: {test_acc}')

# Graficar resultados finales
if 'history' in locals():
    plot_history(history)
