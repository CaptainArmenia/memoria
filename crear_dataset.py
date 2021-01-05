import numpy as np
import tensorflow as tf
import os
from inferencia import partial_forward, knn, most_frequent, absoluteFilePaths
from shutil import copyfile

input_shape = (224, 224, 3)
model = tf.keras.applications.resnet50.ResNet50(include_top= True, classifier_activation="softmax", weights= 'imagenet', input_shape= input_shape, classes=1000)
target_layer_name = "conv5_block3_out"
target_layer = model.get_layer(target_layer_name)
channels = target_layer.output_shape[-1]
#input_path = "C:/Users/andyb/OneDrive/Documentos/memoria/datasets/feidegger-master/feidegger-master/data"
input_path = "C:/Users/andyb/PycharmProjects/kerasResnet/images390p"
baseline_path = "C:/Users/andyb/OneDrive/Documentos/memoria/distance_val_dataset/"
label_files_path = "C:/Users/andyb/OneDrive/Documentos/memoria/datasets/"
#features = ["rojo", "negro", "azul", "verde", "amarillo", "gris", "cafe", "rosado", "morado", "naranjo"]
features = ["cuadros", "rayas", "flores", "leopardo", "puntos", "plano", "piel", "lana", "pata_de_gallo", "lentejuelas"]
n_examples = []
n_neighbors = 3

#se crean clusters de ejemplo

activations = np.empty((0, channels))

for feature in features:
    feature_examples = 0

    for file in absoluteFilePaths(baseline_path + feature):
        feature_examples += 1
        result = partial_forward(model, target_layer_name, file)
        row = np.zeros((1, channels))

        for channel in range(channels):
            feature_map = result[:, :, channel]
            avg = np.average(feature_map)
            row[0][channel] = avg

        activations = np.append(activations, row, axis=0)

    n_examples.append(feature_examples)

#se clasifican los ejemplos usando knn
file_number = 0
files = list(absoluteFilePaths(input_path))

for file in files:
    print("file " + str(file_number) + " of " + str(len(files)))
    result = partial_forward(model, target_layer_name, file)
    input_activation = np.zeros((1, channels))

    for channel in range(channels):
        feature_map = result[:, :, channel]
        avg = np.average(feature_map)
        input_activation[0][channel] = avg

    nearest_neighbors = knn(input_activation, activations, n_neighbors)
    nearest_classes = [features[i // n_examples[0]] for i in [x[0] for x in nearest_neighbors]]
    output_class = most_frequent(nearest_classes)

    output_path = label_files_path + output_class + "/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    #copyfile(file, output_path + str(file_number + 8792) + ".jpg")
    copyfile(file, output_path + str(file_number) + ".jpg")
    file_number += 1
