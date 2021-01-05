import numpy as np
import tensorflow as tf
import os
from inferencia import partial_forward, knn, most_frequent, absoluteFilePaths, divide_set, get_activations
import numpy as np
import tensorflow as tf

import os
from inferencia import partial_forward, knn, most_frequent, absoluteFilePaths, divide_set, get_activations, save_activations, load_activations
from shutil import copyfile
from matplotlib import pyplot as plt, image as mpimg

#configuraciones

input_shape = (224, 224, 3)
model = tf.keras.applications.resnet50.ResNet50(include_top= True, classifier_activation="softmax", weights= 'imagenet', input_shape= input_shape, classes=1000)
target_layer_name = "conv4_block6_out"
#target_layer_name = "conv5_block3_out"
#target_layer_name = "conv3_block4_out"
#target_layer_name = "conv2_block3_out"
#target_layer_name = "conv2_block1_out"
baseline_path = "C:/Users/andyb/OneDrive/Documentos/memoria/datasets/LISTO/"
output_path = "C:/Users/andyb/OneDrive/Documentos/memoria/graficos/resultados_knn/"
input_files = [
    "C:/Users/andyb/OneDrive/Documentos/memoria/datasets/LISTO/leopardo/65.png"
]
colores = ["rojo", "negro", "azul", "verde", "amarillo", "gris", "cafe", "rosado", "morado", "naranjo"]
texturas = ["cuadros", "rayas", "flores", "leopardo", "polka", "plano", "pata_de_gallo", "lentejuelas", "agryle", "paisley"]
n_neighbors = [1, 3, 7, 10]
training_fraction = 0.8
tipo_de_atributo = "texturas"
generar_activaciones = False

#se crean clusters de ejemplo

if tipo_de_atributo == "texturas":
    features = texturas
elif tipo_de_atributo == "colores":
    features = colores
else:
    raise Exception("tipo de atributo incorrecto")

n_training_samples = 0
n_validation_samples = 0
activations = []
training_set = []
validation_set = []
feature_samples = 0

#se dividen los archivos en entrenamiento/validacion
for feature_idx, feature in enumerate(features):
    paths = list(absoluteFilePaths(baseline_path + feature))
    feature_paths = divide_set(paths, training_fraction)
    feature_training_set = feature_paths[0]
    feature_validation_set = feature_paths[1]
    n_training_samples = len(feature_training_set)
    n_validation_samples = len(paths) - n_training_samples
    training_set += feature_training_set
    validation_set.append(feature_validation_set)

#se cargan o se generan las activaciones
if generar_activaciones:
    activations = get_activations(model, training_set, target_layer_name, pooled=True)
    save_activations(activations, tipo_de_atributo, target_layer_name)
else:
    activations = load_activations(tipo_de_atributo, target_layer_name)

#inferencia utilizando knn
series_results_by_feature = []

for input_file in input_files:
    file_activations = get_activations(model, [input_file], target_layer_name, pooled=True)[0]
    result = knn(file_activations, activations, 5)
    for x in result:
        img = mpimg.imread(training_set[x[0]])
        plt.imshow(img)
        plt.show()

