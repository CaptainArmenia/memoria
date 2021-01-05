import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.preprocessing.image as image
import os
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from collections import Counter

#sns.set()

#funciones de distancia
def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def cos_distance(p, q):
    return 1 - (np.dot(p, q) / (np.linalg.norm(p) * np.linalg.norm(q)))

def euclidean_dist(p, q):
    return np.linalg.norm(p-q)

def absoluteFilePaths(directory):
   for dirpath,_,filenames in os.walk(directory):
       for f in filenames:
           yield os.path.abspath(os.path.join(dirpath, f))

#inferencia parcial
def partial_forward(model, target_layer_name, input_file):

    target_layer = model.get_layer(target_layer_name)
    func = K.function([model.input], target_layer.output[0])
    img = image.load_img(input_file)

    # se obtienen las activaciones del input
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    # input para resnet imagenet
    input = preprocess_input(x)
    return func(input)

#vecinos más cercanos
def knn(input, data_lake, k):
    min_distances = [cos_distance(input, x) for x in data_lake]
    min_distances_with_indices = list(enumerate(min_distances))
    min_distances_with_indices.sort(key=(lambda x: x[1]), reverse=False)
    output = [min_distances_with_indices[x] for x in range(k)]
    return output

def most_frequent(input_list):
    occurence_count = Counter(input_list)
    return occurence_count.most_common(1)[0][0]

#divide el dataset en proporcion entrenamiento/validacion
def divide_set(dataset, training_fraction):
    training_set = []
    validation_set = []
    training_set_len = len(dataset) * training_fraction

    for x in range(len(dataset)):
        if x < training_set_len:
            training_set.append(dataset[x])
        else:
            validation_set.append(dataset[x])

    return [training_set, validation_set]

#crea una lista con las activaciones de cada input
def get_activations(model, input_list, target_layer_name, pooled=False):
    target_layer = model.get_layer(target_layer_name)
    channels = target_layer.output_shape[-1]
    activations = np.empty((0, channels))

    for idx, input_file in enumerate(input_list):
        print("calculando activaciones - " + "{:.1f}".format(
            idx * 100 / (len(input_list))) + "%")
        result = partial_forward(model, target_layer_name, input_file)
        result = np.expand_dims(result, axis=0)

        if pooled:
            pooled_result = tf.keras.layers.GlobalAveragePooling2D()(result)
            activations = np.append(activations, pooled_result, axis=0)
        else:
            activations = np.append(activations, result, axis=0)

    return activations

#serializa y guarda las activaciones en un archivo
def save_activations(activations, tipo_de_atributo, target_layer_name):
    np.save('activations_data_' + tipo_de_atributo + "_" + target_layer_name + '.npy', activations)

#carga las activaciones a partir de un archivo
def load_activations(tipo_de_atributo, target_layer_name):
    try:
        return np.load('activations_data_' + tipo_de_atributo + "_" + target_layer_name + '.npy')
    except:
        print("No existen activaciones guardadas para " + tipo_de_atributo + " y capa " + target_layer_name)








