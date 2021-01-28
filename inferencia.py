import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.preprocessing.image as image
import os
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from collections import Counter
from tqdm import tqdm
import utils.configuration as conf
import cv2
import utils.imgproc as imgproc

#archivo de configuraciones
configuration_file = "configs/cnn.config"
configuration = conf.ConfigurationFile(configuration_file, "RES")
shape_file = os.path.join(configuration.get_data_dir(),"shape.dat")
mean_file = os.path.join(configuration.get_data_dir(), "mean.dat")
input_shape = np.fromfile(shape_file, dtype=np.int32)
mean_image = np.fromfile(mean_file, dtype=np.float32)
mean_image = np.reshape(mean_image, input_shape)

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

def prepare(filepath, mean_image, h, w):
    width = w
    height = h
    img_array = cv2.imread(filepath)  # read in the image
    img_array = imgproc.toUINT8(img_array)
    img_array = imgproc.process_image(img_array, (height, width))
    img_array = np.float32(img_array)
    new_array = cv2.resize(img_array, (width, height))  # resize image to match model's expected sizing
    res = new_array.reshape(-1, height, width, 3)

    res = res - mean_image
    return res  # return the image with shaping that TF wants.

#inferencia parcial
def partial_forward(model, weights, target_layer_name, input_file):
    target_layer = model.get_layer(target_layer_name)
    func = K.function([model.input], target_layer.output[0])
    img = image.load_img(input_file)

    # se obtienen las activaciones del input
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    if weights == "imagenet":
        # input para resnet imagenet
        input = preprocess_input(x)
    else:
        # input para keras resnet
        input = prepare(input_file, mean_image, input_shape[0], input_shape[1])
    return func(input)

#vecinos más cercanos
def knn(input, data_lake, k, func):
    if func == "cos":
        min_distances = [cos_distance(input, x) for x in data_lake]
    elif func == "euclid":
        min_distances = [euclidean_dist(input, x) for x in data_lake]
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
def get_activations(model, weights, input_list, target_layer_name, pooled=False):
    pbar = tqdm(total=len(input_list))
    target_layer = model.get_layer(target_layer_name)
    channels = target_layer.output_shape[-1]
    activations = np.empty((0, channels))

    for idx, input_file in enumerate(input_list):
        result = partial_forward(model, weights, target_layer_name, input_file)
        result = np.expand_dims(result, axis=0)

        if pooled:
            pooled_result = tf.keras.layers.GlobalAveragePooling2D()(result)
            activations = np.append(activations, pooled_result, axis=0)
        else:
            activations = np.append(activations, result, axis=0)

        pbar.update(1)

    return activations

#wrapper - serializa y guarda las activaciones en un archivo
def save_activations(activations, path):
    np.save(path, activations)

#wrapper - carga las activaciones a partir de un archivo
def load_activations(path):
    return np.load(path)









