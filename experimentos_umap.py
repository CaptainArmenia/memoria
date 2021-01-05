import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K
import seaborn as sns
import os
from tensorflow.python.framework.ops import disable_eager_execution
import utils.configuration as conf
import umap
from inferencia import absoluteFilePaths, partial_forward

sns.set()

#configuraciones

#activation_poolings = ["avg", "max"]
activation_poolings = ["avg"]
target_layer_names = ["conv2_block3_out", "conv3_block4_out", "conv4_block6_out", "conv5_block3_out"]
#target_layer_names = ["conv5_block3_out"]
#distance_functions = ["kullback_leiber", "cos_similarity", "euclidean_dist"]
distance_functions = ["cos_similarity"]
#features = ["cuadros", "rayas", "flores", "rosado", "texto", "negro", "leopardo", "verde", "amarillo", "azul"]
#features = ["cuadros", "rayas", "flores", "texto", "leopardo", "denim", "rombos", "satin", "puntos", "plano"]
#features = ["rojo", "negro", "azul", "verde", "amarillo", "gris", "cafe", "rosado", "morado", "naranjo"]
features = ["cuadros", "rayas", "flores", "leopardo", "polka", "plano", "paisley", "agryle", "pata_de_gallo", "lentejuelas"]
n_ejemplos = 100

input_shape = (224, 224, 3)


#input_path = "C:/Users/andyb/OneDrive/Documentos/memoria/distance_val_dataset/"
input_path = "C:/Users/andyb/OneDrive/Documentos/memoria/datasets/LISTO/"
output_path = "C:/Users/andyb/OneDrive/Documentos/memoria/graficos/"

disable_eager_execution()
model = tf.keras.applications.resnet50.ResNet50(include_top= True, classifier_activation="softmax", weights= 'imagenet', input_shape= input_shape, classes=1000)
#model = tf.keras.models.load_model('C:/Users/andyb/PycharmProjects/kerasResnet/resnet50.model', compile = True)

for layer in model.layers:
    print(layer.name)

configuration_file = "configs/cnn.config"
configuration = conf.ConfigurationFile(configuration_file, "RES")

shape_file = os.path.join(configuration.get_data_dir(),"shape.dat")
mean_file = os.path.join(configuration.get_data_dir(), "mean.dat")
input_shape = np.fromfile(shape_file, dtype=np.int32)
mean_image = np.fromfile(mean_file, dtype=np.float32)
mean_image = np.reshape(mean_image, input_shape)

layer_activations = []

for target_layer_name in target_layer_names:
    target_layer = model.get_layer(target_layer_name)
    channels = target_layer.output_shape[-1]
    forward = K.function([model.input], target_layer.output[0])
    #activaciones de todas las imagenes
    activations = np.empty((0, channels))

    for feature in features:
        # se recorre un conjunto de imagenes, registrando las activaciones de cada canal en una tabla
        images = list(absoluteFilePaths(input_path + feature))
        contador = 0

        for img_path in images:
            result = partial_forward(model, target_layer_name, img_path)
            print(target_layer_name + " - " + feature + " - " + (contador / len(images)) + "%")
            row = np.zeros((1, channels))

            for channel in range(channels):
                feature_map = result[:, :, channel]
                avg = np.average(feature_map)
                row[0][channel] = avg

            activations = np.append(activations, row, axis=0)
            contador += 1

    layer_activations.append(activations)

for k in range(len(layer_activations)):

    reducer = umap.UMAP()
    embedding = reducer.fit_transform(layer_activations[k])
    prop_iter = iter(plt.rcParams['axes.prop_cycle'])

    fig, ax = plt.subplots()
    for j in range(len(features)):
        ax.scatter(
            embedding[n_ejemplos * j: n_ejemplos * (j + 1), 0],
            embedding[n_ejemplos * j: n_ejemplos * (j + 1):, 1],
            c=[sns.color_palette()[x // n_ejemplos] for x in list(map((lambda x: x[0]), enumerate(layer_activations[k])))[n_ejemplos * j: n_ejemplos * (j + 1)]],
            label=features[j]
        )
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('Proyección UMAP sobre activaciones en capa ' + target_layer_names[k], fontsize=12)
    plt.legend()
    ax.grid(True)
    plt.savefig(output_path + "proyeccion_umap_texturas_" + target_layer_names[k] + ".png")
    #plt.show()

