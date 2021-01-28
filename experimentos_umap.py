import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K
import seaborn as sns
import os
from tqdm import tqdm
import utils.configuration as conf
import umap
from inferencia import absoluteFilePaths, get_activations, load_activations, save_activations
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
import hdbscan

sns.set()

#configuraciones

#target_layer_names = ["conv2_block3_out", "conv3_block4_out", "conv4_block6_out", "conv5_block3_out"]
#features = ["cuadros", "rayas", "flores", "rosado", "texto", "negro", "leopardo", "verde", "amarillo", "azul"]
#features = ["cuadros", "rayas", "flores", "texto", "leopardo", "denim", "rombos", "satin", "puntos", "plano"]
#features = ["cuadros", "rayas", "flores", "leopardo", "polka", "plano", "paisley", "agryle", "pata_de_gallo", "lentejuelas"]
#features = ["cuadros", "rayas"]
n_ejemplos = 100
input_shape = (224, 224, 3)
feature_type = "colores"
n_neighbors = 10
generar_activaciones = False
weights = "imagenet"

if feature_type == "colores":
    target_layer_name = "conv2_block3_out"
    features = ["rojo", "negro", "azul", "verde", "amarillo", "gris", "cafe", "rosado", "morado", "naranjo"]
elif feature_type == "texturas":
    target_layer_name = "conv4_block6_out"
    features = ["cuadros", "rayas", "flores", "leopardo", "polka", "plano", "paisley", "agryle", "pata_de_gallo",
                "lentejuelas"]
else:
    raise Exception("tipo de atributo incorrecto")

#input_path = "C:/Users/andyb/OneDrive/Documentos/memoria/distance_val_dataset/"
input_path = "C:/Users/andyb/OneDrive/Documentos/memoria/datasets/LISTO/"
output_path = "C:/Users/andyb/OneDrive/Documentos/memoria/graficos/experimentos_embedding/"

#disable_eager_execution()
if weights == "imagenet":
    model = tf.keras.applications.resnet50.ResNet50(include_top= True, classifier_activation="softmax", weights= 'imagenet', input_shape= input_shape, classes=1000)
elif weights == "kaggle":
    model = tf.keras.models.load_model('C:/Users/andyb/PycharmProjects/kerasResnet/resnet50.model', compile = True)
else:
    raise Exception("Modelo invalido")

for layer in model.layers:
    print(layer.name)

configuration_file = "configs/cnn.config"
configuration = conf.ConfigurationFile(configuration_file, "RES")

shape_file = os.path.join(configuration.get_data_dir(),"shape.dat")
mean_file = os.path.join(configuration.get_data_dir(), "mean.dat")
input_shape = np.fromfile(shape_file, dtype=np.int32)
mean_image = np.fromfile(mean_file, dtype=np.float32)
mean_image = np.reshape(mean_image, input_shape)

image_set = []

#se obtienen los paths de los archivos del dataset
for feature_idx, feature in enumerate(features):
    paths = list(absoluteFilePaths(input_path + feature))
    image_set += paths

#se cargan o se generan las activaciones
if generar_activaciones:
    activations = get_activations(model, weights, image_set, target_layer_name, pooled=True)
    save_activations(activations, feature_type + "_" + target_layer_name +  "_embedding_activations_" + weights + ".npy")
else:
    try:
        activations = load_activations(feature_type + "_" + target_layer_name +  "_embedding_activations_" + weights + ".npy")
    except Exception as e:
        print("No se pudo cargar activaciones")
        print("Generando nuevas activaciones")
        activations = get_activations(model, weights, image_set, target_layer_name, pooled=True)
        save_activations(activations, feature_type + "_" + target_layer_name +  "_embedding_activations_" + weights + ".npy")

targets = [x // n_ejemplos for x in range(0, n_ejemplos * len(features))]
targets = np.array(targets)

reducer = umap.UMAP(n_neighbors=10)
embedding = reducer.fit_transform(activations)
prop_iter = iter(plt.rcParams['axes.prop_cycle'])
fig, ax = plt.subplots()
labels = hdbscan.HDBSCAN(min_samples=10, min_cluster_size=50,).fit_predict(embedding)
a_rand_score = adjusted_rand_score(targets, labels)
a_mi_score = adjusted_mutual_info_score(targets, labels)

for j in range(len(features)):
    ax.scatter(
        embedding[n_ejemplos * j: n_ejemplos * (j + 1), 0],
        embedding[n_ejemplos * j: n_ejemplos * (j + 1):, 1],
        c=[sns.color_palette()[x // n_ejemplos] for x in list(map((lambda x: x[0]), enumerate(activations)))[n_ejemplos * j: n_ejemplos * (j + 1)]],
        label=features[j],
        alpha=0.3
    )

plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP sobre activaciones en capa ' + target_layer_name + " - " + weights + " - nn = " + str(n_neighbors), fontsize=12)
plt.legend()
plt.text(1, 0, 'adjusted-rand-score: {:.1f}'.format(a_rand_score),
         rotation=0,
         horizontalalignment='right',
         verticalalignment='top',
         transform=ax.transAxes)
plt.text(1, 0.05, 'adjusted-MI-score: {:.1f}'.format(a_mi_score),
         rotation=0,
         horizontalalignment='right',
         verticalalignment='top',
         transform=ax.transAxes)

ax.grid(True)
plt.savefig(output_path + "proyeccion_umap_" + weights + "_" + feature_type + "_" + str(n_neighbors) + "_neighbors_" + target_layer_name + ".png")
plt.show()





