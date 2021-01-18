import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import tensorflow.keras.backend as K
import seaborn as sns
import os
from tqdm import tqdm
import utils.configuration as conf
import umap
from inferencia import absoluteFilePaths, get_activations, save_activations, load_activations, knn
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
import hdbscan

sns.set()

#configuraciones

#activation_poolings = ["avg", "max"]
activation_poolings = ["avg"]
#target_layer_names = ["conv2_block3_out", "conv3_block4_out", "conv4_block6_out", "conv5_block3_out"]
#distance_functions = ["kullback_leiber", "cos_similarity", "euclidean_dist"]
distance_functions = ["cos_similarity"]

n_ejemplos = 100
input_shape = (224, 224, 3)
feature_type = "colores"
generar_activaciones = False

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
baseline_path = "C:/Users/andyb/OneDrive/Documentos/memoria/datasets/LISTO/"
output_path = "C:/Users/andyb/OneDrive/Documentos/memoria/graficos/experimentos_embedding/"

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
target_layer = model.get_layer(target_layer_name)
channels = target_layer.output_shape[-1]
forward = K.function([model.input], target_layer.output[0])
#activaciones de todas las imagenes

image_set = []

#se obtienen los paths de los archivos del dataset
for feature_idx, feature in enumerate(features):
    paths = list(absoluteFilePaths(baseline_path + feature))
    image_set += paths

#se cargan o se generan las activaciones
if generar_activaciones:
    activations = get_activations(model, image_set, target_layer_name, pooled=True)
    save_activations(activations, feature_type, target_layer_name, "embedding_classification")
else:
    try:
        activations = load_activations(feature_type, target_layer_name, "embedding_classification")
    except Exception as e:
        print("No se pudo cargar activaciones")
        print("Generando nuevas activaciones")
        activations = get_activations(model, image_set, target_layer_name, pooled=True)
        save_activations(activations, feature_type, target_layer_name, "embedding_classification")

reducer = umap.UMAP()
embedding = reducer.fit_transform(activations)
nn = knn(embedding[540], embedding, 5, "euclid")

plt.rcParams["axes.grid"] = False
for i, x in enumerate(nn):
    path = image_set[x[0]]
    img = mpimg.imread((path))
    imgplot = plt.imshow(img)
    plt.savefig(output_path + str(i) + ".jpg")
    plt.show()


plt.rcParams["axes.grid"] = True
prop_iter = iter(plt.rcParams['axes.prop_cycle'])
fig, ax = plt.subplots()

for j in range(len(features)):
    x = embedding[n_ejemplos * j: n_ejemplos * (j + 1), 0]
    y = embedding[n_ejemplos * j: n_ejemplos * (j + 1):, 1]
    ax.scatter(
        x,
        y,
        c=[sns.color_palette()[x // n_ejemplos] for x in list(map((lambda x: x[0]), enumerate(embedding)))[n_ejemplos * j: n_ejemplos * (j + 1)]],
        label=features[j],
        alpha=0.3
    )

x = [embedding[k[0]][0] for k in nn]
y = [embedding[k[0]][1] for k in nn]
ax.scatter(
    x,
    y,
    s=150,
    facecolors='none',
    edgecolors='yellow'
)


plt.gca().set_aspect('equal', 'datalim')
plt.title('Proyección UMAP sobre activaciones en capa ' + target_layer_name, fontsize=12)
plt.legend()
ax.grid(True)
plt.savefig(output_path + "proyeccion_umap_" + feature_type + "_" + target_layer_name + ".png")
plt.show()




