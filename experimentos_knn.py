import numpy as np
import tensorflow as tf
import os
from inferencia import partial_forward, knn, most_frequent, absoluteFilePaths, divide_set, get_activations, save_activations, load_activations
from shutil import copyfile
from matplotlib import pyplot as plt

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
colores = ["rojo", "negro", "azul", "verde", "amarillo", "gris", "cafe", "rosado", "morado", "naranjo"]
texturas = ["cuadros", "rayas", "flores", "leopardo", "polka", "plano", "pata_de_gallo", "lentejuelas", "agryle", "paisley"]
#features = ["cuadros", "rayas", "plano"]
n_neighbors = [1, 3, 7, 10]
training_fraction = 0.8
tipo_de_atributo = "colores"
generar_activaciones = True

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

for feature_idx, feature in enumerate(features):
    paths = list(absoluteFilePaths(baseline_path + feature))
    feature_paths = divide_set(paths, training_fraction)
    feature_training_set = feature_paths[0]
    feature_validation_set = feature_paths[1]
    n_training_samples = len(feature_training_set)
    n_validation_samples = len(paths) - n_training_samples
    training_set += feature_training_set
    validation_set.append(feature_validation_set)

if generar_activaciones:
    activations = get_activations(model, training_set, target_layer_name, pooled=True)
    save_activations(activations, tipo_de_atributo, target_layer_name)
else:
    activations = load_activations(tipo_de_atributo, target_layer_name)

#inferencia utilizando knn
series_results_by_feature = []

for feature_index, feature_set in enumerate(validation_set):
    feature_results = np.zeros(len(n_neighbors))
    print("calculando vecinos de atributo " + features[feature_index] + " - " + "{:.1f}".format(
        feature_index * len(feature_set) * 100 / (len(feature_validation_set) * len(features))) + "%")

    for file_index, file in enumerate(feature_set):
        result = partial_forward(model, target_layer_name, file)
        result = np.expand_dims(result, axis=0)
        pooled_result = tf.keras.layers.GlobalAveragePooling2D()(result)

        for idx, k in enumerate(n_neighbors):
            specific_output_path = output_path + tipo_de_atributo + "/" + features[feature_index] + "/" + str(k) + "_neighbors/" + target_layer_name + "/errores/"

            if not os.path.exists(specific_output_path):
                os.makedirs(specific_output_path)

            nearest_neighbors = knn(pooled_result, activations, k)
            nearest_classes = [features[i // n_training_samples] for i in [x[0] for x in nearest_neighbors]]
            output_class = most_frequent(nearest_classes)

            if output_class == features[feature_index]:
                feature_results[idx] += 1
            else:
                copyfile(file, specific_output_path + output_class + "_" + str(feature_idx) + "_" + str(file_index) + ".png")


    feature_results /= n_validation_samples
    series_results_by_feature.append(feature_results)



x = np.arange(len(n_neighbors))
barWidth = 0.75 / len(features)
fig, ax = plt.subplots()

r1 = np.arange(len(n_neighbors))
barras_grafico = [r1]

for i in range(1, len(features)):
    ri = [x + barWidth for x in barras_grafico[i - 1]]
    barras_grafico.append(ri)

prop_iter = iter(plt.rcParams['axes.prop_cycle'])

for i in range(len(features)):
    plt.bar(barras_grafico[i], series_results_by_feature[i], color=next(prop_iter)['color'], width=barWidth, edgecolor='white', label=features[i])


plt.xlabel('Capas', fontweight='bold')
plt.ylabel('Exactitud', fontweight='bold')
plt.title("exactitud usando inferencia knn - " + target_layer_name + " - " + tipo_de_atributo)
plt.xticks([r + barWidth for r in range(len(n_neighbors))], n_neighbors)
plt.legend()

#plt.show()
plt.savefig(output_path + "accuracy_" + target_layer_name + "_" + tipo_de_atributo + ".png")
