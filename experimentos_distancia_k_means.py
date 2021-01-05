import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.preprocessing.image as image
import seaborn as sns
import os
import cv2
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import utils.imgproc as imgproc
from tensorflow.python.framework.ops import disable_eager_execution
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import utils.configuration as conf

sns.set()


#configuraciones

#activation_poolings = ["avg", "max"]
activation_poolings = ["avg"]
target_layer_names = ["conv2_block3_out", "conv3_block4_out", "conv4_block6_out", "conv5_block3_out"]
#distance_functions = ["kullback_leiber", "cos_similarity", "euclidean_dist"]
distance_functions = ["cos_similarity"]
features = ["cuadros", "rayas", "flores", "rosado", "texto", "negro", "leopardo", "verde", "amarillo", "azul"]
k_means_clusters = [1, 2, 3, 4, 5]
input_shape = (224, 224, 3)
epsilon = 0.00000001
output_path = "C:/Users/andyb/OneDrive/Documentos/memoria/graficos/resultados_distancia_distribuciones_k_means_"

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

accuracy_by_pooling_function = []
accuracy_by_pooling_function_by_layer_by_distance_function_by_feature = []

avance = 0

for n_clusters in k_means_clusters:

    accuracy_by_layer = []
    accuracy_by_layer_by_distance_function_by_feature = []

    for target_layer_name in target_layer_names:

        target_layer = model.get_layer(target_layer_name)
        channels = target_layer.output_shape[-1]
        forward = K.function([model.input], target_layer.output[0])

        layer_accuracy_by_distance_function = []
        layer_accuracy_by_distance_function_by_feature = []

        for distance_function in distance_functions:

            #generar distribuciones para cada atributo

            distributions = []

            print("generando distribuciones para capa " + target_layer_name + " usando " + str(n_clusters) + " y distancia " + distance_function + " y pooling avg")
            print(str(int(avance)) + "%")
            avance += 100 / (len(distance_functions) * len(activation_poolings) * len(target_layer_names * len(k_means_clusters)))

            for feature in features:

                #se recorre un conjunto de imagenes, registrando las activaciones de cada canal en una tabla
                activations = np.empty((0, channels))
                images = os.listdir("C:/Users/andyb/OneDrive/Documentos/memoria/distance_baseline_dataset/" + feature)
                for i in range(0, len(images)):
                    #random.shuffle(images)
                    img_name = images[i].split(".")[0]
                    img_path = "C:/Users/andyb/OneDrive/Documentos/memoria/distance_baseline_dataset/" + feature + "/" + img_name + ".png"
                    #print(img_name)
                    img = image.load_img(img_path, target_size=(input_shape[0], input_shape[1]))
                    x = image.img_to_array(img)
                    x = np.expand_dims(x, axis=0)
                    input = preprocess_input(x)

                    #pred = model.predict(input)
                    #print('Predicted:', decode_predictions(pred, top=3)[0])

                    result = forward(input)
                    row = np.zeros((1,channels))

                    for channel in range(channels):

                        feature_map = result[:, :, channel]
                        avg = np.average(feature_map)
                        row[0][channel] = avg

                    activations = np.append(activations, row, axis=0)

                # se evitan ceros en caso de divergencia-kl
                if distance_function == "kullback_leiber":
                    activations += epsilon

                #normalizar distribucion
                activations /= np.sum(activations)
                distributions.append(activations)

            #calcular centroides de cada atributo con k-means

            centroids = []

            for i in range(len(features)):

                feature_activations = distributions[i]
                kmeans = KMeans(n_clusters=n_clusters).fit(feature_activations)
                feature_centroids = kmeans.cluster_centers_

                for centroid in feature_centroids:

                    centroids.append(centroid)

            #experimentos

            input_sizes_by_feature = []
            accuracy_by_feature = []
            correct_predictions_by_feature = []
            correct_feature_index = -1

            for feature in features:

                input_path = "C:/Users/andyb/OneDrive/Documentos/memoria/distance_val_dataset/" + feature + "/"
                specific_output_path = output_path + str(n_clusters) + "_clusters" + "/" + target_layer_name + "/" + distance_function + "/" + feature + "/"

                if not os.path.exists(specific_output_path):
                    os.makedirs(specific_output_path)

                predictions = np.zeros(len(features))
                input_size = len(os.listdir(input_path))
                input_sizes_by_feature.append(input_size)
                correct_feature_index += 1

                for k in range(1, input_size + 1):
                    input_file = input_path + str(k) + ".png"
                    img = image.load_img(input_file, target_size=(input_shape[0], input_shape[1]))

                    #se obtienen las activaciones del input

                    x = image.img_to_array(img)
                    x = np.expand_dims(x, axis=0)

                    #input para resnet imagenet
                    input = preprocess_input(x)

                    #input para resnet custom
                    #input = prepare(input_file, mean_image, input_shape[0], input_shape[1])

                    #pred = model.predict(input)
                    result = forward(input)

                    input_activations = np.zeros(channels)

                    for c in range(channels):

                        feature_map = result[:, :, c]
                        avg = np.average(feature_map)
                        input_activations[c] += avg


                    #se evitan ceros en caso de divergencia-kl
                    if distance_function == "kullback_leiber":
                        input_activations += epsilon

                    #se normaliza la distribucion
                    input_distribution = input_activations / np.sum(input_activations)

                    #se calcula la distancia entre las distribuciones
                    if distance_function == "kullback_leiber":
                        min_distance = kl_divergence(input_distribution, centroids[0])

                    elif distance_function == "cos_similarity":
                        min_distance = 1 - cos_similarity(input_distribution, centroids[0])

                    elif distance_function == "euclidean_dist":
                        min_distance = euclidean_dist(input_distribution, centroids[0])
                    else:
                        raise Exception("Función de distancia inválida (kullback_leiber/cos_similarity/euclidean_dist)")

                    predicted_centroid_index = 0

                    for j in range(0, len(centroids)):
                        if distance_function == "kullback_leiber":
                            distance = kl_divergence(input_distribution, centroids[j])

                        elif distance_function == "cos_similarity":
                            distance = 1 - cos_similarity(input_distribution, centroids[j])

                        elif distance_function == "euclidean_dist":
                            distance = euclidean_dist(input_distribution, centroids[j])
                        else:
                            raise Exception("Función de distancia inválida (kullback_leiber/cos_similarity/euclidean_dist)")

                        #print("distancia entre distribucion de atributo " + features[k] + " y distribucion del input: " + str(distance))
                        if distance < min_distance:
                            min_distance = distance
                            predicted_centroid_index = j

                    predicted_feature_index = predicted_centroid_index // n_clusters

                    predictions[predicted_feature_index] += 1

                    wrong_prediction_path = specific_output_path + "/wrong_predictions/"

                    if not os.path.exists(wrong_prediction_path):
                        os.makedirs(wrong_prediction_path)

                    if predicted_feature_index != correct_feature_index:
                        image.save_img(wrong_prediction_path + features[predicted_feature_index] + "_" + str(k) + ".png", image.img_to_array(img))

                    #print("atributo predicho: " + features[predicted_feature_index])

                fig = plt.figure()
                plt.bar(features, predictions)
                plt.ylabel("Número de predicciones")
                plt.xlabel("Atributo")
                plt.title(feature + " - " + distance_function + " - " + target_layer_name + " - avg" )

                plt.savefig(specific_output_path + "accuracy.png")

                current_feature_index = features.index(feature)
                correct_predictions = predictions[current_feature_index]
                correct_predictions_by_feature.append(correct_predictions)
                accuracy = correct_predictions / input_size
                accuracy_by_feature.append(accuracy)


            layer_accuracy = sum(correct_predictions_by_feature) / sum(input_sizes_by_feature)
            layer_accuracy_by_distance_function.append(layer_accuracy)
            layer_accuracy_by_distance_function_by_feature.append(accuracy_by_feature)

        accuracy_by_layer.append(layer_accuracy_by_distance_function)
        accuracy_by_layer_by_distance_function_by_feature.append(layer_accuracy_by_distance_function_by_feature)

    accuracy_by_pooling_function.append(accuracy_by_layer)
    accuracy_by_pooling_function_by_layer_by_distance_function_by_feature.append(accuracy_by_layer_by_distance_function_by_feature)

for i in range (0, len(activation_poolings)):


    #kullback_leiber_accuracies = [pooling_accuracy[0][0], pooling_accuracy[1][0], pooling_accuracy[2][0], pooling_accuracy[3][0]]

    #euclidean_dist_accuracies = [pooling_accuracy[0][2], pooling_accuracy[1][2], pooling_accuracy[2][2], pooling_accuracy[3][2]]

    x = np.arange(len(target_layer_names))
    barWidth = 0.75 / len(k_means_clusters)
    fig, ax = plt.subplots()

    r1 = np.arange(len(target_layer_names))
    barras_grafico = [r1]
    for u in range(1, len(k_means_clusters)):
        ri = [x + barWidth for x in barras_grafico[u - 1]]
        barras_grafico.append(ri)

    prop_iter = iter(plt.rcParams['axes.prop_cycle'])
    for v in range(len(k_means_clusters)):
        pooling_accuracy = accuracy_by_pooling_function[v]
        cos_similarity_accuracies = [pooling_accuracy[0][0], pooling_accuracy[1][0], pooling_accuracy[2][0],
                                     pooling_accuracy[3][0]]
        plt.bar(barras_grafico[v], cos_similarity_accuracies, color=next(prop_iter)['color'], width=barWidth,
                edgecolor='white', label=str(k_means_clusters[v]) + "_clusters")

    plt.xlabel('Capas', fontweight='bold')
    plt.ylabel('Exactitud', fontweight='bold')
    plt.title('Exactitud de cada bloque - ' + activation_poolings[i] + ' pooling - ' + str(k_means_clusters) + " clusters")
    plt.xticks([r + barWidth for r in range(len(target_layer_names))], target_layer_names)
    plt.legend()

    plt.savefig(output_path + "exactitud_segun_bloque_segun_clusters.png")
    #plt.show()

cluster_number_results = 1
avg_pooling_accuracy = accuracy_by_pooling_function_by_layer_by_distance_function_by_feature[cluster_number_results - 1]
cos_similarity_accuracies = []

for k in range(len(features)):

    cos_similarity_accuracies.append([avg_pooling_accuracy[0][0][k], avg_pooling_accuracy[1][0][k], avg_pooling_accuracy[2][0][k], avg_pooling_accuracy[3][0][k]])


x = np.arange(len(target_layer_names))
barWidth = 0.75 / len(features)
fig, ax = plt.subplots()

r1 = np.arange(len(target_layer_names))
barras_grafico = [r1]
for i in range(1, len(features)):
    ri = [x + barWidth for x in barras_grafico[i - 1]]
    barras_grafico.append(ri)

prop_iter = iter(plt.rcParams['axes.prop_cycle'])
for i in range(len(features)):

    plt.bar(barras_grafico[i], cos_similarity_accuracies[i], color=next(prop_iter)['color'], width=barWidth, edgecolor='white', label=features[i])


plt.xlabel('Capas', fontweight='bold')
plt.ylabel('Exactitud', fontweight='bold')
plt.title('exactitud de cada bloque por atributo - avg pooling - distancia coseno - ' + str(cluster_number_results - 1) + " clusters")
plt.xticks([r + barWidth for r in range(len(target_layer_names))], target_layer_names)
plt.legend()

plt.savefig(output_path + "exactitud_segun_bloque_segun_atributo" + ".png")
#plt.show()











