import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.preprocessing.image as image
import seaborn as sns
import os
import cv2
import utils.imgproc as imgproc
from tensorflow.python.framework.ops import disable_eager_execution
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import utils.configuration as conf

sns.set()

def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def cos_similarity(p, q):
    return np.dot(p, q) / (np.linalg.norm(p) * np.linalg.norm(q))

def euclidean_dist(p, q):
    return np.linalg.norm(p-q)

def prepare(filepath, mean_image, h, w):
    width = w
    height = h
    img_array = cv2.imread(filepath, cv2.IMREAD_UNCHANGED,)  # read in the image, convert to grayscale
    img_array = imgproc.toUINT8(img_array)
    img_array = imgproc.process_image(img_array, (height, width))

    img_array = np.float32(img_array)

    new_array = cv2.resize(img_array, (width, height))  # resize image to match model's expected sizing

    res = new_array.reshape(-1, height, width, 3)

    res = res - mean_image
    return res  # return the image with shaping that TF wants.

#configuraciones

activation_poolings = ["avg", "max"]
target_layer_names = ["conv2_block3_out", "conv3_block4_out", "conv4_block6_out", "conv5_block3_out"]
distance_functions = ["kullback_leiber", "cos_similarity", "euclidean_dist"]
features = ["cuadros", "rayas", "flores", "rosado", "texto"]

#input_shape = (224, 224, 3)
epsilon = 0.00000001

output_path = "C:/Users/andyb/OneDrive/Documentos/memoria/graficos/resultados_distancia_distribuciones/"

#disable_eager_execution()
#model = tf.keras.applications.resnet50.ResNet50(include_top= True, classifier_activation="softmax", weights= 'imagenet', input_shape= input_shape, classes=1000)
model = tf.keras.models.load_model('C:/Users/andyb/PycharmProjects/kerasResnet/resnet50.model', compile = True)
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

for activation_pooling in activation_poolings:

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

            print("generando distribuciones para capa " + target_layer_name + " usando distancia " + distance_function + " y pooling " + activation_pooling)
            print(str(int(avance)) + "%")
            avance += 100 / (len(distance_functions) * len(activation_poolings) * len(target_layer_names))

            for feature in features:

                #se recorre un conjunto de imagenes, acumulando las activaciones de cada canal
                activations = np.zeros(channels)

                for i in range(0, len(os.listdir("C:/Users/andyb/OneDrive/Documentos/memoria/5_50_feature_dataset/" + feature))):
                    images = os.listdir("C:/Users/andyb/OneDrive/Documentos/memoria/5_50_feature_dataset/" + feature)
                    #random.shuffle(images)
                    img_name = images[i].split(".")[0]
                    img_path = "C:/Users/andyb/OneDrive/Documentos/memoria/5_50_feature_dataset/" + feature + "/" + img_name + ".png"
                    #print(img_name)
                    img = image.load_img(img_path, target_size=(input_shape[0], input_shape[1]))
                    x = image.img_to_array(img)
                    x = np.expand_dims(x, axis=0)
                    input = preprocess_input(x)

                    pred = model.predict(input)
                    #print('Predicted:', decode_predictions(pred, top=3)[0])

                    result = forward(input)

                    for channel in range(channels):

                        feature_map = result[:, :, channel]

                        if activation_pooling == "max":
                            max = np.max(feature_map)
                            activations[channel] += max

                        elif activation_pooling == "avg":
                            avg = np.average(feature_map)
                            activations[channel] += avg

                        else:
                            raise Exception("función de pooling incorrecta (avg/max)")

                # se evitan ceros en caso de divergencia-kl
                if distance_function == "kullback_leiber":
                    activations += epsilon

                #normalizar distribucion
                activations /= sum(activations)
                distributions.append(activations)

            #experimentos

            input_sizes_by_feature = []
            accuracy_by_feature = []
            correct_predictions_by_feature = []

            for feature in features:

                input_path = "C:/Users/andyb/OneDrive/Documentos/memoria/distance_val_dataset/" + feature + "/"

                predictions = np.zeros(len(features))
                input_size = len(os.listdir(input_path))
                input_sizes_by_feature.append(input_size)

                for k in range(1, input_size + 1):
                    input_file = input_path + str(k) + ".png"
                    img = image.load_img(input_file, target_size=(input_shape[0], input_shape[1]))

                    #se obtienen las activaciones del input

                    x = image.img_to_array(img)
                    x = np.expand_dims(x, axis=0)

                    #input para resnet imagenet
                    #input = preprocess_input(x)

                    #input para resnet custom
                    input = prepare(input_file, mean_image, input_shape[0], input_shape[1])

                    pred = model.predict(input)
                    result = forward(input)

                    input_activations = np.zeros(channels)

                    for c in range(channels):

                        feature_map = result[:, :, c]

                        if activation_pooling == "max":
                            max = np.max(feature_map)
                            input_activations[c] += max

                        elif activation_pooling == "avg":
                            avg = np.average(feature_map)
                            input_activations[c] += avg

                        else:
                            raise Exception("función de pooling incorrecta (avg/max)")

                    #se evitan ceros en caso de divergencia-kl
                    if distance_function == "kullback_leiber":
                        input_activations += epsilon

                    #se normaliza la distribucion
                    input_distribution = input_activations / sum(input_activations)

                    #se calcula la distancia entre las distribuciones
                    if distance_function == "kullback_leiber":
                        min_distance = kl_divergence(input_distribution, distributions[0])

                    elif distance_function == "cos_similarity":
                        min_distance = 1 - cos_similarity(input_distribution, distributions[0])

                    elif distance_function == "euclidean_dist":
                        min_distance = euclidean_dist(input_distribution, distributions[0])
                    else:
                        raise Exception("Función de distancia inválida (kullback_leiber/cos_similarity/euclidean_dist)")

                    predicted_feature_index = 0

                    for k in range(0, len(distributions)):
                        if distance_function == "kullback_leiber":
                            distance = kl_divergence(input_distribution, distributions[k])

                        elif distance_function == "cos_similarity":
                            distance = 1 - cos_similarity(input_distribution, distributions[k])

                        elif distance_function == "euclidean_dist":
                            distance = euclidean_dist(input_distribution, distributions[k])
                        else:
                            raise Exception("Función de distancia inválida (kullback_leiber/cos_similarity/euclidean_dist)")

                        #print("distancia entre distribucion de atributo " + features[k] + " y distribucion del input: " + str(distance))
                        if distance < min_distance:
                            min_distance = distance
                            predicted_feature_index = k

                    predictions[predicted_feature_index] += 1
                    #print("atributo predicho: " + features[predicted_feature_index])

                fig = plt.figure()
                plt.bar(features, predictions)
                plt.ylabel("Número de predicciones")
                plt.xlabel("Atributo")
                plt.title(feature + " - " + distance_function + " - " + target_layer_name + " - " + activation_pooling)

                if not os.path.exists(output_path + target_layer_name):
                    os.makedirs(output_path + target_layer_name)

                plt.savefig(output_path + target_layer_name + "/" + feature + "_" + distance_function + "_" + activation_pooling + ".png")

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

    pooling_accuracy = accuracy_by_pooling_function[i]
    kullback_leiber_accuracies = [pooling_accuracy[0][0], pooling_accuracy[1][0], pooling_accuracy[2][0], pooling_accuracy[3][0]]
    cos_similarity_accuracies = [pooling_accuracy[0][1], pooling_accuracy[1][1], pooling_accuracy[2][1], pooling_accuracy[3][1]]
    euclidean_dist_accuracies = [pooling_accuracy[0][2], pooling_accuracy[1][2], pooling_accuracy[2][2], pooling_accuracy[3][2]]

    x = np.arange(len(target_layer_names))
    barWidth = 0.25
    fig, ax = plt.subplots()

    r1 = np.arange(len(target_layer_names))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]

    plt.bar(r1, kullback_leiber_accuracies, color='#7f6d5f', width=barWidth, edgecolor='white', label='kl_div')
    plt.bar(r2, cos_similarity_accuracies, color='#557f2d', width=barWidth, edgecolor='white', label='cos_sim')
    plt.bar(r3, euclidean_dist_accuracies, color='#2d7f5e', width=barWidth, edgecolor='white', label='euclid_dist')

    plt.xlabel('Capas', fontweight='bold')
    plt.ylabel('Exactitud', fontweight='bold')
    plt.title('Bondad discriminatoria de cada bloque - ' + activation_poolings[i] + ' pooling')
    plt.xticks([r + barWidth for r in range(len(target_layer_names))], target_layer_names)
    plt.legend()

    plt.savefig(output_path + "exactitud_segun_bloque_" + activation_poolings[i] + ".png")
    plt.show()

avg_pooling_accuracy = accuracy_by_pooling_function_by_layer_by_distance_function_by_feature[0]
cos_similarity_accuracies = []

for k in range(len(features)):

    cos_similarity_accuracies.append([avg_pooling_accuracy[0][1][k], avg_pooling_accuracy[1][1][k], avg_pooling_accuracy[2][1][k], avg_pooling_accuracy[3][1][k]])


x = np.arange(len(target_layer_names))
barWidth = 0.15
fig, ax = plt.subplots()

r1 = np.arange(len(target_layer_names))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]
r5 = [x + barWidth for x in r4]

plt.bar(r1, cos_similarity_accuracies[0], color='#7f6d5f', width=barWidth, edgecolor='white', label=features[0])
plt.bar(r2, cos_similarity_accuracies[1], color='#FFB433', width=barWidth, edgecolor='white', label=features[1])
plt.bar(r3, cos_similarity_accuracies[2], color='#2d7f5e', width=barWidth, edgecolor='white', label=features[2])
plt.bar(r4, cos_similarity_accuracies[3], color='#336DFF', width=barWidth, edgecolor='white', label=features[3])
plt.bar(r5, cos_similarity_accuracies[4], color='#FF3386', width=barWidth, edgecolor='white', label=features[4])

plt.xlabel('Capas', fontweight='bold')
plt.ylabel('Exactitud', fontweight='bold')
plt.title('Bondad discriminatoria de cada bloque por atributo - avg pooling - distancia coseno')
plt.xticks([r + barWidth for r in range(len(target_layer_names))], target_layer_names)
plt.legend()

plt.savefig(output_path + "exactitud_segun_bloque_segun_atributo" + activation_poolings[i] + ".png")
plt.show()











