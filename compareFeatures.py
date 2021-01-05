#agrupa las clases que más peso le dan a cada mapa de características

import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.preprocessing.image as image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import utils.imgproc as imgproc
import os
import utils.configuration as conf
from tensorflow.python.framework.ops import disable_eager_execution


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


disable_eager_execution()

model = tf.keras.models.load_model('C:/Users/andyb/PycharmProjects/kerasResnet/resnet50.model', compile=True)

for layer in model.layers:
    print(layer.name)

configuration_file = "configs/cnn.config"
configuration = conf.ConfigurationFile(configuration_file, "RES")

shape_file = os.path.join(configuration.get_data_dir(), "shape.dat")
mean_file = os.path.join(configuration.get_data_dir(), "mean.dat")
input_shape = np.fromfile(shape_file, dtype=np.int32)
mean_image = np.fromfile(mean_file, dtype=np.float32)
mean_image = np.reshape(mean_image, input_shape)

file = open("C:/Users/andyb/PycharmProjects/kerasResnet/data/mapping.txt")
mapping = file.readlines()
dict = list(map(lambda t: [t.split("\t")[0], t.split("\t")[1].split("\n")[0]], mapping))

last_conv_layer = model.get_layer('conv5_block3_out')
pool = model.get_layer('predictions')


img_name = "3496"
img_path = "C:/Users/andyb/PycharmProjects/kerasResnet/images390p/" + img_name + ".jpg"
img = image.load_img(img_path, target_size=(input_shape[0], input_shape[1]))
x = image.img_to_array(img)

x = np.expand_dims(x, axis=0)

classes = [129]

#for i in range(0, 142):
    #classes.append(i)


usedFeatures = []

weights = np.array(pool.get_weights())[0]

for c in classes:

    output = model.output[:, c]

    for i in range(0, len(dict)):
        if str(c) == dict[i][1]:
            print(str(c) + ". " + dict[i][0])
            break

    grads = K.gradients(output, last_conv_layer.output)[0]

    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([x])

    #print("top-weights:")

    w = weights[:, c]
    weight_values_with_indexes = list(enumerate(w))
    weight_values_with_indexes.sort(key=(lambda t: t[1]), reverse=True)

    #print(weight_values_with_indexes)

    pooled_grads_values_with_indexes = list(enumerate(pooled_grads_value))
    pooled_grads_values_with_indexes.sort(key=(lambda t: t[1]), reverse=True)

    #print("top-grads:")

    #print(pooled_grads_value_with_indexes)
    #print("")
    topFeatures = 20

    for i in range(0, topFeatures):
        index = pooled_grads_values_with_indexes[i][0]
        usedFeatures.append([index, c])

#print(usedFeatures)

usedFeatures.sort(key=(lambda t: t[0]))

#print(usedFeatures)

common_features = []
temp_list = [usedFeatures[0]]
previous_feature = usedFeatures[0][0]

for i in range(1, len(usedFeatures)):
    current_feature = usedFeatures[i][0]
    if current_feature == previous_feature:
        temp_list.append(usedFeatures[i])
    else:
        if len(temp_list) > 1:
            common_features.append(temp_list)
        temp_list = []
    previous_feature = current_feature

#print(common_features)
grouped_features = []
for f in common_features:
    common_classes = []
    for c in f:
        common_classes.append(c[1])
    grouped_features.append([f[0][0], common_classes])

for f in grouped_features:
    for k in range(0, len(f[1])):
        for i in range(0, len(dict)):
            if str(f[1][k]) == dict[i][1]:
                f[1][k] = dict[i][0]
                break

for f in grouped_features:
    print(str(f[0]) + ": " + str(f[1]))
