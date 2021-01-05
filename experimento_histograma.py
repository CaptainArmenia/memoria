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
import random
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

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

input_shape = (224, 224, 3)

model = tf.keras.applications.resnet50.ResNet50(include_top= True, classifier_activation="softmax", weights= 'imagenet', input_shape= input_shape, classes=1000)

for layer in model.layers:
    print(layer.name)

atributo = "flores"

images = os.listdir("C:/Users/andyb/OneDrive/Documentos/memoria/5_50_feature_dataset/" + atributo)
random.shuffle(images)

target_layer_name = "conv5_block3_out"
target_layer = model.get_layer(target_layer_name)
channels = target_layer.output_shape[-1]
forward = K.function([model.input], target_layer.output[0])


activations = np.zeros(channels)

for i in range(0, 50):
    print(str(i) + "%")
    img_name = images[i].split(".")[0]
    img_path = "C:/Users/andyb/OneDrive/Documentos/memoria/5_50_feature_dataset/" + atributo + "/" + img_name + ".png"
    #print(img_name)
    img = image.load_img(img_path, target_size=(input_shape[0], input_shape[1]))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    input = preprocess_input(x)

    pred = model.predict(input)
    print('Predicted:', decode_predictions(pred, top=3)[0])

    result = forward(input)

    for feature in range(channels):

        feature_map = result[:, :, feature]
        avg = np.average(feature_map)
        activations[feature] += avg

activations_with_indexes = list(enumerate(activations))
activations_with_indexes.sort(key=(lambda x: x[1]), reverse=True)

top_channels = 5

lines = plt.plot(activations)
ax = lines[0].axes
plt.ylabel('activación acumulada')
plt.xlabel('canales')
extraticks = []
lim = ax.get_xlim()

for i in range(top_channels):
    extraticks.append(activations_with_indexes[i][0])

plt.xticks(extraticks)
ax.set_xlim(lim)

plt.suptitle("bloque: " + target_layer_name.split("conv")[1].split("_")[0] + " - atributo: " + atributo)
plt.savefig("graficos/activacion_acumulada_" + atributo + "_bloque_" + target_layer_name.split("conv")[1].split("_")[0] + ".png")
plt.show()
