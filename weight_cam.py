import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.preprocessing.image as image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import utils.imgproc as imgproc
import argparse
import os
import utils.configuration as conf
from tensorflow.python.framework.ops import disable_eager_execution
import sys
import resnet_model


def prepare(filepath, mean_image, h, w):
    width = w
    height = h
    img_array = cv2.imread(filepath, cv2.IMREAD_UNCHANGED,)  # read in the image, convert to grayscale
    img_array = imgproc.toUINT8(img_array)
    img_array = imgproc.process_image(img_array, (height, width))
    #cv2.imshow("window", img_array)
    #cv2.waitKey()

    img_array = np.float32(img_array)

    new_array = cv2.resize(img_array, (width, height))  # resize image to match model's expected sizing



    res = new_array.reshape(-1, height, width, 3)

    res = res - mean_image
    return res  # return the image with shaping that TF wants.

disable_eager_execution()

model = tf.keras.models.load_model('C:/Users/andyb/PycharmProjects/kerasResnet/resnet50.model', compile = True)



configuration_file = "configs/cnn.config"
configuration = conf.ConfigurationFile(configuration_file, "RES")

shape_file = os.path.join(configuration.get_data_dir(),"shape.dat")
mean_file = os.path.join(configuration.get_data_dir(), "mean.dat")
input_shape = np.fromfile(shape_file, dtype=np.int32)
mean_image = np.fromfile(mean_file, dtype=np.float32)
mean_image = np.reshape(mean_image, input_shape)

img_name = "19254"
img_path = "C:/Users/andyb/PycharmProjects/kerasResnet/images390p/" + img_name + ".jpg"

img = image.load_img(img_path, target_size=(input_shape[0], input_shape[1]))
x = image.img_to_array(img)

x = np.expand_dims(x, axis=0)


for layer in model.layers:
    print(layer.name)
last_conv_layer = model.get_layer('conv5_block3_out')

input = prepare(img_path, mean_image, input_shape[0], input_shape[1]);
preds = model.predict(input)

#model.summary()

argmax = np.argmax(preds[0])
output = model.output[:, argmax]

file = open("C:/Users/andyb/PycharmProjects/kerasResnet/data/mapping.txt")
mapping = file.readlines()
dict = list(map(lambda x : [x.split("\t")[0], x.split("\t")[1].split("\n")[0]], mapping))
prediction = np.argmax(preds, axis=-1)

for i in range (0, len(dict)):
    if str(prediction[0]) == dict[i][1]:
        print(dict[i][0])
        break


pool = model.get_layer('predictions')
weights = np.array(pool.get_weights())[0]
topFeatures = 50
w = weights[:, argmax]
iterate = K.function([model.input], [last_conv_layer.output[0]])
conv_layer_output_value, = iterate([x])

for i in range(2048):
    conv_layer_output_value[:, :, i] *= w[i]

heatmap = np.mean(conv_layer_output_value, axis=-1)
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)

img = cv2.imread(img_path)
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
hif = .8
superimposed_img = heatmap * hif + img*0.5

output = 'C:/Users/andyb/PycharmProjects/kerasResnet/camOutputs/cam_output_' + img_name + '.jpeg'
cv2.imwrite(output, superimposed_img)

img=mpimg.imread(output)
plt.imshow(img)
plt.show()
plt.axis('off')
