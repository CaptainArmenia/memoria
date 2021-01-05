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

model = tf.keras.models.load_model('C:/Users/andyb/PycharmProjects/kerasResnet/resnet101.model', compile = True)
#model = tf.keras.models.load_model('C:/Users/andyb/PycharmProjects/kerasResnet/resnet50-390p.model', compile = True)


configuration_file = "configs/cnn.config"
configuration = conf.ConfigurationFile(configuration_file, "RES")

shape_file = os.path.join(configuration.get_data_dir(),"shape.dat")
mean_file = os.path.join(configuration.get_data_dir(), "mean.dat")
input_shape = np.fromfile(shape_file, dtype=np.int32)
mean_image = np.fromfile(mean_file, dtype=np.float32)
mean_image = np.reshape(mean_image, input_shape)

img_name = "2276"
img_path = "C:/Users/andyb/PycharmProjects/kerasResnet/images390p/" + img_name + ".jpg"
#img_path = "C:/Users/andyb/Desktop/cartera.jpg"
img = image.load_img(img_path, target_size=(input_shape[0], input_shape[1]))
x = image.img_to_array(img)

x = np.expand_dims(x, axis=0)


for layer in model.layers:
    print(layer.name)
target_layer = model.get_layer('conv5_block3_out')

input = prepare(img_path, mean_image, input_shape[0], input_shape[1]);
preds = model.predict(input)
argmax = np.argmax(preds[0])
output = model.output[:, argmax]

grads = K.gradients(output, target_layer.output)[0]
pooled_grads = K.mean(grads, axis=(0, 1, 2))

iterate = K.function([model.input], [pooled_grads, target_layer.output[0]])
pooled_grads_value, target_layer_output_value = iterate([x])

pooled_grads_value_with_indexes = list(enumerate(pooled_grads_value))
pooled_grads_value_with_indexes.sort(key=(lambda x: x[1]), reverse=True)
print(pooled_grads_value_with_indexes)
topFeatures = 20
print(len(pooled_grads_value_with_indexes))



for i in range(0, topFeatures):

    index = pooled_grads_value_with_indexes[i][0]
    feature = target_layer_output_value[:, :, index]
    feature = np.maximum(feature, 0)
    max = np.max(feature)
    if(max != 0):
        feature /= np.max(feature)

    feature = np.uint8(255 * feature)
    #feature = cv2.applyColorMap(feature, cv2.COLORMAP_JET)

    plt.imshow(feature)
    plt.show()
    plt.axis('off')
    output = 'C:/Users/andyb/PycharmProjects/kerasResnet/camOutputs/image_' + img_name + '_feature_' + str(
        index) + '.jpeg'
    cv2.imwrite(output, feature)