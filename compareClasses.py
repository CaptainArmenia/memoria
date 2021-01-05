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

    #res = res - mean_image
    return res  # return the image with shaping that TF wants.

disable_eager_execution()

#model = tf.keras.models.load_model('C:/Users/andyb/PycharmProjects/kerasResnet/resnet101.model', compile = True)
#model = tf.keras.models.load_model('C:/Users/andyb/PycharmProjects/kerasResnet/resnet50-390p.model', compile = True)


configuration_file = "configs/cnn.config"
configuration = conf.ConfigurationFile(configuration_file, "RES")

shape_file = os.path.join(configuration.get_data_dir(),"shape.dat")
mean_file = os.path.join(configuration.get_data_dir(), "mean.dat")
input_shape = np.fromfile(shape_file, dtype=np.int32)
mean_image = np.fromfile(mean_file, dtype=np.float32)
mean_image = np.reshape(mean_image, input_shape)

model = tf.keras.applications.ResNet101(include_top= True, classifier_activation="softmax", weights= 'imagenet', input_shape= (224, 224, 3), classes=1000)

last_conv_layer = model.get_layer('conv5_block3_out')

file = open("C:/Users/andyb/PycharmProjects/kerasResnet/data/mapping.txt")
mapping = file.readlines()
dict = list(map(lambda x : [x.split("\t")[0], x.split("\t")[1].split("\n")[0]], mapping))

img_names = ["2190", "1613", "2574", "2673"]

usedFeatures = []

for name in img_names:

    img_name = name
    img_path = "C:/Users/andyb/PycharmProjects/kerasResnet/images390p/" + img_name + ".jpg"
    img = image.load_img(img_path, target_size=(input_shape[0], input_shape[1]))
    x = image.img_to_array(img)

    x = np.expand_dims(x, axis=0)

    input = prepare(img_path, mean_image, 224, 224);
    #input = prepare(img_path, mean_image, input_shape[0], input_shape[1]);
    preds = model.predict(input)

    argmax = np.argmax(preds[0])
    print(argmax)
    output = model.output[:, argmax]


    prediction = np.argmax(preds, axis=-1)

    for i in range (0, len(dict)):
        if str(prediction[0]) == dict[i][1]:
            print(dict[i][0])
            break


    grads = K.gradients(output, last_conv_layer.output)[0]

    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input], [grads, pooled_grads, last_conv_layer.output[0]])
    grads_value, pooled_grads_value, conv_layer_output_value = iterate([x])

    pooled_grads_value_with_indexes = list(enumerate(pooled_grads_value))
    pooled_grads_value_with_indexes.sort(key=(lambda x: x[1]), reverse=True)

    topFeatures = 20

    for i in range(0, topFeatures):
        index = pooled_grads_value_with_indexes[i][0]
        usedFeatures.append([index, name])
        heatmap = conv_layer_output_value[:, :, index]

        for x in range(heatmap.shape[0]):
            for y in range(heatmap.shape[1]):
                heatmap[x, y] = np.max(heatmap[x, y], 0)

        heatmap = np.maximum(heatmap, 0)
        heatmap_max = np.max(heatmap)
        if(heatmap_max != 0):
            heatmap /= np.max(heatmap)

        img = cv2.imread(img_path)
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        hif = .8
        superimposed_img = heatmap * hif + img*0.5

        # Write some Text

        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (int(img.shape[1]/30), int(img.shape[0]/15))
        fontScale = img.shape[1]/400
        fontColor = (0, 0, 0)
        lineType = 2
        thickness = 4

        cv2.putText(superimposed_img, "feature_map_" + str(index),
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)

        output = 'C:/Users/andyb/PycharmProjects/kerasResnet/camOutputs/image_' + img_name + '_cam_feature_' + str(index) + '.jpeg'
        cv2.imwrite(output, superimposed_img)

usedFeatures.sort(key=(lambda t: t[0]))
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


