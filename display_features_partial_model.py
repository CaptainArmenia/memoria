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

model = tf.keras.models.load_model('C:/Users/andyb/PycharmProjects/kerasResnet/analysis_resnet.model', compile = True)



configuration_file = "configs/cnn.config"
configuration = conf.ConfigurationFile(configuration_file, "RES")

shape_file = os.path.join(configuration.get_data_dir(),"shape.dat")
mean_file = os.path.join(configuration.get_data_dir(), "mean.dat")
input_shape = np.fromfile(shape_file, dtype=np.int32)
print("input shape:")
print(input_shape)
mean_image = np.fromfile(mean_file, dtype=np.float32)
mean_image = np.reshape(mean_image, input_shape)

images = os.listdir("C:/Users/andyb/PycharmProjects/kerasResnet/images390p/")
random.shuffle(images)

for i in range(0, 100):
    print(str(i + 1) + "%")
    img_name = images[i].split(".")[0]
    img_path = "C:/Users/andyb/PycharmProjects/kerasResnet/images390p/" + img_name + ".jpg"

    img = image.load_img(img_path, target_size=(input_shape[0], input_shape[1]))
    x = image.img_to_array(img)

    x = np.expand_dims(x, axis=0)


    #for layer in model.layers:
    #    print(layer.name)


    input = prepare(img_path, mean_image, input_shape[0], input_shape[1])

    #model.summary()

    binary_mode = True
    threshold = 0.5
    max_activation = 0
    heatmaps = []

    for feature in range(512):

        heatmap = model.predict(input)[0][:, :, feature]
        heatmap_max = np.max(heatmap)
       # if heatmap_max > max_activation:
       #     max_activation = heatmap_max
       #heatmaps.append(heatmap)

    #for feature in range(512):

       # heatmap = heatmaps[i]
        if heatmap_max > 0:
            heatmap /= heatmap_max
        heatmap = np.where(heatmap > threshold, 1.0, 0.0)
        #heatmap = heatmap - threshold
        #
        #heatmap = np.maximum(heatmap, 0)
        #if heatmap_max > 0:
        #    heatmap /= heatmap_max

        img = cv2.imread(img_path)
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask = heatmap
        masks = [mask, mask, mask]
        mask = np.stack(masks, axis=2)

        heatmap = np.uint8(255 * heatmap)
        result = heatmap
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        hif = .8
        superimposed_img = heatmap * hif + img*0.5

        # Write some Text

        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (int(img.shape[1]/30), int(img.shape[0]/15))
        fontScale = img.shape[1]/400
        fontColor = (255, 255, 255)
        lineType = 2
        thickness = 4

        if binary_mode:
            superimposed_img = mask * img

        cv2.putText(superimposed_img, "feature_map_" + str(feature),
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)

        if not os.path.exists('C:/Users/andyb/PycharmProjects/kerasResnet/camOutputs/feature_' + str(feature)):
            os.makedirs('C:/Users/andyb/PycharmProjects/kerasResnet/camOutputs/feature_' + str(feature))
        output = 'C:/Users/andyb/PycharmProjects/kerasResnet/camOutputs/feature_' + str(feature) + '/image_' + img_name + '.jpeg'
        cv2.imwrite(output, superimposed_img)

        #img=mpimg.imread(output)
        #plt.imshow(img)
        #plt.show()
