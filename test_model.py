import cv2
import tensorflow as tf
import sys

sys.path.append("C:/Users/andyb/PycharmProjects/resnetRopaLowRes/images")
sys.path.append("C:/Users/andyb/PycharmProjects/resnetRopaLowRes")
sys.path.append("C:/Users/andyb/PycharmProjects/resnetRopaLowRes/datasets")

import datasets.data as data
import numpy
import argparse
import utils.configuration as conf
import utils.imgproc as imgproc
import skimage.color as color

from sklearn.metrics import classification_report

import os

def prepare(filepath, mean_image):
    width = 60
    height =80
    img_array = cv2.imread(filepath, cv2.IMREAD_UNCHANGED,)  # read in the image, convert to grayscale
    img_array = imgproc.toUINT8(img_array)
    img_array = imgproc.process_image(img_array, (height, width))
    #cv2.imshow("window", img_array)
    #cv2.waitKey()

    img_array = numpy.float32(img_array)

    new_array = cv2.resize(img_array, (width, height))  # resize image to match model's expected sizing



    res = new_array.reshape(-1, height, width, 3)

    res = res - mean_image
    return res  # return the image with shaping that TF wants.

model = tf.keras.models.load_model('C:/Users/andyb/PycharmProjects/kerasResnet/resnet50.model')
#print(model.layers)


#parser = argparse.ArgumentParser(description = "Train a simple cnn model")
#parser.add_argument("-config", type = str, help = "<str> configuration file", required = True)
#parser.add_argument("-name", type=str, help=" name of section in the configuration file", required = True)
#pargs = parser.parse_args()
#configuration_file = pargs.config
#configuration = conf.ConfigurationFile(configuration_file, pargs.name)

configuration_file = "configs/cnn.config"
configuration = conf.ConfigurationFile(configuration_file, "RES")

shape_file = os.path.join(configuration.get_data_dir(),"shape.dat")
mean_file = os.path.join(configuration.get_data_dir(), "mean.dat")
input_shape = numpy.fromfile(shape_file, dtype=numpy.int32)
mean_image = numpy.fromfile(mean_file, dtype=numpy.float32)
mean_image = numpy.reshape(mean_image, input_shape)

file = open("C:/Users/andyb/PycharmProjects/kerasResnet/data/mapping.txt")
mapping = file.readlines()
dict = list(map(lambda x : [x.split("\t")[0], x.split("\t")[1].split("\n")[0]], mapping))


#preparar val_dataset
number_of_classes = configuration.get_number_of_classes()
tfr_test_file = os.path.join(configuration.get_data_dir(), "test.tfrecords")
val_dataset = tf.data.TFRecordDataset(tfr_test_file)
val_dataset = val_dataset.map(lambda x : data.parser_tfrecord(x, input_shape, mean_image, number_of_classes));
target_indexes = list(val_dataset.map(lambda x, y: y).as_numpy_iterator())
target_indexes = list(map(lambda x : numpy.argmax(x, axis=-1), target_indexes))
val_dataset = val_dataset.batch(batch_size = 16)

labels = list(map(lambda x : int(x.split("\t")[1]), mapping))
classes = list(map(lambda x : x.split("\t")[0], mapping))
#classes = list(map(lambda x : mapping[x].split("\t")[0], labels))

#input = prepare("C:/Users/andyb/Pictures/bigImages/16936.jpg", mean_image);

input = val_dataset

output = model.predict(input)
prediction = numpy.argmax(output, axis=-1)
#print(output)
#for i in range (0, len(dict)):
    #if str(prediction[0]) == dict[i][1]:
        #print(prediction[0])
        #print(dict[i])
        #print(dict[i][0])
        #break

print(classification_report(target_indexes, prediction, labels=labels, target_names=classes))
report = classification_report(target_indexes, prediction, labels=labels, target_names=classes, output_dict=True)
print(report)

def topClasses(report, kClasses):
    classes = []
    for key in report:
        classes.append([key, report[key]["f1-score"]])
    classes.sort(key = (lambda x: float(x[1])), reverse=True)
    result = "Top " + str(kClasses) + " (f1-score):\n"
    for i in range(0, kClasses):
        result += classes[i][0] + " - " + str(classes[i][1]) + "\n"
    return result

print(topClasses(report, 10))




