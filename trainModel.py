import sys

# change to ".../CC7221/convnet"
print(sys.path)
import tensorflow as tf
import datasets.data as data
import utils.configuration as conf
import numpy as np
import os
import resnet_functional_model

if __name__ == '__main__':
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    configuration_file = "configs/cnn.config"
    configuration = conf.ConfigurationFile(configuration_file, "RES")

    tfr_train_file = os.path.join(configuration.get_data_dir(), "train.tfrecords")
    tfr_test_file = os.path.join(configuration.get_data_dir(), "test.tfrecords")
    mean_file = os.path.join(configuration.get_data_dir(), "mean.dat")
    shape_file = os.path.join(configuration.get_data_dir(), "shape.dat")
    
    input_shape = np.fromfile(shape_file, dtype=np.int32)
    mean_image = np.fromfile(mean_file, dtype=np.float32)
    mean_image = np.reshape(mean_image, input_shape)

    number_of_classes = configuration.get_number_of_classes()

    tr_dataset = tf.data.TFRecordDataset(tfr_train_file)
    tr_dataset = tr_dataset.map(lambda x: data.parser_tfrecord(x, input_shape, mean_image, number_of_classes));
    tr_dataset = tr_dataset.shuffle(10000)
    tr_dataset = tr_dataset.batch(batch_size=configuration.get_batch_size())
    # tr_dataset = tr_dataset.repeat()

    val_dataset = tf.data.TFRecordDataset(tfr_test_file)
    val_dataset = val_dataset.map(lambda x: data.parser_tfrecord(x, input_shape, mean_image, number_of_classes));
    val_dataset = val_dataset.batch(batch_size=configuration.get_batch_size())

    # DigitModel is instantiated
    #model = tf.keras.applications.ResNet50V2(weights=None, input_shape= tuple(input_shape))
    #model = tf.keras.applications.ResNet101(include_top= True, classifier_activation="softmax", weights= None, input_shape= tuple(input_shape), classes=number_of_classes)
    #model = tf.keras.applications.resnet50.ResNet50(include_top = True, weights = None, input_tensor = None, input_shape = tuple(input_shape), classes = number_of_classes)
    #model = tf.keras.applications.vgg16.VGG16(include_top=True, weights=None, input_tensor=None, input_shape=tuple(input_shape), classes = number_of_classes)
    # build the model indicating the input shape
    #model.build(input_shape = (1, input_shape[0], input_shape[1], input_shape[2]))
    model = resnet_functional_model.create_res_net(input_shape, [3, 4, 6, 3], number_of_classes)
    model.summary()

    # define the training parameters
    # Here, you can test SGD vs Adam
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=configuration.get_learning_rate(), amsgrad=True),  # 'adam'
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    history = model.fit(tr_dataset,
                        epochs=configuration.get_number_of_epochs(),
                        # steps_per_epoch = 100,
                        validation_data=val_dataset,
                        validation_steps=configuration.get_validation_steps()),

    # save the model
    model.save("resnet18_functional.model")
