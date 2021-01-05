import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, \
    Add, AveragePooling2D, Flatten, Dense
from tensorflow.keras.models import Model
import utils.configuration as conf

configuration_file = "configs/cnn.config"
configuration = conf.ConfigurationFile(configuration_file, "RES")


def relu_bn(inputs: Tensor) -> Tensor:
    relu = ReLU()(inputs)
    bn = BatchNormalization()(relu)
    return bn


def residual_block(x: Tensor, downsample: bool, filters: int, kernel_size: int = 3) -> Tensor:
    y = Conv2D(kernel_size=kernel_size,
               strides=(1 if not downsample else 2),
               filters=filters,
               padding="same")(x)
    y = relu_bn(y)
    y = Conv2D(kernel_size=kernel_size,
               strides=1,
               filters=filters,
               padding="same")(y)

    if downsample:
        x = Conv2D(kernel_size=1,
                   strides=2,
                   filters=filters,
                   padding="same")(x)
    out = Add()([x, y])
    out = relu_bn(out)
    return out


def create_res_net(input_shape, block_list, num_classes):
    inputs = Input(shape=input_shape)
    num_filters = 64

    t = BatchNormalization()(inputs)
    t = Conv2D(kernel_size=3,
               strides=1,
               filters=num_filters,
               padding="same")(t)
    t = relu_bn(t)

    num_blocks_list = block_list
    for i in range(len(num_blocks_list)):
        num_blocks = num_blocks_list[i]
        for j in range(num_blocks):
            t = residual_block(t, downsample=(j == 0 and i != 0), filters=num_filters)
        num_filters *= 2

    t = tf.keras.layers.GlobalAveragePooling2D()(t)
    t = Flatten()(t)
    outputs = Dense(num_classes, activation='softmax')(t)

    model = Model(inputs, outputs)

    #initial_learning_rate = configuration.get_learning_rate()
    #lr_schedule = tf.keras.experimental.CosineDecay(initial_learning_rate=initial_learning_rate,
                                                    #decay_steps=300000,
                                                    #alpha=0.0001)

    #opt = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9, nesterov=True)
    #opt = tf.keras.optimizers.Adam(learning_rate=configuration.get_learning_rate()) # 'adam'

    #model.compile(
        #optimizer=opt,
        # optimizer=tf.keras.optimizers.Adam(learning_rate = configuration.get_learning_rate()), # 'adam'
        # loss= losses.crossentropy_loss,
        #loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        #metrics=['accuracy'])

    return model