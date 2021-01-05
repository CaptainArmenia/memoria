# author: jsaavedr
# April, 2020 
# This is py model for cnn

import tensorflow as tf

class ResnetModel(tf.keras.Model):
    def __init__(self):
        super(ResnetModel, self).__init__()

        #input layer

        #self.input_layer = tf.keras.layers.InputLayer()
        self.input_layer = tf.keras.layers.InputLayer(input_shape = (80, 60, 3))

        #define layers which require parameters to be learned
        self.bn_input = tf.keras.layers.BatchNormalization()

        self.conv_0 = tf.keras.layers.Conv2D(64, (3,3), strides= 2, padding = 'same',  kernel_initializer = 'he_normal')
        self.bn_conv_0 = tf.keras.layers.BatchNormalization()

        self.res_0 = tf.keras.layers.Conv2D(64, (1,1), strides= 1, padding = 'same',  kernel_initializer = 'he_normal')

        self.conv_1 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')
        self.bn_conv_1 = tf.keras.layers.BatchNormalization()
        self.conv_2 = tf.keras.layers.Conv2D(64, (3,3), padding = 'same',  kernel_initializer='he_normal')
        self.bn_conv_2 = tf.keras.layers.BatchNormalization()


        self.res_1 = tf.keras.layers.Conv2D(64, (1, 1), strides=1, padding='same', kernel_initializer='he_normal')

        self.conv_3 = tf.keras.layers.Conv2D(64, (3,3), padding = 'same', kernel_initializer='he_normal')
        self.bn_conv_3 = tf.keras.layers.BatchNormalization()
        self.conv_4 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')
        self.bn_conv_4 = tf.keras.layers.BatchNormalization()

        self.res_2 = tf.keras.layers.Conv2D(64, (1, 1), strides=2, padding='same', kernel_initializer='he_normal')

        self.conv_5 = tf.keras.layers.Conv2D(64, (3, 3), strides=2, padding='same', kernel_initializer='he_normal')
        self.bn_conv_5 = tf.keras.layers.BatchNormalization()
        self.conv_6 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')
        self.bn_conv_6 = tf.keras.layers.BatchNormalization()

        self.res_3 = tf.keras.layers.Conv2D(64, (1, 1), strides=1, padding='same', kernel_initializer='he_normal')

        self.conv_7 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')
        self.bn_conv_7 = tf.keras.layers.BatchNormalization()
        self.conv_8 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')
        self.bn_conv_8 = tf.keras.layers.BatchNormalization()

        self.res_4 = tf.keras.layers.Conv2D(128, (1, 1), strides=2, padding='same', kernel_initializer='he_normal')

        self.conv_9 = tf.keras.layers.Conv2D(128, (3, 3), strides=2, padding='same', kernel_initializer='he_normal')
        self.bn_conv_9 = tf.keras.layers.BatchNormalization()
        self.conv_10 = tf.keras.layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')
        self.bn_conv_10 = tf.keras.layers.BatchNormalization()

        self.res_5 = tf.keras.layers.Conv2D(128, (1, 1), strides=1, padding='same', kernel_initializer='he_normal')

        self.conv_11 = tf.keras.layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')
        self.bn_conv_11 = tf.keras.layers.BatchNormalization()
        self.conv_12 = tf.keras.layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')
        self.bn_conv_12 = tf.keras.layers.BatchNormalization()

        self.res_6 = tf.keras.layers.Conv2D(128, (1, 1), strides=2, padding='same', kernel_initializer='he_normal')

        self.conv_13 = tf.keras.layers.Conv2D(128, (3, 3), strides=2, padding='same', kernel_initializer='he_normal')
        self.bn_conv_13 = tf.keras.layers.BatchNormalization()
        self.conv_14 = tf.keras.layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')
        self.bn_conv_14 = tf.keras.layers.BatchNormalization()

        self.res_7 = tf.keras.layers.Conv2D(128, (1, 1), strides=1, padding='same', kernel_initializer='he_normal')

        self.conv_15 = tf.keras.layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')
        self.bn_conv_15 = tf.keras.layers.BatchNormalization()
        self.conv_16 = tf.keras.layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')
        self.bn_conv_16 = tf.keras.layers.BatchNormalization()

        self.res_8 = tf.keras.layers.Conv2D(256, (1, 1), strides=2, padding='same', kernel_initializer='he_normal')

        self.conv_17 = tf.keras.layers.Conv2D(256, (3, 3), strides=2, padding='same', kernel_initializer='he_normal')
        self.bn_conv_17 = tf.keras.layers.BatchNormalization()
        self.conv_18 = tf.keras.layers.Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal')
        self.bn_conv_18 = tf.keras.layers.BatchNormalization()

        self.res_9 = tf.keras.layers.Conv2D(256, (1, 1), strides=1, padding='same', kernel_initializer='he_normal')

        self.conv_19 = tf.keras.layers.Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal')
        self.bn_conv_19 = tf.keras.layers.BatchNormalization()
        self.conv_20 = tf.keras.layers.Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal')
        self.bn_conv_20 = tf.keras.layers.BatchNormalization()

        self.relu = tf.keras.layers.ReLU()
        self.gap = tf.keras.layers.GlobalAveragePooling2D()

        self.fc1 = tf.keras.layers.Dense(256, kernel_initializer='he_normal')
        self.bn_fc_1 = tf.keras.layers.BatchNormalization()

        self.fc2 = tf.keras.layers.Dense(142)
        
    # here the architecture is defined
    def call(self, inputs):

        _input = self.input_layer(inputs)
        input_bn = self.bn_input(_input)
        _conv0 = self.conv_0(input_bn)
        _conv0 = self.bn_conv_0(_conv0)
        _conv0 = self.relu(_conv0)

        #residual branch
        _res0 = self.res_0(_conv0)

        #first block

        _conv1 = self.conv_1(_conv0)
        _conv1 = self.bn_conv_1(_conv1)
        _conv1 = self.relu(_conv1)
        _conv2 = self.conv_2(_conv1)
        _conv2 = self.bn_conv_2(_conv2)
        _conv2 = _conv2 + _res0
        _conv2 = self.relu(_conv2)

        # residual branch
        _res1 = self.res_1(_conv2)

        _conv3 = self.conv_3(_conv2)
        _conv3 = self.bn_conv_3(_conv3)
        _conv3 = self.relu(_conv3)
        _conv4 = self.conv_4(_conv3)
        _conv4 = self.bn_conv_4(_conv4)
        _conv4 = _conv4 + _res1
        _conv4 = self.relu(_conv4)

        #second block

        # residual branch
        _res2= self.res_2(_conv4)

        _conv5 = self.conv_5(_conv4)
        _conv5 = self.bn_conv_5(_conv5)
        _conv5 = self.relu(_conv5)
        _conv6 = self.conv_6(_conv5)
        _conv6 = self.bn_conv_6(_conv6)
        _conv6 = _conv6 + _res2
        _conv6 = self.relu(_conv6)

        # residual branch
        _res3 = self.res_3(_conv6)

        _conv7 = self.conv_7(_conv6)
        _conv7 = self.bn_conv_7(_conv7)
        _conv7 = self.relu(_conv7)
        _conv8 = self.conv_8(_conv7)
        _conv8 = self.bn_conv_8(_conv8)
        _conv8 = _conv8 + _res3
        _conv8 = self.relu(_conv8)

        # residual branch
        _res4 = self.res_4(_conv8)

        #third block

        _conv9 = self.conv_9(_conv8)
        _conv9 = self.bn_conv_9(_conv9)
        _conv9 = self.relu(_conv9)
        _conv10 = self.conv_10(_conv9)
        _conv10 = self.bn_conv_10(_conv10)
        _conv10 = _conv10 + _res4
        _conv10 = self.relu(_conv10)

        # residual branch
        _res5 = self.res_5(_conv10)

        _conv11 = self.conv_11(_conv10)
        _conv11 = self.bn_conv_11(_conv11)
        _conv11 = self.relu(_conv11)
        _conv12 = self.conv_12(_conv11)
        _conv12 = self.bn_conv_12(_conv12)
        _conv12 = _conv12 + _res5
        _conv12 = self.relu(_conv12)

        #fourth block

        # residual branch
        _res6 = self.res_6(_conv12)

        _conv13 = self.conv_13(_conv12)
        _conv13 = self.bn_conv_13(_conv13)
        _conv13 = self.relu(_conv13)
        _conv14 = self.conv_14(_conv13)
        _conv14 = self.bn_conv_14(_conv14)
        _conv14 = _conv14 + _res6
        _conv14 = self.relu(_conv14)

        # residual branch
        _res7 = self.res_7(_conv14)

        _conv15 = self.conv_15(_conv14)
        _conv15 = self.bn_conv_15(_conv15)
        _conv15 = self.relu(_conv15)
        _conv16 = self.conv_16(_conv15)
        _conv16 = self.bn_conv_16(_conv16)
        _conv16 = _conv16 + _res7
        _conv16 = self.relu(_conv16)

        #fifth block

        # residual branch
        _res8 = self.res_8(_conv16)

        _conv17 = self.conv_17(_conv16)
        _conv17 = self.bn_conv_17(_conv17)
        _conv17 = self.relu(_conv17)
        _conv18 = self.conv_18(_conv17)
        _conv18 = self.bn_conv_18(_conv18)
        _conv18 = _conv18 + _res8
        _conv18 = self.relu(_conv18)

        # residual branch
        _res9 = self.res_9(_conv18)

        _conv19 = self.conv_19(_conv18)
        _conv19 = self.bn_conv_19(_conv19)
        _conv19 = self.relu(_conv19)
        _conv20 = self.conv_20(_conv19)
        _conv20 = self.bn_conv_20(_conv20)
        _conv20 = _conv20 + _res9
        _conv20 = self.relu(_conv20)

        #last block        
        _gap = self.gap(_conv20)
        _fc1 = self.fc1(_gap)
        _fc1 = self.bn_fc_1(_fc1) #[B, 256]
        _fc1 = self.relu(_fc1) #[B, 10]
        
        output = self.fc2(_fc1)
        return output
