
from keras.models import *
from keras.layers import *
from keras.layers.core import Activation, Reshape
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate

from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from src.metrics_add import *
from keras import backend as K

class MaxPoolingWithArgmax2D(Layer):
    def __init__(self, pool_size=(2, 1), strides=(2, 1), padding='same', **kwargs):
        super(MaxPoolingWithArgmax2D, self).__init__(**kwargs)
        self.padding = padding
        self.pool_size = pool_size
        self.strides = strides

    def call(self, inputs, **kwargs):
        padding = self.padding
        pool_size = self.pool_size
        strides = self.strides
        if K.backend() == 'tensorflow':
            ksize = [1, pool_size[0], pool_size[1], 1]
            padding = padding.upper()
            strides = [1, strides[0], strides[1], 1]
            output, argmax = K.tf.nn.max_pool_with_argmax(inputs, ksize=ksize, strides=strides, padding=padding)
        else:
            errmsg = '{} backend is not supported for layer {}'.format(K.backend(), type(self).__name__)
            raise NotImplementedError(errmsg)
        argmax = K.cast(argmax, K.floatx())
        return [output, argmax]

    def compute_output_shape(self, input_shape):
        ratio = (1, 2, 1, 1)
        output_shape = [dim // ratio[idx] if dim is not None else None for idx, dim in enumerate(input_shape)]
        output_shape = tuple(output_shape)
        return [output_shape, output_shape]

    def compute_mask(self, inputs, mask=None):
        return 2 * [None]


class MaxUnpooling2D(Layer):
    def __init__(self, size=(2, 1), **kwargs):
        super(MaxUnpooling2D, self).__init__(**kwargs)
        self.size = size

    def call(self, inputs, output_shape=None):
        updates, mask = inputs[0], inputs[1]
        with K.tf.variable_scope(self.name):
            mask = K.cast(mask, 'int32')
            input_shape = K.tf.shape(updates, out_type='int32')
            #  calculation new shape
            if output_shape is None:
                output_shape = (input_shape[0], input_shape[1] * self.size[0], input_shape[2] * self.size[1], input_shape[3])
            self.output_shape1 = output_shape

            # calculation indices for batch, height, width and feature maps
            one_like_mask = K.ones_like(mask, dtype='int32')
            batch_shape = K.concatenate([[input_shape[0]], [1], [1], [1]], axis=0)
            batch_range = K.reshape(K.tf.range(output_shape[0], dtype='int32'), shape=batch_shape)
            b = one_like_mask * batch_range
            y = mask // (output_shape[2] * output_shape[3])
            # if output_shape[2] == 1:
            #             #     x = (2*(mask // output_shape[3])+1) % (output_shape[2]+1) #当只有1时，任何数对1求余都为0，这是不对的
            #             # else:
            #             #     x = (mask // output_shape[3]) % output_shape[2]
            x = (mask // output_shape[3]) % output_shape[2]
            feature_range = K.tf.range(output_shape[3], dtype='int32')
            f = one_like_mask * feature_range

            # transpose indices & reshape update values to one dimension
            updates_size = K.tf.size(updates)
            indices = K.transpose(K.reshape(K.stack([b, y, x, f]), [4, updates_size]))
            values = K.reshape(updates, [updates_size])
            ret = K.tf.scatter_nd(indices, values, output_shape)
            return ret

    def compute_output_shape(self, input_shape):
        mask_shape = input_shape[1]
        return mask_shape[0], mask_shape[1] * self.size[0], mask_shape[2] * self.size[1], mask_shape[3]


alpha =0.1
def segunet(
        input_shape,
        n_labels=3,
        kernel=(3,1),
        pool_size=(2, 1),
        output_mode="softmax"):

    inputs = Input(shape=input_shape)
   # inputs = Dropout(0.1)(inputs)
    # encoder
    conv_1 = Convolution2D(8, kernel, padding="same")(inputs)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = LeakyReLU(alpha=alpha)(conv_1)
    conv_1 = Dropout(0.1)(conv_1)
    pool_1, mask_1 = MaxPoolingWithArgmax2D(pool_size)(conv_1)

    conv_2 = Convolution2D(12, kernel, padding="same")(pool_1)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = LeakyReLU(alpha=alpha)(conv_2)
    conv_2 = Dropout(0.1)(conv_2)
    pool_2, mask_2 = MaxPoolingWithArgmax2D(pool_size)(conv_2)

    conv_3 = Convolution2D(16, kernel, padding="same")(pool_2)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = LeakyReLU(alpha=alpha)(conv_3)
    conv_3 = Dropout(0.1)(conv_3)
    pool_3, mask_3 = MaxPoolingWithArgmax2D(pool_size)(conv_3)

    conv_4 = Convolution2D(24, kernel, padding="same")(pool_3)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = LeakyReLU(alpha=alpha)(conv_4)
    conv_4 = Dropout(0.1)(conv_4)
    pool_4, mask_4 = MaxPoolingWithArgmax2D(pool_size)(conv_4)

    conv_5 = Convolution2D(24, kernel, padding="same")(pool_4)
    conv_5 = BatchNormalization()(conv_5)
    conv_5 = LeakyReLU(alpha=alpha)(conv_5)
    conv_5 = Dropout(0.1)(conv_5)
    # decoder
    unpool_1 = MaxUnpooling2D(pool_size)([conv_5, mask_4])
    concat_1 = Concatenate()([unpool_1, conv_4])

    conv_6 = Convolution2D(24, kernel, padding="same")(concat_1)
    conv_6 = BatchNormalization()(conv_6)
    conv_6 = LeakyReLU(alpha=alpha)(conv_6)
    conv_6 = Dropout(0.1)(conv_6)
    conv_6 = Convolution2D(16, kernel, padding="same")(conv_6)
    conv_6 = BatchNormalization()(conv_6)
    conv_6 = LeakyReLU(alpha=alpha)(conv_6)

    unpool_2 = MaxUnpooling2D(pool_size)([conv_6, mask_3])
    concat_2 = Concatenate()([unpool_2, conv_3])

    conv_7 = Convolution2D(16, kernel, padding="same")(concat_2)
    conv_7 = BatchNormalization()(conv_7)
    conv_7 = LeakyReLU(alpha=alpha)(conv_7)
    conv_7 = Dropout(0.1)(conv_7)
    conv_7 = Convolution2D(12, kernel, padding="same")(conv_7)
    conv_7 = BatchNormalization()(conv_7)
    conv_7 = LeakyReLU(alpha=alpha)(conv_7)

    unpool_3 = MaxUnpooling2D(pool_size)([conv_7, mask_2])
    concat_3 = Concatenate()([unpool_3, conv_2])

    conv_8 = Convolution2D(12, kernel, padding="same")(concat_3)
    conv_8 = BatchNormalization()(conv_8)
    conv_8 = LeakyReLU(alpha=alpha)(conv_8)
    conv_8 = Dropout(0.1)(conv_8)
    conv_8 = Convolution2D(8, kernel, padding="same")(conv_8)
    conv_8 = BatchNormalization()(conv_8)
    conv_8 = LeakyReLU(alpha=alpha)(conv_8)

    unpool_4 = MaxUnpooling2D(pool_size)([conv_8, mask_1])
    concat_4 = Concatenate()([unpool_4, conv_1])

    conv_9 = Convolution2D(8, kernel, padding="same")(concat_4)
    conv_9 = BatchNormalization()(conv_9)
    conv_9 = LeakyReLU(alpha=alpha)(conv_9)
    conv_9 = Convolution2D(8, kernel, padding="same")(conv_9)
    conv_9 = BatchNormalization()(conv_9)
    conv_9 = LeakyReLU(alpha=alpha)(conv_9)

    conv_10 = Convolution2D(n_labels, (1, 1), padding="valid")(conv_9)
    conv_10 = BatchNormalization()(conv_10)
    conv_10 = Reshape(
            (input_shape[0] * input_shape[1], n_labels),
            input_shape=(input_shape[0], input_shape[1], n_labels))(conv_10)

    outputs = Activation(output_mode)(conv_10)
    print("Build decoder done..")

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy',
                  metrics=['accuracy', fmeasure, precision, recall])
    return model

