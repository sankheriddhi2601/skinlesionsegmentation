# model.py
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, UpSampling2D, Concatenate, \
    Input, add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def batchnorm_relu(inputs):
    x = BatchNormalization()(inputs)
    x = Activation("relu")(x)
    return x


def MultiResBlock(U, inp, alpha=1.67):
    W = alpha * U

    shortcut = inp

    conv3x3 = Conv2D(int(W * 0.167), kernel_size=(3, 3), padding='same')(inp)
    conv3x3 = BatchNormalization()(conv3x3)
    conv3x3 = Activation('relu')(conv3x3)

    conv5x5 = Conv2D(int(W * 0.333), kernel_size=(3, 3), padding='same')(conv3x3)
    conv5x5 = BatchNormalization()(conv5x5)
    conv5x5 = Activation('relu')(conv5x5)

    conv7x7 = Conv2D(int(W * 0.5), kernel_size=(3, 3), padding='same')(conv5x5)
    conv7x7 = BatchNormalization()(conv7x7)
    conv7x7 = Activation('relu')(conv7x7)

    out = Concatenate()([conv3x3, conv5x5, conv7x7])

    if shortcut.shape[-1] != out.shape[-1]:
        shortcut = Conv2D(out.shape[-1], kernel_size=(1, 1), padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)

    out = add([shortcut, out])
    out = Activation('relu')(out)
    out = BatchNormalization()(out)

    return out


def ResPath(filters, length, inp):
    shortcut = inp
    out = Conv2D(filters, kernel_size=(3, 3), padding='same')(inp)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)

    for i in range(length - 1):
        shortcut = out
        out = Conv2D(filters, kernel_size=(3, 3), padding='same')(out)
        out = BatchNormalization()(out)
        out = Activation('relu')(out)

        out = add([shortcut, out])
        out = BatchNormalization()(out)
        out = Activation('relu')(out)

    return out


def decoder_block(inputs, skip_features, num_filters):
    x = UpSampling2D((2, 2))(inputs)
    x = Concatenate()([x, skip_features])
    x = MultiResBlock(num_filters, x)
    return x


def build_multi_resunet(input_shape):
    inputs = Input(input_shape)

    """ Encoder 1 """
    mresblock1 = MultiResBlock(32, inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(mresblock1)
    mresblock1 = ResPath(32, 4, mresblock1)

    """ Encoder 2 """
    mresblock2 = MultiResBlock(64, pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(mresblock2)
    mresblock2 = ResPath(64, 3, mresblock2)

    """ Encoder 3 """
    mresblock3 = MultiResBlock(128, pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(mresblock3)
    mresblock3 = ResPath(128, 2, mresblock3)

    """ Bridge """
    mresblock4 = MultiResBlock(256, pool3)

    """ Decoder 1, 2, 3 """
    d1 = decoder_block(mresblock4, mresblock3, 128)
    d2 = decoder_block(d1, mresblock2, 64)
    d3 = decoder_block(d2, mresblock1, 32)

    """ Classifier """
    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d3)

    """ Model """
    model = Model(inputs, outputs)
    return model


if __name__ == "__main__":
    model = build_multi_resunet((256, 256, 3))
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

