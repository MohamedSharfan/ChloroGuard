import tensorflow as tf
from tensorflow.keras import layers, Model

def inverted_residual_block(x, expansion_factor, output_filters, stride):
    input_filters = x.shape[-1]

    expanded = layers.Conv2D(input_filters * expansion_factor,(1,1),padding='same', use_bias=False)(expanded)
    expanded = layers.batchNormalization()(expanded)
    expanded = layers.ReLU(max_value=6)(expanded)

    depthwise = layers.depthwiseConv2D(3,3),strides=stride, padding="same", use_bias=False)(expanded)
    depthwise = layers.BatchNormalization()(depthwise)
    depthwise = layers.ReLU(max_value=6)(depthwise)

    projected = layers.Conv2D(output_filters, (1, 1), padding="same", use_bias=False)(depthwise)
    projected = layers.BatchNormalization()(projected)

    if stride == 1 and input_filters == output_filters:
        return layers.Add()([x, projected])
    else:
        return projected


def MobileNetV2(input_shape=(224, 224, 3), num_classes=10):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, (3, 3), strides=2, padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6)(x)

    bottlenecks = [
        (1, 16, 1, 1),
        (6, 24, 2, 2),
        (6, 32, 3, 2),
        (6, 64, 4, 2),
        (6, 96, 3, 1),
        (6, 160, 3, 2),
        (6, 320, 1, 1),
    ]

    for expansion, filters, repeats, stride in bottlenecks:
        x = inverted_residual_block(x, expansion, filters, stride)
        for _ in range(repeats - 1):
            x = inverted_residual_block(x, expansion, filters, 1)

    x = layers.Conv2D(1280, (1, 1), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6)(x)

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs)
    return model