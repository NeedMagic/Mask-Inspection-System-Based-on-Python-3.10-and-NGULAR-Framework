import tensorflow as tf


def create_mask_detection_model(num_classes):
    # 构建基础模型，使用MobileNetV2作为基础模型
    base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3),
                                                   include_top=False,
                                                   weights='imagenet')

    # 冻结基础模型的所有层
    base_model.trainable = False

    # 在基础模型的顶部添加自定义分类层
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.layers.experimental.preprocessing.Rescaling(scale=1. / 255)(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes)(x)

    # 构建完整模型
    model = tf.keras.Model(inputs, outputs)

    return model
