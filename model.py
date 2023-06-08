import tensorflow as tf
from keras import layers
from keras import Model

def create_model(num_classes):
    input_layer = layers.Input(shape=(None, None, 3), name='image')
    backbone = tf.keras.applications.EfficientNetB0(include_top=False, input_tensor=input_layer)

    # Add classification head
    x = layers.GlobalAveragePooling2D(name='avg_pool')(backbone.output)
    x = layers.Dropout(0.2, name='top_dropout')(x)
    output_layer = layers.Dense(num_classes, activation='softmax', name='probs')(x)

    # Compile the model
    model = Model(inputs=input_layer, outputs=output_layer, name='mask_detection')
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
