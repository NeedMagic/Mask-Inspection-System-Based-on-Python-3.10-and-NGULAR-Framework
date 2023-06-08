import tensorflow as tf
import tensorflow as tf
from keras.layers import Flatten, Dense, concatenate
from keras.optimizers import Adam
from tensorflow import keras
from keras import layers, Input, Model
from tensorflow import keras
from keras import layers
from keras.applications import EfficientNetB0, ResNet50

'''
# 数据集路径
train_tfrecord_path = 'train.tfrecord'
val_tfrecord_path = 'val.tfrecord'

# 模型参数
batch_size = 32
epochs = 10
num_classes = 3
input_shape = (224, 224, 3)

# 构建模型
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# 编译模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# Parse the TFRecord datasets
def parse_tfrecord(tfrecord):
    feature_description = {
        'image/height': tf.io.FixedLenFeature([], tf.int64),
        'image/width': tf.io.FixedLenFeature([], tf.int64),
        'image/filename': tf.io.FixedLenFeature([], tf.string),
        'image/source_id': tf.io.FixedLenFeature([], tf.string),
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/format': tf.io.FixedLenFeature([], tf.string),
        'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
        'image/object/class/text': tf.io.VarLenFeature(tf.string),
        'image/object/class/label': tf.io.VarLenFeature(tf.int64),
    }
    example = tf.io.parse_single_example(tfrecord, feature_description)

    # decode image
    image = tf.io.decode_png(example['image/encoded'], channels=3)
    # resize image
    image = tf.image.resize(image, (224, 224))

    # get bbox coordinates
    xmins = tf.sparse.to_dense(example['image/object/bbox/xmin'])
    ymins = tf.sparse.to_dense(example['image/object/bbox/ymin'])
    xmaxs = tf.sparse.to_dense(example['image/object/bbox/xmax'])
    ymaxs = tf.sparse.to_dense(example['image/object/bbox/ymax'])
    bboxes = tf.stack([ymins, xmins, ymaxs, xmaxs], axis=-1)

    # get class labels
    class_labels = tf.sparse.to_dense(example['image/object/class/label'])

    return image, bboxes, class_labels


# 读取数据集
train_dataset = tf.data.TFRecordDataset(train_tfrecord_path)
train_dataset = train_dataset.map(parse_tfrecord)
# train_dataset = train_dataset.shuffle(buffer_size=get_dataset_size(train_tfrecord_path))
train_dataset = train_dataset.batch(batch_size)

val_dataset = tf.data.TFRecordDataset(val_tfrecord_path)
val_dataset = val_dataset.map(parse_tfrecord)
val_dataset = val_dataset.batch(batch_size)

# 训练模型
history = model.fit(train_dataset,
                    epochs=epochs,
                    validation_data=val_dataset)
# Save the model
model.save('model.h5')
'''


import tensorflow as tf
import numpy as np
import os

batch_size = 32
epochs = 10
num_classes = 3
input_shape = (224, 224, 3)
# Define the paths for the annotation and image files
ANNOTATION_DIR = 'annotations'
IMAGE_DIR = 'images'

# Define the output paths for the TFRecord files
TRAIN_TFRECORD_PATH = 'train.tfrecord'
VAL_TFRECORD_PATH = 'val.tfrecord'

# Define the batch size and number of epochs
BATCH_SIZE = 1
EPOCHS = 10

# Define the number of classes
NUM_CLASSES = 3

# Load the train and validation datasets
train_dataset = tf.data.TFRecordDataset([TRAIN_TFRECORD_PATH])
val_dataset = tf.data.TFRecordDataset([VAL_TFRECORD_PATH])

print(tf.data.experimental.cardinality(train_dataset))
print(tf.data.experimental.cardinality(val_dataset))


# Parse the TFRecord datasets
def parse_tfrecord(tfrecord):
    feature_description = {
        'image/height': tf.io.FixedLenFeature([], tf.int64),
        'image/width': tf.io.FixedLenFeature([], tf.int64),
        'image/filename': tf.io.FixedLenFeature([], tf.string),
        'image/source_id': tf.io.FixedLenFeature([], tf.string),
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/format': tf.io.FixedLenFeature([], tf.string),
        'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
        'image/object/class/text': tf.io.VarLenFeature(tf.string),
        'image/object/class/label': tf.io.VarLenFeature(tf.int64),
    }
    example = tf.io.parse_single_example(tfrecord, feature_description)

    # decode image
    image = tf.io.decode_png(example['image/encoded'], channels=3)
    # resize image
    image = tf.image.resize(image, (224, 224))

    # get bbox coordinates
    xmins = tf.sparse.to_dense(example['image/object/bbox/xmin'])
    ymins = tf.sparse.to_dense(example['image/object/bbox/ymin'])
    xmaxs = tf.sparse.to_dense(example['image/object/bbox/xmax'])
    ymaxs = tf.sparse.to_dense(example['image/object/bbox/ymax'])
    bboxes = tf.stack([ymins, xmins, ymaxs, xmaxs], axis=-1)

    # get class labels
    class_labels = tf.sparse.to_dense(example['image/object/class/label'])

    return image, bboxes, class_labels

sample = next(iter(train_dataset))
image, bboxes, class_label = parse_tfrecord(sample)

print(image.shape)
print(bboxes.shape)
print(class_label.shape)

sample = next(iter(val_dataset))
image, bboxes, class_labels = parse_tfrecord(sample)

print(image.shape)
print(bboxes.shape)
print(class_labels.shape)

print("Train dataset size:", len(list(train_dataset)))
print("Validation dataset size:", len(list(val_dataset)))

# Apply the parse function to the datasets
# train_dataset = train_dataset.map(parse_tfrecord)
# val_dataset = val_dataset.map(parse_tfrecord)

# Apply the parse function to the datasets
train_dataset = train_dataset.map(parse_tfrecord)
train_dataset = train_dataset.shuffle(buffer_size=1000)
train_dataset = train_dataset.batch(batch_size)

val_dataset = val_dataset.map(parse_tfrecord)
val_dataset = val_dataset.batch(batch_size)
print(train_dataset)
print(val_dataset)

def build_model(input_shape, num_classes):
    # input layers
    image_input = tf.keras.Input(shape=input_shape, name='image_input')
    bbox_input = tf.keras.Input(shape=(None, 4), name='bbox_input')

    # image preprocessing layers
    x = tf.keras.layers.experimental.preprocessing.Rescaling(scale=1./255)(image_input)
    x = tf.keras.layers.experimental.preprocessing.RandomFlip(mode='horizontal')(x)
    x = tf.keras.layers.experimental.preprocessing.RandomRotation(factor=0.02)(x)
    x = tf.keras.layers.experimental.preprocessing.RandomZoom(height_factor=(-0.2, 0.2), width_factor=(-0.2, 0.2))(x)
    x = tf.keras.layers.experimental.preprocessing.CenterCrop(height=input_shape[0], width=input_shape[1])(x)
    x = tf.keras.layers.experimental.preprocessing.Resizing(224, 224, interpolation='bilinear')(x)

    # feature extraction using ResNet50
    resnet50 = tf.keras.applications.ResNet50(include_top=False, input_shape=(224, 224, 3), weights='imagenet')
    resnet50.trainable = False
    features = resnet50(x)

    # object detection layers
    x = tf.keras.layers.GlobalAveragePooling2D()(features)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    # final model
    model = tf.keras.Model(inputs=[bbox_input, image_input], outputs=x, name='mask_detection_model')

    return model

model = build_model(input_shape=(224, 224, 3), num_classes=3)

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

model.fit(train_dataset, epochs=10, validation_data=val_dataset)


'''
# Define the ResNet50 model architecture
base_model = tf.keras.applications.ResNet50(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze the base model
base_model.trainable = False

# Add a custom classifier on top of the base model
inputs = tf.keras.Input(shape=(224, 224, 3))
x = tf.keras.applications.resnet50.preprocess_input(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
outputs = tf.keras.layers.Dense(3, activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)

# Define the loss function, optimizer, and evaluation metrics
loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()
metrics = ['accuracy']

# Compile the model
model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

# 打印模型的摘要信息
model.summary()

# Define the callbacks
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    'model.h5',
    save_best_only=True,
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max',
    verbose=1
)

early_stopping_cb = tf.keras.callbacks.EarlyStopping(
    patience=3,
    monitor='val_accuracy',
    mode='max',
    verbose=1
)

# Determine the number of steps per epoch
train_steps_per_epoch = len(TRAIN_TFRECORD_PATH) // BATCH_SIZE
val_steps_per_epoch = len(VAL_TFRECORD_PATH) // BATCH_SIZE

# Train the model
history = model.fit(
    train_dataset.batch(BATCH_SIZE),
    validation_data=val_dataset.batch(BATCH_SIZE),
    epochs=EPOCHS,
    steps_per_epoch=train_steps_per_epoch,
    validation_steps=val_steps_per_epoch,
    callbacks=[checkpoint_cb, early_stopping_cb],
    verbose=1
)
'''

'''
# input layer
inputs = tf.keras.layers.Input(shape=(224, 224, 3), name='input_image')

# feature extraction backbone
backbone = tf.keras.applications.MobileNetV2(
    include_top=False, input_shape=(224, 224, 3), weights='imagenet')
x = backbone(inputs)

# object detection head
x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
x = tf.keras.layers.Flatten(name='flatten')(x)
x = tf.keras.layers.Dense(256, activation='relu', name='fc1')(x)
x = tf.keras.layers.Dense(64, activation='relu', name='fc2')(x)

# bounding box prediction head
bbox_pred = tf.keras.layers.Dense(4, activation='sigmoid', name='bbox')(x)

# class prediction head
class_pred = tf.keras.layers.Dense(3, activation='softmax', name='class')(x)

# define model
model = tf.keras.models.Model(inputs=inputs, outputs=[bbox_pred, class_pred])

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss={'bbox': 'mean_squared_error', 'class': 'categorical_crossentropy'})

'''
'''
# 1. 定义模型输入
input_shape = (224, 224, 3)
input_tensor = Input(shape=input_shape, name='input_tensor')

# 2. 定义特征提取层
base_model = ResNet50(input_tensor=input_tensor, include_top=False, weights='imagenet')
x = base_model.output

# 3. 定义边界框预测层

bbox_output = Dense(4, activation='sigmoid', name='bbox_output')(x)

# 4. 定义类别预测层

class_output = Dense(3, activation='softmax', name='class_output')(x)

# 5. 将边界框和类别预测结果合并
output_tensor = tf.concat([bbox_output, class_output], axis=-1, name='tf.concat')

# 6. 定义模型
model = tf.keras.models.Model(inputs=input_tensor, outputs=output_tensor)

# 7. 编译模型
losses = {'tf.concat': 'mse'}
loss_weights = {'tf.concat': 1.}
optimizer = tf.keras.optimizers.Adam()

model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights, metrics=['accuracy'])

# model.summary()

# train model
model.fit(train_dataset,
          epochs=EPOCHS,
          validation_data=val_dataset)
'''
# Save the model
model.save('model.h5')
