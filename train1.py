import os
import cv2
import time
import numpy as np
import tensorflow as tf
import scipy
from scipy import integrate

from tensorflow import keras
from keras.models import Model
import matplotlib.pyplot as plt
from keras import layers
from keras.optimizers import Adam
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
from keras.layers import Dense,Dropout,Flatten,BatchNormalization,Conv2D,MaxPool2D

#训练的类别：有口罩和无口罩
num_classes = 2
#输入网路图片的大小
img_h, img_w = 224, 224
#训练一次性加载数据集大小
batch_size = 4
#训练集和验证集图片位置
train_data_dir='New Masks Dataset/Train'
validation_data_dir='New Masks Dataset/Validation'

# 对图片数据进行增强
train_datagen = ImageDataGenerator(rescale=1.0 / 255)
val_datagen = ImageDataGenerator(rescale=1.0 / 255)

# 对图片进行缩放大小，设置加载图片数量，定义类别模式和随机打散
train_generate = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_h, img_w),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

val_generate = val_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_h, img_w),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

# 使用迁移学习的方式定义模型
model_inception_v3 = keras.applications.inception_v3.InceptionV3(weights='imagenet', include_top=False, input_shape=(img_h, img_w, 3))
x = model_inception_v3.output
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dense(1024, activation='relu')(x)
predications = keras.layers.Dense(2, activation='softmax')(x)
model_V3 = Model(inputs=model_inception_v3.input, outputs=predications)

# 冻结住所有的层
for layer in model_inception_v3.layers:
    layer.trainable = False

# 查看模型
model_V3.summary()

# 保存最优的模型
checkpoint = ModelCheckpoint(
    'checkMask.h5',
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    save_weights_only=False
)
# 当经过5代之后，验证集的损失值没有下降就提前终止训练
earlyStop = EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=10,
    verbose=1,
    restore_best_weights=True
)
# 当经过3代的训练之后验证集的损失值没有下降就学习衰减率
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.8,
    patience=10,
    verbose=1,
    min_delta=0.0001
)

callbacks = [checkpoint, earlyStop, reduce_lr]

# 模型编译
model_V3.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001),
                 metrics=['accuracy'])
# 获取训练集和验证集大小
n_train = train_generate.n
n_val = val_generate.n

# 迭代次数
epoches = 50

# 开始训练
history = model_V3.fit(
    train_generate,
    steps_per_epoch=n_train // batch_size,
    callbacks=callbacks,
    epochs=epoches,
    validation_data=val_generate,
    validation_steps=n_val // batch_size
)

#绘制图像
x=range(1,len(history.history['accuracy'])+1)
plt.plot(x,history.history['accuracy'])
plt.plot(x,history.history['val_accuracy'])
plt.title('mode accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.xticks(x)
plt.legend(['Train','Val'],loc='upper left')
plt.savefig(fname='inception_v3', dpi=300)
plt.show()