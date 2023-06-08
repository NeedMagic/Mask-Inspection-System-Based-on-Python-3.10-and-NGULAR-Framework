import os
from tensorflow import keras
from keras import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from keras_preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from keras.applications import InceptionV3
from keras.layers import Dense, Flatten, Dropout

# 定义数据集目录
train_dir = 'New Masks Dataset/Train'
test_dir = 'New Masks Dataset/Test'
# 定义图像大小
img_height, img_width = 224, 224

# 对图片数据进行增强
train_datagen = ImageDataGenerator(rescale=1.0 / 255)
val_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generate = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=4,
    class_mode='categorical',
    shuffle=True
)

val_generate = val_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=4,
    class_mode='categorical',
    shuffle=True
)
'''
# 定义数据增强器
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=30,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

# 加载训练集数据
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=4,
    class_mode="binary"
)

# 加载测试集数据
test_datagen = ImageDataGenerator(rescale=1.0/255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=16,
    class_mode="binary"
)
'''
'''
# 加载预训练模型
base_model = InceptionV3(
    weights="imagenet",
    include_top=False,
    input_shape=(img_width, img_height, 3)
)

# 添加自定义层
x = base_model.output
x = Flatten()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation="sigmoid")(x)
# 定义模型
model = Model(inputs=base_model.input, outputs=predictions)

# 冻结模型层
for layer in base_model.layers:
    layer.trainable = False
'''
# 使用迁移学习的方式定义模型
model_inception_v3 = InceptionV3(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
x = model_inception_v3.output
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dense(1024, activation='relu')(x)
predications = keras.layers.Dense(2, activation='softmax')(x)
model = Model(inputs=model_inception_v3.input, outputs=predications)

# 输出模型摘要
model.summary()

# 设置adam参数 lr：学习率，控制每次更新参数的步长。默认值是0.001，beta_1：一阶矩估计的指数衰减率。默认值是0.9,beta_2：二阶矩估计的指数衰减率。默认值是0.999。
adam = Adam(learning_rate=0.0001, beta_1=0.8, beta_2=0.9)
# 编译模型
model.compile(
    loss="binary_crossentropy",
    optimizer=adam,
    metrics=["accuracy"],
)

# 定义回调函数
checkpoint = ModelCheckpoint('model_V3.h5',
                             monitor='val_accuracy',
                             verbose=1, save_best_only=True,
                             mode='max')

early_stop = EarlyStopping(monitor='val_loss',
                           patience=3,
                           verbose=1)


callbacks = [checkpoint, early_stop]

# 训练模型
history = model.fit(
    train_generate,
    steps_per_epoch=len(train_generate),
    callbacks=callbacks,
    epochs=50,
    validation_data=val_generate,
    validation_steps=len(val_generate),
)

# 评估模型
loss, acc = model.evaluate(val_generate, steps=len(val_generate))
print(f"Loss: {loss:.4f}, Accuracy: {acc:.4f}")

#绘制图像
x = range(1,len(history.history['accuracy'])+1)
plt.plot(x,history.history['accuracy'])
plt.plot(x,history.history['val_accuracy'])
plt.title('mode accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.xticks(x)
plt.legend(['Train','Val'],loc='upper left')
plt.savefig(fname='inception_v3', dpi=300)
plt.show()
