import os
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.models import load_model
from MobileSSD300x300 import detectTime, SignalDetect, processImage
from keras.utils import img_to_array, load_img


# 训练模型的位置
modelPath = 'model_V3.h5'
# 输入网络的图片大小
target_size = (224, 224)
# 预测图片位置
# imgPath = 'New Masks Dataset/Test/Non Mask/real_01035.jpg'
imgPath = 'New Masks Dataset/Train/Non Mask/0.jpg'
# 图片的预测类别
classes = {0: 'mask', 1: 'non_mask'}


# 对图像进行预处理
def preprocess_image(img_path, target_size):
    """
    :param img_path: 图片路径
    :param target_size: 图片大小
    :return:
    """
    image = Image.open(img_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)
    image = img_to_array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image


# 单张图片检测
def detectImage(imgPath):
    modelIn_V3 = load_model(modelPath)
    img = preprocess_image(img_path=imgPath, target_size=target_size)
    predications = modelIn_V3.predict(img)[0].tolist()
    print(predications)
    print(np.argmax(predications))
    PreResult = int(np.argmax(predications))
    print('预测类别: {}'.format(classes[PreResult]))

    imgSize = cv2.imread(imgPath)
    imgSize = cv2.resize(src=imgSize, dsize=(520, 520))
    imgSize = cv2.flip(src=imgSize, flipCode=2)
    imgSize = processImage(imgSize=imgSize, mask=classes[PreResult])
    print(np.shape(imgSize))
    cv2.imshow('imgSize', imgSize)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 实时检测
def Timedetect():
    modelInV3 = load_model(modelPath)
    detectTime(modelInV3=modelInV3, classes=classes)


if __name__ == '__main__':
    # Timedetect()
    detectImage(imgPath=imgPath)