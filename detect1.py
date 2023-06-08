from flask import Flask, jsonify, request
import os
import cv2
from keras.models import load_model
from keras.utils import img_to_array
from PIL import Image
import numpy as np
from flask_cors import CORS

from MobileSSD300x300 import processImage

app = Flask(__name__)
CORS(app)
# 训练模型的位置
modelPath = 'model_V3.h5'
# 输入网络的图片大小
target_size = (224, 224)
# 图片的预测类别
classes = {0: 'mask', 1: 'non_mask'}


# 对图像进行预处理
def preprocess_image(image, target_size):
    image = image.convert('RGB')
    image = image.resize(target_size)
    image = img_to_array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image


# 单张图片检测
def detectImage(img):
    modelIn_V3 = load_model(modelPath)
    image = Image.open(img)
    image_array = preprocess_image(image, target_size)
    predictions = modelIn_V3.predict(image_array)[0].tolist()
    PreResult = int(np.argmax(predictions))
    print('预测类别: {}'.format(classes[PreResult]))
    imgSize = cv2.imread(img)
    imgSize = cv2.resize(src=imgSize, dsize=(520, 520))
    imgSize = cv2.flip(src=imgSize, flipCode=2)
    imgSize = processImage(imgSize=imgSize, mask=classes[PreResult])
    cv2.imwrite('/tmp/result.jpg', imgSize)
    return classes[PreResult]


# 处理上传图片并返回预测结果
@app.route('/detect', methods=['POST'])
def detect():
    # 获取上传的图片数据
    file = request.files['file']
    # 将图片保存到指定位置
    file_path = 'tmp/upload1.jpg'
    file.save(file_path)
    # 对图片进行检测
    result = detectImage(file_path)
    print('结果:' + result)
    print('类型:' + str(type(result)))
    # 封装检测结果为 JSON 格式并返回
    return jsonify({'result': result})


if __name__ == '__main__':
    app.run(debug=True)
