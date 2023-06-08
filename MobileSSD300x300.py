"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2023-01-01 17:40
"""
import os
import cv2
import cvzone
import numpy as np
from PIL import Image
from keras.models import load_model
from keras.utils import img_to_array, load_img

# 设置图片的宽高
img_w, img_h = 300, 300
# 得到图片的高宽比
WHRatio = img_w / float(img_h)
# 设置图片的缩放因子
ScaleFactor = 0.007843
# 设置平均值
meanVal = 127.5
# 设置置信度阈值
threadVal = 0.2

# 预测类别
# mobileNetSSD可以检测类别数21=20+1（背景）
classNames = ['background',
              'aeroplane', 'bicycle', 'bird', 'boat',
              'bottle', 'bus', 'car', 'cat', 'chair',
              'cow', 'diningtable', 'dog', 'horse',
              'motorbike', 'person', 'pottedplant',
              'sheep', 'sofa', 'train', 'tvmonitor']
# 加载文件
net = cv2.dnn.readNetFromCaffe(prototxt='cv2/MobileNetSSD_300x300.prototxt',
                               caffeModel='cv2/mobilenet_iter_73000.caffemodel')


# 对图片进行处理和设置网络的输入同时进行前向传播
def processImage(imgSize, mask):
    # 对图片进行预处理
    blob = cv2.dnn.blobFromImage(image=imgSize, scalefactor=ScaleFactor,
                                 size=(img_w, img_h), mean=meanVal)
    # 设置网络的输入并进行前向传播
    net.setInput(blob)
    detections = net.forward()
    # 对图像进行按比例裁剪
    height, width, channel = np.shape(imgSize)
    if width / float(height) > WHRatio:  # 说明高度比较小
        # 裁剪多余的宽度
        cropSize = (int(height * WHRatio), height)
    else:  # 说明宽度比较小
        # 裁剪多余的高度
        cropSize = (width, int(width / WHRatio))
    x1 = int((width - cropSize[0]) / 2)
    x2 = int(x1 + cropSize[0])
    y1 = int((height - cropSize[1]) / 2)
    y2 = int(y1 + cropSize[1])
    imgSize = imgSize[y1:y2, x1:x2]
    height, width, channel = np.shape(imgSize)

    # 遍历检测的目标
    # print('detection.shape: {}'.format(detections.shape))
    # print('detection: {}'.format(detections))
    for i in range(detections.shape[2]):
        # 保留两位小数
        confidence = round(detections[0, 0, i, 2] * 100, 2)
        # 这里只检测人这个目标
        if confidence > threadVal:
            class_id = int(detections[0, 0, i, 1])
            if class_id == 15:
                xLeftBottom = int(detections[0, 0, i, 3] * width)
                yLeftBottom = int(detections[0, 0, i, 4] * height)
                xRightTop = int(detections[0, 0, i, 5] * width)
                yRightTop = int(detections[0, 0, i, 6] * height)

                # 绘制是否佩戴口罩标志的位置
                mask_x = int(xLeftBottom + (xRightTop - xLeftBottom) / 2)
                mask_y = int(yLeftBottom - 20)
                if mask == 'mask':
                    cv2.rectangle(img=imgSize, pt1=(xLeftBottom, yLeftBottom),
                                  pt2=(xRightTop, yRightTop), color=(0, 255, 0), thickness=2)
                    label = classNames[class_id] + ": " + str(confidence)
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                    cvzone.putTextRect(img=imgSize, text=label, pos=(xLeftBottom + 9, yLeftBottom - 12),
                                       scale=1, thickness=1, colorR=(0, 255, 0))
                    cv2.putText(imgSize, str(mask), (mask_x, mask_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0))

                elif mask == 'non_mask':
                    cv2.rectangle(img=imgSize, pt1=(xLeftBottom, yLeftBottom),
                                  pt2=(xRightTop, yRightTop), color=(0, 0, 255), thickness=2)
                    label = classNames[class_id] + ": " + str(confidence)
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                    cvzone.putTextRect(img=imgSize, text=label, pos=(xLeftBottom + 9, yLeftBottom - 12),
                                       scale=1, thickness=1, colorR=(0, 0, 255))
                    # cv2.rectangle(imgSize, (xLeftBottom, yLeftBottom - labelSize[1]),
                    #               (xLeftBottom + labelSize[0], yLeftBottom + baseLine),
                    #               (255, 255, 255), cv2.FILLED)
                    cv2.putText(imgSize, str(mask), (mask_x, mask_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))
    return imgSize


# 对单张图片进行检测
def SignalDetect(img_path='images//8.png'):
    imgSize = cv2.imread(img_path)
    imgSize = processImage(imgSize=imgSize, mask='mask')
    cv2.imshow('imgSize', imgSize)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 输入网络的图片大小
target_size = (224, 224)


# 对图像进行预处理
def preprocess_image(image, target_size):
    """
    :param img_path: 图片路径
    :param target_size: 图片大小
    :return:
    """
    image = Image.fromarray(image, mode='RGB')
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)
    image = img_to_array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image


# 实时检测
def detectTime(modelInV3, classes):
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        # 预测图像类别
        img = preprocess_image(frame, target_size=target_size)
        predications = modelInV3.predict(img)[0].tolist()
        print(predications)
        print(np.argmax(predications))
        PreResult = int(np.argmax(predications))
        print('预测类别: {}'.format(classes[PreResult]))

        frame = cv2.resize(src=frame, dsize=(520, 520))
        frame = cv2.flip(src=frame, flipCode=2)
        frame = processImage(frame, mask=classes[PreResult])
        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    SignalDetect()
