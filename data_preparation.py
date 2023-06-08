import os
import numpy as np

import cv2
import tensorflow as tf
import xml.etree.ElementTree as ET
from PIL import Image
import io


# Define the paths for the annotation and image files
ANNOTATION_DIR = 'annotations'
IMAGE_DIR = 'images'
image_size = (224, 224)

# Define the output paths for the TFRecord files
TRAIN_OUTPUT_PATH = 'train.tfrecord'
VAL_OUTPUT_PATH = 'val.tfrecord'

# Define the classes
CLASSES = ['with_mask', 'without_mask', 'mask_weared_incorrect']

# Define the split ratio for train and val datasets
SPLIT_RATIO = 0.8


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

    # get bbox coordinates
    xmins = tf.sparse.to_dense(example['image/object/bbox/xmin'])
    ymins = tf.sparse.to_dense(example['image/object/bbox/ymin'])
    xmaxs = tf.sparse.to_dense(example['image/object/bbox/xmax'])
    ymaxs = tf.sparse.to_dense(example['image/object/bbox/ymax'])
    bboxes = tf.stack([ymins, xmins, ymaxs, xmaxs], axis=-1)

    # get class labels
    class_labels = tf.sparse.to_dense(example['image/object/class/label'])

    return image, bboxes, class_labels



def create_tf_example(filename, annotation, image_size):
    # Load the image
    image_path = os.path.join(IMAGE_DIR, filename)
    with tf.io.gfile.GFile(image_path, 'rb') as f:
        image = Image.open(io.BytesIO(f.read()))
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)  # Convert PIL Image to OpenCV image
        image = cv2.resize(image, image_size)  # Resize image
        width, height = image.shape[1], image.shape[0]

    # Read the annotation
    xml_path = os.path.join(ANNOTATION_DIR, annotation)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Create the example proto
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename.encode('utf-8')])),
        'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename.encode('utf-8')])),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.encode_png(image).numpy()])),
        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=[b'png'])),
    }))

    # Add the bounding boxes to the example proto
    xmin_list = []
    ymin_list = []
    xmax_list = []
    ymax_list = []
    class_id_list = []
    class_text_list = []

    for obj in root.findall('object'):
        class_text = obj.find('name').text
        class_id = CLASSES.index(class_text)
        bbox = obj.find('bndbox')
        xmin_list.append(float(bbox.find('xmin').text) / width)
        ymin_list.append(float(bbox.find('ymin').text) / height)
        xmax_list.append(float(bbox.find('xmax').text) / width)
        ymax_list.append(float(bbox.find('ymax').text) / height)
        class_id_list.append(class_id)
        class_text_list.append(class_text.encode('utf-8'))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename.encode('utf-8')])),
        'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename.encode('utf-8')])),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.encode_png(image).numpy()])),
        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=[b'png'])),
        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmin_list)),
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymin_list)),
        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmax_list)),
        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymax_list)),
        'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=class_text_list)),
        'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=class_id_list)),
    }))

    return tf_example


def main():
    # Get the list of annotations and images
    annotations = sorted(os.listdir(ANNOTATION_DIR))
    images = sorted(os.listdir(IMAGE_DIR))

    # Split the dataset into train and val sets
    num_train = int(len(annotations) * SPLIT_RATIO)
    train_annotations = annotations[:num_train]
    val_annotations = annotations[num_train:]
    train_images = [os.path.splitext(x)[0] + '.png' for x in train_annotations]
    val_images = [os.path.splitext(x)[0] + '.png' for x in val_annotations]

    # Write the train and val TFRecord files
    with tf.io.TFRecordWriter(TRAIN_OUTPUT_PATH) as train_writer:
        for i, (image, annotation) in enumerate(zip(train_images, train_annotations)):
            tf_example = create_tf_example(image, annotation, image_size)
            train_writer.write(tf_example.SerializeToString())
            print('Processed train example %d of %d' % (i + 1, len(train_images)))

    with tf.io.TFRecordWriter(VAL_OUTPUT_PATH) as val_writer:
        for i, (image, annotation) in enumerate(zip(val_images, val_annotations)):
            tf_example = create_tf_example(image, annotation, image_size)
            val_writer.write(tf_example.SerializeToString())
            print('Processed val example %d of %d' % (i + 1, len(val_images)))




if __name__ == '__main__':
    main()