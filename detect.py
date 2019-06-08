import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny, Custom
)
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs
import yolov3_tf2.dataset as dataset

import glob

from eval.lib.BoundingBox import BoundingBox
from eval.lib.BoundingBoxes import BoundingBoxes
from eval.lib.Evaluator import Evaluator
from eval.lib.utils import *


flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('image', './data/', 'path to input image')
flags.DEFINE_string('output', './output', 'path to output image')
flags.DEFINE_string('val_dataset', '', 'validation dataset path')

def main(_argv):
    if FLAGS.tiny:
       yolo = YoloV3Tiny(classes=49)
    else:
       yolo = YoloV3()
      
      #yolo = Custom(size=FLAGS.size)
    yolo.load_weights(FLAGS.weights)
    logging.info('weights loaded')

    files = glob.glob(FLAGS.image + '/*')
    
    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    val_dataset = dataset.load_tfrecord_test(FLAGS.val_dataset, FLAGS.classes)
    #val_dataset = val_dataset.make_one_shot_iterator()

    allBoundingBoxes = BoundingBoxes()
    
    for val in val_dataset:
        #img = tf.image.decode_image(val[0], channels=3)
        img = tf.expand_dims(val[0], 0)
        img = transform_images(img, FLAGS.size)
        #print(img)
        t1 = time.time()
        boxes, scores, classes, nums = yolo(img)
        t2 = time.time()
        logging.info('time: {}'.format(t2 - t1))

        logging.info('detections:')
        for i in range(nums[0]):
            logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                               np.array(scores[0][i]),
                                               np.array(boxes[0][i])))
            
        img = np.array(val[0])
        img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
        out = FLAGS.output + '/{}'.format(str(val[2]).encode("utf-8").decode("utf-8"))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(out, img)
        logging.info('output saved to: {}'.format(out))
    
        boxes, objectness, classes, nums = boxes[0], scores[0], classes[0], nums[0]
        wh = np.flip(img.shape[0:2])
        for i in range(nums):
            x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
            x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
            bb = BoundingBox(imageName=val[2], classId=class_names[int(classes[i])],
                             x=x1y1[0], y=x1y1[1], w=x2y2[0], h=x2y2[1],
                             typeCoordinates=CoordinatesType.Absolute,
                             bbType=BBType.Detected,
                             classConfidence=objectness[i], format=BBFormat.XYX2Y2)
            allBoundingBoxes.addBoundingBox(bb)
            
        for i in range(tf.shape(val[1])[0]):
            
            x1y1 = tuple((np.array(val[1][i][0:2]) * wh).astype(np.int32))
            x2y2 = tuple((np.array(val[1][i][2:4]) * wh).astype(np.int32))
            bb = BoundingBox(imageName=val[2], classId=class_names[int(val[1][i][4])],
                             x=x1y1[0], y=x1y1[1], w=x2y2[0], h=x2y2[1],
                             typeCoordinates=CoordinatesType.Absolute,
                             bbType=BBType.GroundTruth, format=BBFormat.XYX2Y2)
            allBoundingBoxes.addBoundingBox(bb)

    evaluator = Evaluator()
    metricsPerClass = evaluator.GetPascalVOCMetrics(allBoundingBoxes, IOUThreshold=0.1)
    print("Average precision values per class:\n")
    # Loop through classes to obtain their metrics
    mAP = 0
    n_classes = 0
    for mc in metricsPerClass:
        # Get metric values per each class
        c = mc['class']
        precision = mc['precision']
        recall = mc['recall']
        average_precision = mc['AP']
        ipre = mc['interpolated precision']
        irec = mc['interpolated recall']
        # Print AP per class
        print('%s: %f' % (c, average_precision))
        n_classes += 1
        mAP += average_precision
    print("mAP: {}".format(mAP/n_classes))
        
if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
