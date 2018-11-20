# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

#!/usr/bin/env python2.7

"""Send PNG image to tensorflow_model_server loaded with inception model.
"""

from __future__ import print_function

# This is a placeholder for a Google-internal import.

import grpc
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

import cv2
import time


tf.app.flags.DEFINE_string('server', 'localhost:9000',
                           'PredictionService host:port')
tf.app.flags.DEFINE_string('image', '', 'path to image in PNG format')
FLAGS = tf.app.flags.FLAGS


def main(_):
  channel = grpc.insecure_channel(FLAGS.server)
  stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
  data = cv2.imread(FLAGS.image)
  print(data.shape)
  request = predict_pb2.PredictRequest()
  request.model_spec.name = 'faster_rcnn'
  request.inputs['inputs'].CopyFrom(
      tf.contrib.util.make_tensor_proto(data, shape=[1,480,800,3]))
  starttime = time.time()
  result = stub.Predict(request, 10.0)  # 10 secs timeout
  endtime = time.time()
  print("detecting time is {}".format(endtime-starttime))
  #get the prediction result
  model_detection_boxes = result.outputs['detection_boxes'].float_val
  model_detection_classes = result.outputs['detection_classes'].float_val
  model_detection_scores = result.outputs['detection_scores'].float_val
  #count the number of detection score is greater 0.9
  count = 0
  for score in model_detection_scores:
    if score < 0.9:
      break
    count = count+1
  #get each box information
  response_info = []
  class_info = {"1":"touched","2":"untouched"}
  for i in range(count):
    box_info = {}
    box_info["class"] = class_info[str(int(model_detection_classes[i]))]
    x = model_detection_boxes[i*4]*480
    y = model_detection_boxes[i*4+1]*800
    height = (model_detection_boxes[i*4+2] - model_detection_boxes[i*4])*480
    width = (model_detection_boxes[i*4+3] - model_detection_boxes[i*4+1])*800
    x_y_h_w = [x,y,height,width]
    box_info["location_info"] = x_y_h_w
    response_info.append(box_info)
  print("detect {} button in the image".format(count))
  print(response_info)
  
        


if __name__ == '__main__':
  tf.app.run()
