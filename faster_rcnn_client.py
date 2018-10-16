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

"""Send JPEG image to tensorflow_model_server loaded with inception model.
"""

from __future__ import print_function

# This is a placeholder for a Google-internal import.

import grpc
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

from PIL import Image
import cv2
import datetime


tf.app.flags.DEFINE_string('server', 'localhost:9000',
                           'PredictionService host:port')
tf.app.flags.DEFINE_string('image', '', 'path to image in JPEG format')
FLAGS = tf.app.flags.FLAGS


def main(_):
  channel = grpc.insecure_channel(FLAGS.server)
  stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
  #img = Image.open(FLAGS.image)
  # Send request
  #with open(FLAGS.image, 'rb') as f:
    # See prediction_service.proto for gRPC request/response details.
  data = cv2.imread(FLAGS.image)
  print(data.shape)
  request = predict_pb2.PredictRequest()
  request.model_spec.name = 'faster_rcnn'
    #request.model_spec.signature_name = 'predict_images'
  request.inputs['inputs'].CopyFrom(
      tf.contrib.util.make_tensor_proto(data, shape=[1,480,800,3]))
  starttime = datetime.datetime.now()
  result = stub.Predict(request, 10.0)  # 10 secs timeout
  endtime = datetime.datetime.now()
  print((endtime-starttime).seconds)
  print(result)


if __name__ == '__main__':
  tf.app.run()
