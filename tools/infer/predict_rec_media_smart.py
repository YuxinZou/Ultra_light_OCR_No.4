# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import cv2
import numpy as np
import math
import time
import traceback
import paddle

import tools.infer.utility as utility
from ppocr.postprocess import build_post_process
from ppocr.utils.logging import get_logger
from ppocr.utils.utility import get_image_file_list
from ppocr.data import create_operators, transform
from tools.program import load_config

logger = get_logger()


class TextRecognizer(object):
    def __init__(self, args):
        self.config = load_config(args.rec_config)
        self.rec_batch_num = args.rec_batch_num
        self.rec_algorithm = args.rec_algorithm
        self.rec_batch_num = 1  # it should be set to 1 for inference
        self.predictor, self.input_tensor, self.output_tensors = \
            utility.create_predictor(args, 'rec', logger)

        self.get_paras()

    def get_paras(self):
        global_config = self.config['Global']
        self.postprocess_op = build_post_process(self.config['PostProcess'],
                                                 global_config)

        transforms = []
        for op in self.config['Test']['transforms']:
            transforms.append(op)
        global_config['infer_mode'] = True
        self.ops = create_operators(transforms, global_config)

    def __call__(self, img):
        with open(img, 'rb') as f:
            img = f.read()
            data = {'image': img, 'epoch': 0, 'mode': 'test'}
        batch = transform(data, self.ops)
        images = np.expand_dims(batch[0], axis=0)

        elapse = 0
        starttime = time.time()
        self.input_tensor.copy_from_cpu(images)
        self.predictor.run()

        outputs = []
        for output_tensor in self.output_tensors:
            output = output_tensor.copy_to_cpu()
            outputs.append(output)
        preds = outputs[0]
        self.predictor.try_shrink_memory()
        rec_result = self.postprocess_op(preds)
        elapse += time.time() - starttime
        return rec_result[0], elapse


def main(args):
    image_file_list = get_image_file_list(args.image_dir)
    text_recognizer = TextRecognizer(args)
    total_run_time = 0.0
    total_images_num = 0
    os.makedirs(os.path.dirname(args.save_res_path), exist_ok=True)
    with open(args.save_res_path, "w") as fout:
        for idx, image_file in enumerate(image_file_list):
            rec_res, predict_time = text_recognizer(image_file)
            total_run_time += predict_time
            logger.info("Predicts of {}:{}".format(image_file, rec_res))
            total_images_num += 1
            fout.write(image_file + "\t" + rec_res[0] + "\t" + str(
                rec_res[1]) + "\n")
    logger.info("Total predict time for {} images, cost: {:.3f}".format(
        total_images_num, total_run_time))


if __name__ == "__main__":
    main(utility.parse_args())
