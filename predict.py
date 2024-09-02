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

import codecs
import os
import time
import sys
sys.path.append('PaddleDetection')
import json
import yaml
from functools import reduce
import multiprocessing

from PIL import Image
import cv2
import numpy as np
import paddle
# import paddleseg.transforms as T
from paddle.inference import Config
from paddle.inference import create_predictor
from multiprocessing.dummy import Pool as ThreadPool
from functools import partial
from deploy.python.preprocess import preprocess,Resize, NormalizeImage, Permute, PadStride
from deploy.python.utils import argsparser, Timer, get_current_memory_mb

class PredictConfig():
    def __init__(self, model_dir):
        # parsing Yaml config for Preprocess
        deploy_file = os.path.join(model_dir, 'infer_cfg.yml')
        with open(deploy_file) as f:
            yml_conf = yaml.safe_load(f)
        self.arch = yml_conf['arch']
        self.preprocess_infos = yml_conf['Preprocess']
        self.min_subgraph_size = yml_conf['min_subgraph_size']
        self.labels = yml_conf['label_list']
    #     self.print_config()

    # def print_config(self):
    #     print('%s: %s' % ('Model Arch', self.arch))
    #     for op_info in self.preprocess_infos:
    #         print('--%s: %s' % ('transform op', op_info['type']))


def get_test_images(infer_file):
    with open(infer_file, 'r') as f:
        dirs = f.readlines()
    images = []
    for dir in dirs:
        images.append(eval(repr(dir.replace('\n',''))).replace('\\', '/'))
    assert len(images) > 0, "no image found in {}".format(infer_file)
    return images

def load_predictor(model_dir):
    config = Config(
        os.path.join(model_dir, 'model.pdmodel'),
        os.path.join(model_dir, 'model.pdiparams'))
    # initial GPU memory(M), device ID
    config.enable_use_gpu(10000, 0)
    # optimize graph and fuse op
    config.switch_ir_optim(True)
    # disable print log when predict
    config.disable_glog_info()
    # enable shared memory
    config.enable_memory_optim()
    # disable feed, fetch OP, needed by zero_copy_run
    config.switch_use_feed_fetch_ops(False)
    predictor = create_predictor(config)
    return predictor, config



def create_inputs(imgs, im_info):
    inputs = {}

    im_shape = []
    scale_factor = []
    if len(imgs) == 1:
        inputs['image'] = np.array((imgs[0], )).astype('float32')
        inputs['im_shape'] = np.array(
            (im_info[0]['im_shape'], )).astype('float32')
        inputs['scale_factor'] = np.array(
            (im_info[0]['scale_factor'], )).astype('float32')
        return inputs

    for e in im_info:
        im_shape.append(np.array((e['im_shape'], )).astype('float32'))
        scale_factor.append(np.array((e['scale_factor'], )).astype('float32'))

    inputs['im_shape'] = np.concatenate(im_shape, axis=0)
    inputs['scale_factor'] = np.concatenate(scale_factor, axis=0)

    imgs_shape = [[e.shape[1], e.shape[2]] for e in imgs]
    max_shape_h = max([e[0] for e in imgs_shape])
    max_shape_w = max([e[1] for e in imgs_shape])
    padding_imgs = []
    for img in imgs:
        im_c, im_h, im_w = img.shape[:]
        padding_im = np.zeros(
            (im_c, max_shape_h, max_shape_w), dtype=np.float32)
        padding_im[:, :im_h, :im_w] = img
        padding_imgs.append(padding_im)
    inputs['image'] = np.stack(padding_imgs, axis=0)
    return inputs


class Detector(object):

    def __init__(self,
                 pred_config,
                 model_dir):
        self.pred_config = pred_config
        self.predictor, self.config = load_predictor(model_dir)
        self.preprocess_ops = self.get_ops()
    
    def get_ops(self):
        preprocess_ops = []
        for op_info in self.pred_config.preprocess_infos:
            new_op_info = op_info.copy()
            op_type = new_op_info.pop('type')
            preprocess_ops.append(eval(op_type)(**new_op_info))
        return preprocess_ops

    def predict(self, inputs):
        # preprocess
        input_names = self.predictor.get_input_names()
        for i in range(len(input_names)):
            input_tensor = self.predictor.get_input_handle(input_names[i])
            input_tensor.copy_from_cpu(inputs[input_names[i]])

        # model prediction
        self.predictor.run()
        output_names = self.predictor.get_output_names()
        boxes_tensor = self.predictor.get_output_handle(output_names[0])
        np_boxes = boxes_tensor.copy_to_cpu()
        boxes_num = self.predictor.get_output_handle(output_names[1])
        np_boxes_num = boxes_num.copy_to_cpu()

        # postprocess
        results = []
        if reduce(lambda x, y: x * y, np_boxes.shape) < 6:
            results = {'boxes': np.zeros([]), 'boxes_num': [0]}
        else:
            results = {'boxes': np_boxes, 'boxes_num': np_boxes_num}
        return results

# 将原preprocess的两个参数转为一个参数para
def my_preprocess(para):
    im_path, preprocess_ops = para
    im, im_info = preprocess(im_path, preprocess_ops)
    return im, im_info

def predict_image(detector, image_list, result_path,multiclass_thres):
    c_results = {"result": []}
    # 不同目标设定不同输出阈值
    num_worker = 4
    # processes这个参数可以不设置，如果不设置函数会跟根据计算机的实际情况来决定要运行多少个进程
    pool = ThreadPool(processes=num_worker)# 多线程处理输入图像，预处理速度快一些
    img_length = len(image_list)
    # 根据评估数据自行调整每次多线程处理的样本数量, len(image_list) >= img_iter_filter
    img_iter_filter = 10
    img_iter_range = list(range(img_length//img_iter_filter))
    for start_index in img_iter_range:
        if start_index == img_iter_range[-1]:
            im_paths = image_list[start_index*img_iter_filter:]
        else:
            im_paths = image_list[start_index*img_iter_filter:(start_index+1)*img_iter_filter]
        image_ids = [os.path.basename(im_p).split('.')[0] for im_p in im_paths]
        para = [[i,detector.preprocess_ops] for i in im_paths]
        imandinfos = pool.map(my_preprocess, para)
        # print('imandinfos',imandinfos)
        for idx, imandinfo in enumerate(imandinfos):
            # 检测模型图像预处理
            image_id = image_ids[idx]
            inputs = create_inputs([imandinfo[0]], [imandinfo[1]])

            # 检测模型预测结果
            det_results = detector.predict(inputs)
            # 检测模型写结果
            im_bboxes_num = det_results['boxes_num'][0]
            if im_bboxes_num > 0:
                bbox_results = det_results['boxes'][0:im_bboxes_num, 2:]
                id_results = det_results['boxes'][0:im_bboxes_num, 0]
                score_results = det_results['boxes'][0:im_bboxes_num, 1]
                for idx in range(im_bboxes_num):
                    if float(score_results[idx]) >= multiclass_thres[int(id_results[idx])]:
                        c_results["result"].append({"image_id": image_id,
                                                    "type": int(id_results[idx]) + 1,
                                                    "x": float(bbox_results[idx][0]),
                                                    "y": float(bbox_results[idx][1]),
                                                    "width": float(bbox_results[idx][2]) - float(bbox_results[idx][0]),
                                                    "height": float(bbox_results[idx][3]) - float(bbox_results[idx][1]),
                                                    "segmentation": []})

    # 写文件
    with open(result_path, 'w') as ft:
        json.dump(c_results, ft)

def main(infer_txt, result_path, det_model_path,threshold):
    pred_config = PredictConfig(det_model_path)
    detector = Detector(pred_config, det_model_path)

    # predict from image
    img_list = get_test_images(infer_txt)
    predict_image(detector, img_list, result_path,threshold)


if __name__ == '__main__':
    print('start…')
    start_time = time.time()
    det_model_path = "model/"

    paddle.enable_static()
    infer_txt = sys.argv[1]
    result_path = sys.argv[2]

    # threshold=[ 0.8,        #bomb
    #         0.8,        #bridge
    #         0.8,        #safety
    #         0.8,        #锥桶            0.75刚好检测到入镜头一半的锥桶0.85下降了0.003
    #         0.7,        #crosswalk
    #         0.8,        #danger
    #         0.7,        #evil
    #         0.8,        #block
    #         0.8,        #patient
    #         0.8,        #prop
    #         0.7,        #spy            0.2没变
    #         0.7,        #thief
    #         0.8]        #tumble
    threshold=[ 0.9,        #bomb
                0.9,        #bridge
                0.9,        #safety
                0.9,        #锥桶            0.75刚好检测到入镜头一半的锥桶0.85下降了0.003
                0.9,        #crosswalk
                0.9,        #danger
                0.9,        #evil
                0.9,        #block
                0.9,        #patient
                0.9,        #prop
                0.9,        #spy            0.2没变
                0.9,        #thief
                0.9]        #tumble

    main(infer_txt, result_path, det_model_path,threshold)
    print('total time:', time.time() - start_time)
