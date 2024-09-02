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

#根据paddledetection/deploy/python/infer.py改来的  inference
#有一部分类别的处理采用的是默认值需要参赛选手自己设计算法修改
#根据data.txt文件预测 结果保存在result.jason

#你训练好了一个模型，在训练数据集中表现良好，但是我们的期望是它可以对以前没看过的图片进行识别。
#你重新拍一张图片扔进网络让网络做判断，这种图片就叫做现场数据（livedata），
#如果现场数据的区分准确率非常高，那么证明你的网络训练的是非常好的。
#我们把训练好的模型拿出来遛一遛的过程，称为推理（Inference）。




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

#使用paddleinference来进行模型推理
from paddle.inference import Config
from paddle.inference import create_predictor


from deploy.python.preprocess import preprocess, Resize, NormalizeImage, Permute, PadStride
from deploy.python.utils import argsparser, Timer, get_current_memory_mb


class PredictConfig():
    """set config of preprocess, postprocess and visualize
    Args:
        model_dir (str): root path of model.yml
    """

    def __init__(self, model_dir):
        # parsing Yaml config for Preprocess
        #模型 链接地址
        deploy_file = os.path.join(model_dir, 'infer_cfg.yml')
        #执行一个有enter和exit的方法
        #yml.safe是用作从infer_cfg取出代码的
        with open(deploy_file) as f:
            yml_conf = yaml.safe_load(f)
        self.arch = yml_conf['arch']
        self.preprocess_infos = yml_conf['Preprocess']
        self.min_subgraph_size = yml_conf['min_subgraph_size']
        self.labels = yml_conf['label_list']
        self.mask = False
        self.use_dynamic_shape = yml_conf['use_dynamic_shape']
        #可选选项？
        if 'mask' in yml_conf:
            self.mask = yml_conf['mask']
        self.tracker = None
        if 'tracker' in yml_conf:
            self.tracker = yml_conf['tracker']
        if 'NMS' in yml_conf:
            self.nms = yml_conf['NMS']
        if 'fpn_stride' in yml_conf:
            self.fpn_stride = yml_conf['fpn_stride']
        self.print_config()

    def print_config(self):
        print('%s: %s' % ('Model Arch', self.arch))
        for op_info in self.preprocess_infos:
            print('--%s: %s' % ('transform op', op_info['type']))


#读入测试数据集
def get_test_images(infer_file):
    with open(infer_file, 'r') as f:
        dirs = f.readlines()
    images = []
    for dir in dirs:
        images.append(eval(repr(dir.replace('\n',''))).replace('\\', '/'))
    assert len(images) > 0, "no image found in {}".format(infer_file)
    return images


#模型地址
def load_predictor(model_dir):
    config = Config(
        os.path.join(model_dir, 'model.pdmodel'),
        os.path.join(model_dir, 'model.pdiparams'))
    # initial GPU memory(M), device ID
    config.enable_use_gpu(2000, 0)
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
    for e in im_info:
        im_shape.append(np.array((e['im_shape'], )).astype('float32'))
        scale_factor.append(np.array((e['scale_factor'], )).astype('float32'))

    origin_scale_factor = np.concatenate(scale_factor, axis=0)

    imgs_shape = [[e.shape[1], e.shape[2]] for e in imgs]
    max_shape_h = max([e[0] for e in imgs_shape])
    max_shape_w = max([e[1] for e in imgs_shape])
    padding_imgs = []
    padding_imgs_shape = []
    padding_imgs_scale = []
    for img in imgs:
        im_c, im_h, im_w = img.shape[:]
        padding_im = np.zeros(
            (im_c, max_shape_h, max_shape_w), dtype=np.float32)
        padding_im[:, :im_h, :im_w] = np.array(img, dtype=np.float32)
        padding_imgs.append(padding_im)
        padding_imgs_shape.append(
            np.array([max_shape_h, max_shape_w]).astype('float32'))
        rescale = [float(max_shape_h) / float(im_h), float(max_shape_w) / float(im_w)]
        padding_imgs_scale.append(np.array(rescale).astype('float32'))
    inputs['image'] = np.stack(padding_imgs, axis=0)
    inputs['im_shape'] = np.stack(padding_imgs_shape, axis=0)
    inputs['scale_factor'] = origin_scale_factor
    return inputs

#
class Detector(object):

    def __init__(self,
                 pred_config,
                 model_dir,
                 device='CPU',
                 run_mode='paddle',
                 batch_size=1,
                 trt_min_shape=1,
                 trt_max_shape=1280,
                 trt_opt_shape=640,
                 trt_calib_mode=False,
                 cpu_threads=1,
                 enable_mkldnn=False):
        self.pred_config = pred_config
        self.predictor, self.config = load_predictor(model_dir)
        self.det_times = Timer()
        self.cpu_mem, self.gpu_mem, self.gpu_util = 0, 0, 0
        self.preprocess_ops = self.get_ops()

    def get_ops(self):
        preprocess_ops = []
        for op_info in self.pred_config.preprocess_infos:
            new_op_info = op_info.copy()
            op_type = new_op_info.pop('type')
            preprocess_ops.append(eval(op_type)(**new_op_info))
        return preprocess_ops

#               图像大小和图像名称
    def predict(self, inputs):
        # preprocess

        #将图片写入输入张量
        input_names = self.predictor.get_input_names()
        for i in range(len(input_names)):
            #输入张量
            input_tensor = self.predictor.get_input_handle(input_names[i])
            input_tensor.copy_from_cpu(inputs[input_names[i]])
#* 首先，它获取模型的所有输入张量的名称。
#* 然后，对于每一个输入张量，它获取一个句柄，并从CPU上复制相应的数据到该张量中。


        np_boxes, np_boxes_num = [], []


        # model_prediction 模型预测
        self.predictor.run()

        #清零存储
        np_boxes.clear()
        np_boxes_num.clear()

        #获取输出层的输出张量的名称列表
        output_names = self.predictor.get_output_names()


        #outputname 的长度 除二取整
        num_outs = int(len(output_names) / 2)


        for out_idx in range(num_outs):
            np_boxes.append(
                #获取张量句柄
                self.predictor.get_output_handle(output_names[out_idx])
                .copy_to_cpu())

            np_boxes_num.append(
                self.predictor.get_output_handle(output_names[
                    out_idx + num_outs]).copy_to_cpu())
#########################################################################################
#推测npboxes中的预测框不应该只取出第一个

#他上面哪个数组就是多添加了一个[]的

#        np_boxes = np.array(item for item in np_boxes)
#        np_boxes_num = np.array(item for item in np_boxes_num)

############################################################################
        #将np_boxes和num创建成np的array数组
        #                                np_boxes和np_boxes_num的第一个元素
        np_boxes, np_boxes_num = np.array(np_boxes[0]), np.array(np_boxes_num[0])
################################################################################


        #dict 字典  返回预测框   预测框的num？
        return dict(boxes=np_boxes, boxes_num=np_boxes_num)



#使用阈值预测类别；
#将其改为根据不同类别的不同阈值筛选
    #输出预测框    导出模型的类 测试机路径   存放结果的文件路径 阈值
def predict_image(detector, image_list, result_path, threshold):

    #存放预测结果的变量
    #            key      列表
    c_results = {"result": []}

    #遍历测试集文件
    for index in range(len(image_list)):
        #

        # 检测模型图像预处理
        input_im_lst = []
        input_im_info_lst = []

        #返回测试集中文件（0-len）
        im_path = image_list[index]
        #处理后图像，图像缩放因子和大小     图像地址    预处理设置
        im, im_info = preprocess(im_path, detector.preprocess_ops)
        #preprocess预处理图像，返回处理好的图像 和对应数据

        #将处理好的图像添加到文件列表
        input_im_lst.append(im)
        #添加大小数据到info列表
        input_im_info_lst.append(im_info)
        #大概是创建了input类体；包括图像和对用
        inputs = create_inputs(input_im_lst, input_im_info_lst)

        #读取图像id（图像在路径中的名称）
        image_id = os.path.basename(im_path).split('.')[0]

#得到了input（图像大小和图像本身）    imageid（图像名称）
        # 检测模型预测结果
        #返回一个字典 key：np_boxes,value:boxes_num
        det_results = detector.predict(inputs)

        # 检测模型写结果
###############################################################################
        #你需要确保im_bboxes_num是一个整数
        #key:boxes_num的第一个值
        im_bboxes_num = det_results['boxes_num'][0]

        #如果 预测框数量？ 大于0
        if im_bboxes_num > 0:
#            保留的检测框的坐标：这通常是一个二维数组，其中每行代表一个检测框的坐标信息（如左上角和右下角的坐标）。
#            保留的检测框的得分或置信度：这是一个一维数组，与保留的检测框一一对应，表示每个检测框的得分或置信度。
#           可能还包含其他元数据，如检测框所属的类别ID等。

            #预测结果
            bbox_results = det_results['boxes'][0:im_bboxes_num, 2:]
            #类型名称
            id_results = det_results['boxes'][0:im_bboxes_num, 0]
            #置信度
            score_results = det_results['boxes'][0:im_bboxes_num, 1]


            for idx in range(im_bboxes_num):
                #如果置信度大于这个值

####################################################################################################################
                if float (score_results[idx]) > threshold[int (id_results[idx])]:
####################################################################################################################
#                if float(score_results[idx]) >= threshold:

                    c_results["result"].append({"image_id": image_id,
                                                #类型可能是从1开始的
                                                "type": int(id_results[idx]) + 1,
###############################################################################################################################
                                                #坐标框中心点？
                                                "x":(float(bbox_results[idx][0])+float(bbox_results[idx][0]))/2,
                                                "y":( float(bbox_results[idx][1])+float(bbox_results[idx][1]))/2,
###############################################################################################################################
                                                #宽高
                                                "width": float(bbox_results[idx][2]) - float(bbox_results[idx][0]),
                                                "height": float(bbox_results[idx][3]) - float(bbox_results[idx][1]),
                                                "segmentation": []})


    # 写文件
    with open(result_path, 'w') as ft:
        json.dump(c_results, ft)



#       验证集路径   结果文件路径      检测模型路径      过滤阈值
def main(infer_txt, result_path, det_model_path, threshold):
    pred_config = PredictConfig(det_model_path)
    detector = Detector(pred_config, det_model_path)    #detector类去定义模型

    # predict from image
    #读入测试集的文件
    img_list = get_test_images(infer_txt)
    #输出预测框    导出模型的类 测试机路径   存放结果的文件路径 阈值
    predict_image(detector, img_list, result_path, threshold)






#并不作为程序唯一对外接口，而是作为提醒，告诉代码阅读者这是进入口
if __name__ == '__main__':
    start_time = time.time()                            #返回当前的时间戳
    det_model_path = "model/picodet_m_320_coco_lcnet/"  #输入预测模型性的文件


#    threshold = 0.1                                     #对检测结果进行过

#    threshold=[ 0.269565,        #bomb          -0.2
#                0.25448,        #bridge         -0.2
#                0.22861,        #safety         -0.2
#               0.75,           #锥桶           0.75刚好检测到入镜头一半的锥桶
#                0.134783,       #crosswalk
#                0.64783,        #danger         -0.2
#                0.204348,       #evil
#                0.484783,       #block          -0.2
#                0.380435,       #patient
#                0.680435,       #prop           -0.2
#                0.05,           #spy
#                0.0804335,      #thief
#                0.278261]       #tumble

#全0.8   0.89478		
    threshold=[ 0.8,        #bomb
                0.8,        #bridge
                0.8,        #safety
                0.8,        #锥桶            0.75刚好检测到入镜头一半的锥桶0.85下降了0.003
                0.7,        #crosswalk
                0.8,        #danger
                0.7,        #evil
                0.8,        #block
                0.8,        #patient
                0.8,        #prop
                0.7,        #spy            0.2没变
                0.7,        #thief
                0.8]        #tumble
#调成0.75跌了0.003

    paddle.enable_static()
    #-c
    infer_txt = sys.argv[1]                             #输入图片列表  （验证集）

    result_path = sys.argv[2]                           #生成结果的文件


#进入main函数
    main(infer_txt, result_path, det_model_path, threshold)

    print('total time:', time.time() - start_time)


