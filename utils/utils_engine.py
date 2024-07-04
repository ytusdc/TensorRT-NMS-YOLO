#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Time    : 2024-07-04 16:21
Author  : sdc
"""

import tensorrt as trt
from cuda import cudart
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from utils import common
from utils.utils import prepareImage, vis, letterbox, multiclass_nms, xywh2xyxy
from utils.utils_yaml import get_id_cls_dict

class BaseEngine(object):
    def __init__(self, engine_path, cfg_path):
        self.mean = None
        self.std = None
        self.mode = "v8"  # 默认engine模型
        self.read_cfg(cfg_path)
        self.engine = self.loadEngine(engine_path)
        # get the read shape of model, in case user input it wrong
        self.imgsz = self.engine.get_tensor_shape(self.engine.get_tensor_name(0))[2:]
        self.context = self.engine.create_execution_context()
        # Setup I/O bindings
        self.inputs = []
        self.outputs = []
        self.allocations = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = self.engine.get_tensor_dtype(name)
            shape = self.engine.get_tensor_shape(name)
            is_input = False
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                is_input = True
            if is_input:
                self.batch_size = shape[0]
            size = np.dtype(trt.nptype(dtype)).itemsize
            for s in shape:
                size *= s
            allocation = common.cuda_call(cudart.cudaMalloc(size))
            binding = {
                'index': i,
                'name': name,
                'dtype': np.dtype(trt.nptype(dtype)),
                'shape': list(shape),
                'allocation': allocation,
                'size': size
            }
            self.allocations.append(allocation)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)

    def read_cfg(self, cfg_file_path):
        # self.id_cls_dict = get_id_cls_dict(cfg_file_path)
        # self.n_classes = len(self.id_cls_dict)
        self.class_names = get_id_cls_dict(cfg_file_path)
        self.n_classes = len(self.class_names)

    def loadEngine(self, engine_file_path):
        """从已经存在的文件中读取 TRT 模型
        Args:
            engine_file_path: 已经存在的 TRT 模型的路径
        Returns:
            加载完成的 engine
        """
        logger = trt.Logger(trt.Logger.WARNING)
        logger.min_severity = trt.Logger.Severity.ERROR
        runtime = trt.Runtime(logger)
        # trt.init_libnvinfer_plugins(logger,'') # initialize TensorRT plugins
        engine_path = os.path.realpath(engine_file_path)
        print("Loading TRT fil from : {}".format(engine_file_path))
        with open(engine_path, "rb") as f:
            serialized_engine = f.read()
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        assert engine, "反序列化之后的 engien 为空，确保转换过程的正确性 . "
        print("From {} load engine sucess . ".format(engine_file_path))
        return engine

    def output_spec(self):
        """
        Get the specs for the output tensors of the network. Useful to prepare memory allocations.
        :return: A list with two items per element, the shape and (numpy) datatype of each output tensor.
        """
        specs = []
        for o in self.outputs:
            specs.append((o['shape'], o['dtype']))
        return specs

    def infer(self, img):
        """
        Execute inference on a batch of images. The images should already be batched and preprocessed, as prepared by
        the ImageBatcher class. Memory copying to and from the GPU device will be performed here.
        :param batch: A numpy array holding the image batch.
        :param scales: The image resize scales for each image in this batch. Default: No scale postprocessing applied.
        :return: A nested list for each image in the batch and each detection in the list.
        """

        # Prepare the output data.
        outputs = []
        for shape, dtype in self.output_spec():
            outputs.append(np.zeros(shape, dtype))

        # Process I/O and execute the network.
        common.memcpy_host_to_device(self.inputs[0]['allocation'], np.ascontiguousarray(img))

        self.context.execute_v2(self.allocations)
        for o in range(len(outputs)):
            common.memcpy_device_to_host(outputs[o], self.outputs[o]['allocation'])
        return outputs

    def detect_video(self, video_path, conf=0.5, end2end=False):
        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter('results.avi',fourcc,fps,(width,height))
        fps = 0
        import time
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            blob, ratio = prepareImage(frame, self.imgsz, self.mean, self.std)
            t1 = time.time()
            data = self.infer(blob)
            fps = (fps + (1. / (time.time() - t1))) / 2
            frame = cv2.putText(frame, "FPS:%d " %fps, (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 255), 2)
            if end2end:
                num, final_boxes, final_scores, final_cls_inds = data
                final_boxes = np.reshape(final_boxes/ratio, (-1, 4))
                dets = np.concatenate([np.array(final_boxes)[:int(num[0])], np.array(final_scores)[:int(num[0])],
                                       np.array(final_cls_inds)[:int(num[0])]], axis=-1)
            else:
                predictions = np.reshape(data, (1, -1, int(5+self.n_classes)))[0]
                dets = self.postprocess(predictions,ratio)

            if dets is not None:
                final_boxes, final_scores, final_cls_inds = dets[:,
                                                                :4], dets[:, 4], dets[:, 5]
                frame = vis(frame, final_boxes, final_scores, final_cls_inds,
                                conf=conf, class_names=self.class_names)
            cv2.imshow('frame', frame)
            out.write(frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        out.release()
        cap.release()
        cv2.destroyAllWindows()

    def inference(self, img_path, conf=0.5, end2end=False):
        origin_img = cv2.imread(img_path)
        # img, ratio = preproc(origin_img, self.imgsz, self.mean, self.std)
        img, ratio, dwdh = letterbox(origin_img, self.imgsz)
        data = self.infer(img)
        if end2end:
            num, final_boxes, final_scores, final_cls_inds  = data
            # final_boxes, final_scores, final_cls_inds  = data
            dwdh = np.asarray(dwdh * 2, dtype=np.float32)
            final_boxes -= dwdh
            final_boxes = np.reshape(final_boxes/ratio, (-1, 4))
            final_scores = np.reshape(final_scores, (-1, 1))
            final_cls_inds = np.reshape(final_cls_inds, (-1, 1))
            dets = np.concatenate([np.array(final_boxes)[:int(num[0])], np.array(final_scores)[:int(num[0])],
                                   np.array(final_cls_inds)[:int(num[0])]], axis=-1)
        else:
            predictions = data[0]
            #v8 的结果是 [1, 84, 8400]
            # dets = self.postprocess_2(predictions, ratio, dwdh)
            # v5  结果能出来，[1, 25200, 85]但是调节位置错误，问题不大
            dets = self.postprocess(predictions, ratio, dwdh=dwdh, num_classes=3)

        if dets is not None:
            final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
            origin_img = vis(origin_img, final_boxes, final_scores, final_cls_inds,
                             conf=conf, class_names=self.class_names)
        return origin_img

    def postprocess(self, prediction, ratio, dwdh=(0, 0), num_classes=80, conf_thres=0.25, iou_thres=0.45):
        """对检测头输出的多个检测框, 进行非极大值抑制
        Args:
            prediction  : 检测头输出的全部的检测框，具体的维度信息为 (batch * 25200 * 85)
            ratio       : 恢复到原图的尺度需要的尺度改变大小
            conf_thres  : 检测框的置信度. Defaults to 0.25.
            iou_thres   : nms 时框和框之间的 iou 阈值. Defaults to 0.45.
        Returns:
            输出每张图片上进行过非极大值抑制的结果，最终的维度为：(n, 6); 6 -> [xyxy, conf, cls]
        """
        if self.mode == 'v5':
            # 1、首先将 prediction 的维度转换一下 (batch * 25200 * 85)-> (25200, 85)
            prediction = np.reshape(prediction, (1, -1, int(5 + num_classes)))[0]
            # 2、得到每个检测框的的得分数，-> box_scores = obj_conf * cls_conf
            scores = prediction[:, 4:5] * prediction[:, 5:]
        if self.mode == 'v8':
            prediction = prediction.squeeze()
            prediction = prediction.transpose(1, 0)
            scores = prediction[:, 4:]
        # 3、转换 (center x, center y, width, height) to (x1, y1, x2, y2), 并转换为适应图片的大小
        boxes_xyxy = xywh2xyxy(prediction[:, :4])
        # 4、按照不同的 类别 进行 nms
        boxes_after_nms = multiclass_nms(boxes_xyxy, scores, conf_thres, iou_thres)
        # 5、letterbox 方式还原
        dwdh = np.asarray(dwdh * 2, dtype=np.float32)
        boxes_after_nms[:, :4] -= dwdh
        boxes_after_nms[:, :4] /= ratio
        return boxes_after_nms

    def get_fps(self):
        import time
        img = np.ones((1,3,self.imgsz[0], self.imgsz[1]))
        img = np.ascontiguousarray(img, dtype=np.float32)
        for _ in range(5):  # warmup
            _ = self.infer(img)

        t0 = time.perf_counter()
        for _ in range(100):  # calculate average time
            _ = self.infer(img)
        print(100/(time.perf_counter() - t0), 'FPS')

