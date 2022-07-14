import sys
sys.path.insert(0, './yolov5')

from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device, time_synchronized
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np


class VideoCamera:

    def __init__(self, source):
        self.palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
        self.out = 'inference/output2'
        self.weights = 'yolov5/weights/yolov5s.pt'
        self.view_img = False
        self.save_txt = 'false'
        self.imgsz = 640
        self.source = source
        self.webcam = self.source == '0' or self.source.startswith(
            'rtsp') or self.source.startswith('http') or self.source.endswith('.txt')
        self.dataset = LoadImages(self.source, img_size=self.imgsz)

    # 把YOLOv5的标注框格式转换为Deepsort的标注框输入格式
    def bbox_rel(self, *xyxy):
        # 读取YOLOv5的格式：左上顶点坐标(x,y)，长，宽
        bbox_left = min([xyxy[0].item(), xyxy[2].item()])
        bbox_top = min([xyxy[1].item(), xyxy[3].item()])
        bbox_w = abs(xyxy[0].item() - xyxy[2].item())
        bbox_h = abs(xyxy[1].item() - xyxy[3].item())
        # 换算为Deepsort格式：中心点(x,y)，长，宽
        x_c = (bbox_left + bbox_w / 2)
        y_c = (bbox_top + bbox_h / 2)
        w = bbox_w
        h = bbox_h
        return x_c, y_c, w, h


    def compute_color_for_labels(self, label):
        """
        Simple function that adds fixed color depending on the class
        """
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in self.palette]
        return tuple(color)


    def draw_boxes(self, img, bbox, identities=None, offset=(0, 0)):
        for i, box in enumerate(bbox):
            x1, y1, x2, y2 = [int(i) for i in box]
            x1 += offset[0]
            x2 += offset[0]
            y1 += offset[1]
            y2 += offset[1]
            # box text and bar
            id = int(identities[i]) if identities is not None else 0
            color = self.compute_color_for_labels(id)
            label = '{}{:d}'.format("", id)
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            cv2.rectangle(
                img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
            cv2.putText(img, label, (x1, y1 +
                                     t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
        return img


    # xyxy2tlwh函数  这个函数一般都会自带
    def xyxy2tlwh(self, x):
        '''
        (top left x, top left y,width, height)
        '''
        y = torch.zeros_like(x) if isinstance(x,
                                              torch.Tensor) else np.zeros_like(x)
        y[:, 0] = x[:, 0]
        y[:, 1] = x[:, 1]
        y[:, 2] = x[:, 2] - x[:, 0]
        y[:, 3] = x[:, 3] - x[:, 1]
        return y

    def detect(self, save_img=False):
        # out, source, weights, view_img, save_txt, imgsz = \
        #     opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
        # out = 'inference/output2'
        # source = 'E:\\coco2017\\video\\test.mp4'
        # weights = 'yolov5/weights/yolov5s.pt'
        # view_img = False
        # save_txt = 'false'
        # imgsz = 640
        # webcam = self.source == '0' or source.startswith(
        #     'rtsp') or source.startswith('http') or source.endswith('.txt')

        # initialize deepsort
        cfg = get_config()
        # cfg.merge_from_file(opt.config_deepsort)
        cfg.merge_from_file('deep_sort_pytorch/configs/deep_sort.yaml')
        deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                            max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                            use_cuda=True)

        # Initialize
        # device = select_device(opt.device)
        device = select_device('')
        if os.path.exists(self.out):
            shutil.rmtree(self.out)  # delete output folder
        os.makedirs(self.out)  # make new output folder
        half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        model = torch.load(self.weights, map_location=device)[
            'model'].float()  # load to FP32
        model.to(device).eval()
        if half:
            model.half()  # to FP16

        # Set Dataloader
        vid_path, vid_writer = None, None
        if self.webcam:
            self.view_img = True
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(self.source, img_size=self.imgsz)
        else:
            self.view_img = True
            save_img = True
            dataset = LoadImages(self.source, img_size=self.imgsz)

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names

        # Run inference
        t0 = time.time()
        img = torch.zeros((1, 3, self.imgsz, self.imgsz), device=device)  # init img
        # run once
        _ = model(img.half() if half else img) if device.type != 'cpu' else None

        save_path = str(Path(self.out))\

        txt_path = str(Path(self.out)) + '/results.txt'

        # 用来存取id对应的轨迹点信息
        dict_box=dict()

        for frame_idx, (path, img, im0s, vid_cap) in enumerate(self.dataset):
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            # pred = model(img, augment=opt.augment)[0]
            pred = model(img, augment=False)[0]

            # Apply NMS
            # pred = non_max_suppression(
            #     pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            pred = non_max_suppression(
                pred, 0.4, 0.5, classes=[0], agnostic=False)
            t2 = time_synchronized()

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if self.webcam:  # batch_size >= 1
                    p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
                else:
                    p, s, im0 = path, '', im0s

                s += '%gx%g ' % img.shape[2:]  # print string
                save_path = str(Path(self.out) / Path(p).name)

                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(
                        img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, names[int(c)])  # add to string

                    bbox_xywh = []
                    confs = []

                    # Adapt detections to deep sort input format
                    for *xyxy, conf, cls in det:
                        x_c, y_c, bbox_w, bbox_h = self.bbox_rel(*xyxy)
                        obj = [x_c, y_c, bbox_w, bbox_h]
                        bbox_xywh.append(obj)
                        confs.append([conf.item()])

                    xywhs = torch.Tensor(bbox_xywh)
                    confss = torch.Tensor(confs)

                    # Pass detections to deepsort
                    outputs = deepsort.update(xywhs, confss, im0)
                    # outputs = [x1, y1, x2, y2, track_id]

                    # 存放当前帧中出现的所有id，由于这不是全局变量，因此切换帧时数组会重新初始化
                    current_id = []

                    if len(outputs) > 0:
                        bbox_xyxy = outputs[:, :4]  # 提取前四列  坐标
                        identities = outputs[:, -1]  # 提取最后一列 ID, 注意这好像是个二维数组
                        box_xywh = self.xyxy2tlwh(bbox_xyxy)
                        # xyxy2tlwh是坐标格式转换，从x1, y1, x2, y2转为top left x ,top left y, w, h 具体函数看文章最后
                        for j in range(len(box_xywh)):
                            x_center = box_xywh[j][0] + box_xywh[j][2] / 2  # 求框的中心x坐标
                            y_center = box_xywh[j][1] + box_xywh[j][3] / 2  # 求框的中心y坐标
                            id = outputs[j][-1]
                            center = [x_center, y_center]
                            current_id.append(id)  # 将当前帧的中出现的id扔进去
                            dict_box.setdefault(id, []).append(center)  # 这个字典需要提前定义 dict_box = dict()
                        # 以下为画轨迹，原理就是将前后帧同ID的跟踪框中心坐标连接起来
                        if frame_idx > 2:
                            for key, value in dict_box.items():
                                # 如果当前帧中有这个id，则画出该id对应的轨迹
                                if key in current_id:
                                    for a in range(len(value) - 1):
                                        # 这里给轨迹随机上色
                                        # color = COLORS_10[key % len(COLORS_10)]
                                        index_start = a
                                        index_end = index_start + 1
                                        cv2.line(im0, tuple(map(int, value[index_start])), tuple(map(int, value[index_end])),
                                                 # map(int,"1234")转换为list[1,2,3,4]
                                                (255, 0, 0), thickness=2, lineType=8)

                    # draw boxes for visualization
                    if len(outputs) > 0:
                        bbox_xyxy = outputs[:, :4]
                        identities = outputs[:, -1]
                        self.draw_boxes(im0, bbox_xyxy, identities)

                    # Write MOT compliant results to file
                    if self.save_txt and len(outputs) != 0:
                        for j, output in enumerate(outputs):
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2]
                            bbox_h = output[3]
                            identity = output[-1]
                            with open(txt_path, 'a') as f:
                                f.write(('%g ' * 10 + '\n') % (frame_idx, identity, bbox_left,
                                                               bbox_top, bbox_w, bbox_h, -1, -1, -1, -1))  # label format

                else:
                    deepsort.increment_ages()

                # Print time (inference + NMS)
                print('%sDone. (%.3fs)' % (s, t2 - t1))



            ret, img_encode = cv2.imencode('.jpg', im0)
            return img_encode.tobytes()




