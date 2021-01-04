# -*- coding: utf-8 -*-
# @Time : 2020/9/10 下午5:29 
# @Author : midaskong 
# @File : face_direction.py 
# @Description:


from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg_mnetv1, cfg_mnetv2, cfg_mnetv3, cfg_efnetb0
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
import time

REFERENCE_FACIAL_POINTS = [        # default reference facial points for crop_size = (112, 112); should adjust REFERENCE_FACIAL_POINTS accordingly for other crop_size
    [30.29459953+8,  51.69630051],
    [65.53179932+8,  51.50139999],
    [48.02519989+8,  71.73660278],
    [33.54930115+8,  92.3655014],
    [62.72990036+8,  92.20410156]
]

direction = {0: 'front',1: 'right',2: 'left',3: 'up',4: 'down',5: 'others'}

parser = argparse.ArgumentParser(description='Retinaface')

parser.add_argument('-m', '--trained_model', default='./weights/mobilenetv2_0.1_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='mobilenetv2', help='Backbone network mobile0.25 ,mobilenetv2 ,mobilenetv3 or efficientnetb0')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
args = parser.parse_args()


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def Face_Direction(landmarks, face):
    """ @landmarks: (x1,y1,x2,y2,...,x5,y5);
        @w: face width;
        @h: face height;
        @:return:   0: front;
                    1: right;
                    2: left;
                    3: up;
                    4: down;
                    5: others
    """
    h,w = face.shape[:2]
    eye_centerx = (landmarks[0]+landmarks[2])/2
    eye_centery = (landmarks[1]+landmarks[3])/2
    nosex = landmarks[4]
    nosey = landmarks[6]
    mouse_centerx = (landmarks[6]+landmarks[8])/2
    mouse_centery = (landmarks[7]+landmarks[9])/2

    for kp in range(0,10,2):
        cv2.circle(face, (landmarks[kp], landmarks[kp+1]), 2, (0, 0, 255), 8)

    cv2.circle(face, (int(eye_centerx), int(eye_centery)), 2, (255, 0, 255), 8)

    cv2.imshow("a",face)
    cv2.waitKey(0)

    if (eye_centerx>w*0.6) :
        return 1
    elif (eye_centerx<w*0.4):
        return 2
    elif (w*0.4<eye_centerx<w*0.6) and (w*0.3<mouse_centerx<w*0.7):
        return 0
    elif (nosey < h*0.5):
        return 3
    else:
        return 5



if __name__ == '__main__':
    torch.set_grad_enabled(False)
    cfg = None
    if args.network == "mobile0.25":
        cfg = cfg_mnetv1
    elif args.network == "mobilenetv2":
        cfg = cfg_mnetv2
    elif args.network == "mobilenetv3":
        cfg = cfg_mnetv3
    elif args.network == "efficientnetb0":
        cfg = cfg_efnetb0
    # net and model
    net = RetinaFace(cfg=cfg, phase = 'test')
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')
    print(net)
    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)

    resize = 1
    nums = 0

    # testing begin
    for imgs in range(1):
        image_path = "/home/hwits/Documents/FaceRec/EdgeFace/LightWeightFaceDetector/images/2bdb4f733416cc8a9390d449f0768744.jpg"
        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
        # print("img raw:",img_raw.shape)
        h, w = img_raw.shape[:2]
        if h>w:
            img_raw = cv2.resize(img_raw,(640,int(640*h/w)))
        else:
            img_raw = cv2.resize(img_raw, (int(640*w/h), 640))
        #img_raw = cv2.resize(img_raw,(int(img_raw.shape[1]*1.5),int(img_raw.shape[0]*1.5)))
        img = np.float32(img_raw)

        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])

        img -= (104, 117, 123)

        img = img.transpose(2, 0, 1)
        # print("image : ", img[2][0])

        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)

        tic = time.time()
        loc, conf, landms = net(img)  # forward pass
        # print('net forward time: {:.4f}'.format(time.time() - tic))
        # print("loc : ",conf.size())
        # print("loc idx: ", torch.argmax(conf[0][:,1]))
        # print("loc max: ", torch.max(conf[0][:, 1]))
        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
        print(loc)
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > args.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, args.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:args.keep_top_k, :]
        landms = landms[:args.keep_top_k, :]

        dets = np.concatenate((dets, landms), axis=1)

        # show image
        if args.save_image:
            for b in dets:
                if b[4] < args.vis_thres:
                    continue
                text = "{:.4f}".format(b[4])
                b = list(map(int, b))
                # cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 4)
                cx = b[0]
                cy = b[1] + 12
                cv2.putText(img_raw, text, (cx, cy),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

                # landms
                # cv2.circle(img_raw, (b[5], b[6]), 2, (0, 0, 255), 8)
                # cv2.circle(img_raw, (b[7], b[8]), 2, (0, 255, 255), 8)
                # cv2.circle(img_raw, (b[9], b[10]), 2, (255, 0, 255), 8)
                # cv2.circle(img_raw, (b[11], b[12]), 2, (0, 255, 0), 8)
                # cv2.circle(img_raw, (b[13], b[14]), 2, (255, 0, 0), 8)
                face = img_raw[b[1]:b[3],b[0]:b[2]]

                landmk = [b[5]-b[0], b[6]-b[1],b[7]-b[0], b[8]-b[1],b[9]-b[0], b[10]-b[1],b[11]-b[0], b[12]-b[1],b[13]-b[0], b[14]-b[1]]

                # print(landmkx)
                # print(landmky)

                # cv2.imshow("face",img_raw)
                # cv2.waitKey(0)


                ret = Face_Direction(landmk, face)
                print(direction[ret])
