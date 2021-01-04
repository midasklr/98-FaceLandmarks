import torch
import torch.nn as nn
import torchvision.models.detection.backbone_utils as backbone_utils
import torchvision.models._utils as _utils
import torch.nn.functional as F
from collections import OrderedDict

from models.net import MobileNetV1 as MobileNetV1
from models.net import MobileNetV3_Small as MobileNetV3
from models.net import MobileNetV2 as MobileNetV2
from models.net import EfficientNet as EfficientNet
from models.net import FPN as FPN
from models.net import SSH as SSH
from models.net import PAN as PAN



class ClassHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(ClassHead,self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels,self.num_anchors*2,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()
        
        return out.view(out.shape[0], -1, 2)

class BboxHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(BboxHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*4,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 4)

class LandmarkHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(LandmarkHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*196,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 196)

class RetinaFace(nn.Module):
    def __init__(self, cfg = None, phase = 'train'):
        """
        :param cfg:  Network related settings.
        :param phase: train or test.
        """
        super(RetinaFace,self).__init__()
        self.phase = phase
        backbone = None
        if cfg['name'] == 'mobilenet0.25':
            backbone = MobileNetV1()
            if cfg['pretrain']:
                checkpoint = torch.load("./weights/mobilenetV1X0.25_pretrain.tar", map_location=torch.device('cpu'))
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    name = k[7:]  # remove module.
                    new_state_dict[name] = v
                # load params
                backbone.load_state_dict(new_state_dict)
        elif cfg['name'] == 'Resnet50':
            import torchvision.models as models
            backbone = models.resnet50(pretrained=cfg['pretrain'])
        elif cfg['name'] == 'mobilenetv3':
            backbone = MobileNetV3()
            # bbweight = backbone.state_dict()
            # layername = []
            # oldln = []
            # for i,key in enumerate(bbweight):
            #     print("Layer {} : {}; Size : {}".format(i,key,bbweight[key].size()))
            #     layername.append(key)
            # cpk = torch.load("/home/hwits/Documents/ImageRetrieval/mobilenetv3/mbv3_small.pth.tar")['state_dict']
            # # print(cpk.keys())
            # print(cpk['epoch'])
            # print("="*200)
            # for i,key in enumerate(cpk):
            #     print("Old Layer {} : {}; Size : {}".format(i,key,cpk[key].size()))
            #     if i < len(layername):
            #         oldln.append(key)
            # print(bbweight.keys())
            # print(oldln)
            #
            # import collections
            # mbv3_cpk = collections.OrderedDict()
            # for i in range(len(layername)):
            #     mbv3_cpk[layername[i]] = cpk[oldln[i]]
            #
            # torch.save(mbv3_cpk,"mobilenetv3_truck.pth")
            if cfg['pretrain']:
                checkpoint = torch.load("./weights/mobilenetv3_truck.pth", map_location=torch.device('cpu'))
                print("Loading Pretrained Weights ...")
                backbone.load_state_dict(checkpoint)
        elif cfg['name'] == 'mobilenetv2_0.1':
            backbone = MobileNetV2()
            if cfg['pretrain']:
                checkpoint = torch.load("./weights/mobilenetv2_0.1_face.pth", map_location=torch.device('cpu'))
                model_dict = backbone.state_dict()
                pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}  # 用于过滤掉修改结构处的权重
                model_dict.update(pretrained_dict)
               # print("Pre-Trained:",pretrained_dict.keys())
                backbone.load_state_dict(model_dict)
        elif cfg['name'] == 'efficientnetb0':
            backbone = EfficientNet.from_name("efficientnet-b0")
            if cfg['pretrain']:
                checkpoint = torch.load("./weights/efficientnetb0_face.pth", map_location=torch.device('cpu'))
                backbone.load_state_dict(checkpoint)
                print("succeed loaded weights...")
        self.body = _utils.IntermediateLayerGetter(backbone, cfg['return_layers'])
        if cfg['name'] == 'mobilenet0.25':
            in_channels_stage2 = cfg['in_channel']
            in_channels_list = [
                # in_channels_stage2 * 2,
                in_channels_stage2*4,
                in_channels_stage2*8,
            ]
        elif cfg['name'] == 'mobilenetv2_0.1':
            in_channels_stage3 = cfg['in_channel3']
            in_channels_stage4 = cfg['in_channel4']
            in_channels_list = [
                in_channels_stage3,
                in_channels_stage4
            ]
        elif cfg['name'] == 'mobilenetv3':
            in_channels_stage1 = cfg['in_channel0']
            in_channels_stage2 = cfg['in_channel1']
            in_channels_stage3 = cfg['in_channel2']
            in_channels_list = [
                in_channels_stage1,
                in_channels_stage2,
                in_channels_stage3,
            ]
        elif cfg['name'] == 'efficientnetb0':
            # in_channels_stage2 = cfg['in_channel1']
            in_channels_stage3 = cfg['in_channel2']
            in_channels_list = [
                # in_channels_stage2 * 2,
                # in_channels_stage2,
                in_channels_stage3,
            ]
        out_channels = cfg['out_channel']
        self.pan = PAN(in_channels_list,out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

        self.ClassHead = self._make_class_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.BboxHead = self._make_bbox_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.LandmarkHead = self._make_landmark_head(fpn_num=3, inchannels=cfg['out_channel'])

    def _make_class_head(self,fpn_num=2,inchannels=64,anchor_num=2):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels,anchor_num))
        return classhead
    
    def _make_bbox_head(self,fpn_num=2,inchannels=64,anchor_num=2):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels,anchor_num))
        return bboxhead

    def _make_landmark_head(self,fpn_num=2,inchannels=64,anchor_num=2):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels,anchor_num))
        return landmarkhead

    def forward(self,inputs):
        out = self.body(inputs)
        # print("out size : ",out.size())
        # FPN
        pan = self.pan(out)

        # SSH
        feature1 = self.ssh1(pan[0])
        feature2 = self.ssh2(pan[1])
        feature3 = self.ssh2(pan[2])
        # print("Feature1 size {} 2 size : {}".format(feature2.size(),feature3.size()))
        # features = [feature1, feature2, feature3]
        features = [feature1, feature2,feature3]

        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)],dim=1)
        ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)

        if self.phase == 'train':
            output = (bbox_regressions, classifications, ldm_regressions)
        else:
            output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions)
        return output

