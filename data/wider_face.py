import os
import os.path
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np

class WiderFaceDetection(data.Dataset):
    def __init__(self, data_path, preproc=None):
        self.preproc = preproc
        self.imgs_path = []
        self.words = []
        txt_path = os.path.join(data_path,"jd_all.txt")
        f = open(txt_path,'r')
        lines = f.readlines()
        labels = []
        for line in lines:
            line = line.strip().split()
            label = line[1:201]
            label = [float(x) for x in label]
            imgname = line[0]
            self.imgs_path.append(os.path.join(data_path,"images",imgname))
            labels.append(label)

        self.words = labels

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        img = cv2.imread(self.imgs_path[index])
        # print(self.imgs_path[index])
        height, width, _ = img.shape
        # print(len(self.words))
        label = self.words[index]
        # print(labels)
        annotations = np.zeros((0, 201))
        # print(annotations)
        if len(label) == 0:
            return annotations
        # for idx, label in enumerate(labels):
            # print("idx:",idx,label)
        annotation = np.zeros((1, 201))
        # bbox
        annotation[0, 0] = label[-4]  # x1
        annotation[0, 1] = label[-3]  # y1
        annotation[0, 2] = label[-2]  # x2
        annotation[0, 3] = label[-1]  # y2

        # landmarks
        for nx in range(196):
            annotation[0, 4+nx] = label[nx]
        # annotation[0, 4] = label[4]    # l0_x
        # annotation[0, 5] = label[5]    # l0_y
        # annotation[0, 6] = label[7]    # l1_x
        # annotation[0, 7] = label[8]    # l1_y
        # annotation[0, 8] = label[10]   # l2_x
        # annotation[0, 9] = label[11]   # l2_y
        # annotation[0, 10] = label[13]  # l3_x
        # annotation[0, 11] = label[14]  # l3_y
        # annotation[0, 12] = label[16]  # l4_x
        # annotation[0, 13] = label[17]  # l4_y
        if (annotation[0, 4]<0):
            annotation[0, 200] = -1
        else:
            annotation[0, 200] = 1

        annotations = np.append(annotations, annotation, axis=0)
        # print(annotations)
        target = np.array(annotations)
        if self.preproc is not None:
            img, target = self.preproc(img, target)

        return torch.from_numpy(img), target

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                targets.append(annos)

    return (torch.stack(imgs, 0), targets)
