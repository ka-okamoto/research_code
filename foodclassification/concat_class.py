#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
import sklearn.cross_validation as crv
from PIL import ImageFile
import argparse

parser = argparse.ArgumentParser("description = VGG Triplet class feature")
parser.add_argument('--month',type = str ,default = 7,metavar = "month",
                    help = "month")
parser.add_argument('--batch-size', type=int, default=50, metavar='N',
                    help='input batch size for training (default: 50)')
parser.add_argument('--device',type = str, default = 0,
                    help = "device number")

args = parser.parse_args()

ImageFile.LOAD_TRUNCATED_IMAGES = True#大きな画像を高速にロードするために必要

device = torch.device("cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu") 
#-----------------testdata----------------------------
data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

if int(args.month) < 7:
 class_dir = '/export/space/nagano-t/tw/img/2016/{}/'.format(args.month)
 splittag = 'tw'
else:
 class_dir = '/export/space/okamoto-ka/resize_img2016/{}/'.format(args.month)
 splittag = 'okamoto-ka'

print('data_load:{}'.format(class_dir))

twitter_dataset = datasets.ImageFolder(class_dir,data_transforms)
dataloader = torch.utils.data.DataLoader(twitter_dataset,batch_size=50,shuffle=False,num_workers=4)

#-----------------classification---------------------- 
savetxt = 'food_16_{}.txt'.format(args.month)
namepath = twitter_dataset.samples

def classification(model,txtpath):
 
 model.eval()
 txt_list = []
 print('start_class')
 with torch.no_grad():
   for i,(inputs,labels) in enumerate(dataloader):
      if i % 1000 == 0:
       print('{} / {}'.format(i,len(twitter_dataset)/args.batch_size))
       print(len(txt_list))

      inputs = inputs.to(device)
      outputs = model(inputs)
      outputs = outputs.cpu()
      _,preds = torch.max(outputs,1)
      
      preds = preds.numpy()
      #0が食事画像で1が非食事画像
      idx = np.where(preds != 1)[0].tolist()
      fnames = [namepath[i * args.batch_size + j][0] + '\n' for j in idx]
      txt_list.extend(fnames)


 with open(savetxt,'w') as f:
  f.writelines(txt_list)
 
          
#-----------------model-------------------------------
if __name__ == '__main__':

 model_ft = models.resnet50(pretrained=True)#初期のモデルを定義

 num_ftrs = model_ft.fc.in_features
 model_ft.fc = nn.Linear(num_ftrs,2)
 weightpass = 'trained_weight.pth'#事前に学習した非食事,食事分類モデルの重み
 param = torch.load(weightpass)
 model_ft.load_state_dict(param)#学習した重みをモデルに変換
 model_ft = model_ft.to(device)

 #-----------------vgg model----------------------------
 classification(model_ft,savetxt)

