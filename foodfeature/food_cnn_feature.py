#coding: UTF-8

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torch.utils.data import Dataset,DataLoader
from torchvision import datasets, models, transforms
import time
from PIL import Image
import random
from sklearn.decomposition import PCA
import argparse
import sys
import os

from image_dataset import TxtReadData
from vgg_model import VGGFeature

parser = argparse.ArgumentParser(description = 'before duplicate or after duplicate')

parser.add_argument('-f','--feature',default = 'imagenet')
parser.add_argument('-d','--datapath',default = 'files.txt')

parser.add_argument('-m','--month',default = '0')

args = parser.parse_args()

#-------------------seed Fixation ----------------------
def worker_init_fn(worker_id):
    random.seed(worker_id)

#--------------------transform--------------------------
data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


datapath = args.datapath
dataset = TxtReadData(datapath,data_transforms)
filenamelist = dataset.filenamelist

dataloader = DataLoader(dataset,batch_size=20,shuffle=False,num_workers=4,worker_init_fn = worker_init_fn)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#-----------------------train---------------------------
def extraction(t_model):

     
 since = time.time()
 
 pred_list = []
 save_feature = []
 
 for i, inputs in enumerate(dataloader): 
        t_model.eval()
        
        if i % 500 == 0:
                sys.stdout.write('\r {}/{}'.format(i,len(filenamelist) /20))
                sys.stdout.flush()
                time.sleep(0.01)

        inputs = inputs.to(device)
        output,pred = t_model(inputs)        
        ## pred 
        pred =  pred.cpu().numpy().tolist()

        for p in pred:
                pred_list.append(p)

        ## feature
        feature = torch.squeeze(t_model.cnn_feature[0].cpu())
        feature = feature.detach().numpy()
        feature[np.isnan(feature)] = 0
        #normalize
        try:
                feature = feature.T / np.linalg.norm(feature,axis = 1)
        except:
                print(feature.shape)
                size = feature.shape[0]
                feature = np.reshape(feature,(1,size))
                feature = feature.T / np.linalg.norm(feature,axis = 1)

        feature = feature.T
        # dimensionality reduction by pca
        feature_list = feature.tolist()
        for f in feature_list:
                save_feature.append(f)
        
 ##推論カテゴリの保存
 save_pred = [name + ',' + str(pre) for name, pre in zip(filenamelist,pred_list) ]
 save_txt_name = 'pred_class_{}.txt'.format(args.feature)
 
 
 with open(save_txt_name,'w') as f:
         f.writelines('\n'.join(save_pred))

 ## feature             
 save_feature_name = 'feature_{}.npy'.format(args.feature)
 save_feature = np.array(save_feature)
 pca = PCA(n_components = 128)
 feature_pca = pca.fit_transform(save_feature)
 np.save(save_feature_name,feature_pca)

#--------------------main-------------------------------
if __name__ == "__main__":
        
 torch.manual_seed(1)
 #tripletnet
 t_model  = VGGFeature(args)
 t_model.to(device)
  
 extraction(t_model)
