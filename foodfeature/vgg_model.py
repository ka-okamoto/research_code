#coding: UTF-8
import torch
import torch.nn as nn
from torchvision import models

class VGGFeature(nn.Module):

    def __init__(self,args):
        super(VGGFeature,self).__init__()

        vgg = models.vgg16_bn(pretrained = True)
        vgg.classifier[-1] =  nn.Linear(4096, 106) 
        if args.feature == 'food':
                weightpath = 'food_cnn_train.pth'
                param = torch.load(weightpath)
                vgg.load_state_dict(param)

        elif args.feature == 'triplet':
                weightpath = 'triplet_cnn_train.pth'
                param = torch.load(weightpath)
                vgg.load_state_dict(param)

        elif args.feature == 'imagenet':
                weightpath = ''

        print(weightpath)
    
 
        self.features = vgg.features
        self.classifier = vgg.classifier
        self.classifier[3].register_forward_hook(self._feature)
        
        self.cnn_feature = []

    def forward(self,input):

        output = self.features(input)
        output = output.view(output.size(0), -1)
        output = self.classifier(output)

        _,pred = torch.max(output,1)

        return output , pred

    def _feature(self,layer ,input, output):
        #save cnn_feature
    
        if len(self.cnn_feature) == 1:
            self.cnn_feature = []

        self.cnn_feature.append(output)

