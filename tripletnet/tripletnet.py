import torch
import torch.nn as nn
import torch.nn.functional as F



class VGGTriplet(nn.Module):

    def __init__(self,Model):
        super(VGGTriplet,self).__init__()

        self.features = Model.features
        self.classifier = Model.classifier
        self.classifier[3].register_forward_hook(self._printnorm)
        
        self.cnn_feature = []

    def forward(self,query,positive,negative):

        query = self.features(query)
        query = query.view(query.size(0), -1)
        query = self.classifier(query)

        
        positive = self.features(positive)
        positive = positive.view(positive.size(0), -1)
        positive = self.classifier(positive)

        
        negative = self.features(negative)
        negative = negative.view(negative.size(0), -1)
        negative = self.classifier(negative)

        return query,positive,negative

    def _printnorm(self,layer ,input, output):
        #save cnn_feature
    
        if len(self.cnn_feature) == 3:
            self.cnn_feature = []

        
        self.cnn_feature.append(output)
       
