#coding: UTF-8

import torch
import torch.nn as nn
import torch.nn.functional as F

def tripletloss(emb_q,emb_p,emb_n,margin,device):
    

     criterion =  torch.nn.MarginRankingLoss(margin = margin)
     
     emb_q = emb_q.view(emb_q.size(0),-1)
     emb_p = emb_p.view(emb_p.size(0),-1)
     emb_n = emb_n.view(emb_n.size(0),-1)
    
     dist_p = F.pairwise_distance(emb_q, emb_p, 2)
     dist_n = F.pairwise_distance(emb_q, emb_n, 2)
     y = torch.FloatTensor(dist_p.size()).fill_(-1).to(device)
        
     loss_triplet = criterion(dist_p, dist_n, y)
     loss_embedd = emb_q.norm(2) + emb_p.norm(2) + emb_n.norm(2)
     loss = loss_triplet + 0.001 * loss_embedd

     return loss 

def classloss(output,label):
     criterion =  nn.CrossEntropyLoss()
     
     _,preds = torch.max(output,1)
     loss = criterion(output,label)
     corrects = torch.sum(preds == label.data)

     return loss, corrects
