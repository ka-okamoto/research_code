#coding: UTF-8

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import copy
import numpy as np
import time
import warnings
import sys


from image_dataset import TripletCategory
from tripletnet import VGGTriplet
import loss 

warnings.filterwarnings('ignore')

# Training settings
parser = argparse.ArgumentParser(description='PyTorch  Example')
parser.add_argument('--t-batch-size', type=int, default=8, metavar='N',
                    help='input batch size for training (default: 20)')
parser.add_argument('--epochs', type=int, default=25, metavar='N',
                    help='number of epochs to train (default: 25)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.1, metavar='M',
                    help='SGD momentum (default: 0.1)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--margin', type=float, default=0.3, metavar='M',
                    help='margin for triplet loss (default: 0.3)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='TripletNet', type=str,
                    help='name of experiment')

args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#-------------------------------------------data_transforms-------------------------------------------
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'eval': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

#-------------------------------------------t_dataset-------------------------------------------
#学習と評価のtxtはクエリ、ポジティブ、ネガティブの画像のパスです。
#query1path,positive1path,negative1path
#query2path,positive2path,negative2path
#   .
#   .
#   .

train_txt = 'train.txt'
eval_txt = 'eval.txt'

train_data =  TripletCategory(filenames_txt = train_txt, transform = data_transforms['train'])
eval_data = TripletCategory(filenames_txt = eval_txt, transform = data_transforms['eval'])

train_loader = torch.utils.data.DataLoader(
    train_data,batch_size=args.t_batch_size, shuffle=True, num_workers = 4)
eval_loader = torch.utils.data.DataLoader(
    eval_data,batch_size=args.t_batch_size, shuffle=True, num_workers = 4)

data_loaders = {'train':train_loader,'eval':eval_loader}
data_size = {'train':len(train_data),'eval':len(eval_data)}


#-------------------------------------------train and eval -------------------------------------------
def model_train(t_model, optimizer, num_epochs = 50 , start_epoch = 0, best_loss = 100):
     since = time.time()

     best_t_model_wts = copy.deepcopy(t_model.state_dict())
     best_loss = best_loss

     savetxt = 'save.txt'
     result = []

     for epoch in range(start_epoch,num_epochs):

        print('Epoch {}/{}'.format(epoch,num_epochs-1))
        print('-'*10)
        result.append(str(epoch) + 'epoch')
     
        for mode_phase in ['train','eval']:

            result.append(mode_phase) 

            running_t_loss,running_class_loss = 0.0,0.0
            running_t_acc,running_class_corrects = 0.0,0.0

            if mode_phase == 'train':
                 t_model.train()
            if mode_phase == 'eval':
                 t_model.eval()
    
            print(mode_phase)

            # train triplet model
            for i,(query, positive, negative,q_label,p_label,n_label) \
                in enumerate(data_loaders[mode_phase]) :
           
     
                if i % 200 == 0:
                     sys.stdout.write('\r{}/{}'.format(i,data_size[mode_phase] / args.t_batch_size))
                     sys.stdout.flush()
                     time.sleep(0.01)

                query, positive, negative = query.to(device), positive.to(device),negative.to(device)
                q_label,p_label,n_label = q_label.to(device),p_label.to(device),n_label.to(device)

                optimizer.zero_grad()

                # compute output
                with torch.set_grad_enabled(mode_phase == 'train'):
                     
                    #forward 
                    q_out,p_out,n_out = t_model(query,positive,negative) 
                    #feature value
                    emb_q,emb_p,emb_n = t_model.cnn_feature
                    t_loss = loss.tripletloss(emb_q, emb_p, emb_n,args.margin,device) 
                    #class_loss
                    q_loss,q_corrects = loss.classloss(q_out,q_label)
                    p_loss,p_corrects = loss.classloss(p_out,p_label)
                    n_loss,n_corrects = loss.classloss(n_out,n_label)

                    class_loss =  (q_loss + p_loss + n_loss) / 3
                    class_corrects =  (q_corrects + p_corrects + n_corrects) / 3

                    running_t_loss += t_loss.item()*query.size(0)
                    running_class_loss += class_loss.item()*query.size(0)

                    running_class_corrects += class_corrects
                                            
                l = 1.0
                total_loss = t_loss + l*class_loss
                if mode_phase == 'train':
                    
                    total_loss.backward()
                    optimizer.step()
                
            #class model epoch loss 
            epoch_class_loss = running_class_loss / data_size[mode_phase]
            epoch_class_acc = running_class_corrects.double() / data_size[mode_phase] 
            epoch_t_loss = running_t_loss / data_size[mode_phase]
            
            total_loss = epoch_t_loss + epoch_class_loss
            Triplet_result = 'Triplet Loss :{:.4f}'.format(epoch_t_loss)
            class_result = 'Class Loss :{:.4f} Acc: {:.4f}'.format(epoch_class_loss,epoch_class_acc)

            print(Triplet_result)
            print(class_result)

            result.append(Triplet_result)
            result.append(class_result)
            result.append('')
            
            if mode_phase == 'eval' and epoch_t_loss < best_loss:
                best_acc = epoch_class_acc
                save_checkpoint({
                     'epoch':epoch,
                     'state_dict':t_model.state_dict(),
                     'best_loss':epoch_t_loss

                })


     time_elapsed = time.time() - since     
     training_time = 'Training coplete in {:0f}m {:.0f}s'.format(
         time_elapsed // 60, time_elapsed % 60)   
     b_acc = 'Best val Acc: {:4f}'.format(best_acc)
             
     print(training_time)
     print(b_acc)

     with open(savetxt,'w') as f:
         f.writelines('\n'.join(result))
   
     return t_model


#-------------------------------------------checkpoint-------------------------------------------

def save_checkpoint(state,filename = 'model_best.pth.tar'):

    directory = "runs/%s/"%(args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename

    torch.save(state, filename)

def checkpoint_load(model):
    start_epoch = 0
    best_loss = 100
 
    if args.resume:
         if os.path.isfile(args.resume):
              
              print("=> loading checkpoint '{}'".format(args.resume))
              checkpoint = torch.load(args.resume)
              start_epoch = checkpoint['epoch']
              best_loss = checkpoint['best_loss']
              model.load_state_dict(checkpoint['state_dict'])
              print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

    return start_epoch, best_loss, model



if __name__ == '__main__':

    model = models.vgg16_bn(pretrained = True)
    model.classifier[-1] =  nn.Linear(4096, 106)
    
    start_epoch, best_loss, model = checkpoint_load(model)

    #freeze
    for i,p in enumerate(model.features.parameters()):
         #fc7から学習
         if i < 37:
              p.requires_grad = False 
     

    t_model  = VGGTriplet(model)
    t_model  = t_model.to(device)

    #class_net to t_net no class 
    optimizer = optim.SGD(t_model.parameters(), lr=0.001, momentum=0.9)
    t_model = model_train(t_model, optimizer, args.epochs, start_epoch, best_loss)
 
