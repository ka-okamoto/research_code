from PIL import Image
import os
import os.path

import torch.utils.data
import torchvision.transforms as transforms

def default_image_loader(path):
    return Image.open(path).convert('RGB')


class TxtReadData(torch.utils.data.Dataset):

    def __init__(self,readtxt,transform = None,loader = default_image_loader):
        """
        dirpath : directory address have file or sub directory
        readtxt : txt file writen file path
        """
                
        self.filenamelist = self._makelist(readtxt)
        self.transform = transform
        self.loader = loader
        self.class_to_idx = self._find_classes()
        
    def _makelist(self,readtxt):
        with open(readtxt, 'r') as f :
            data = f.read()
            
        datalist = data.split('\n')
        datalist.pop()
    
        return datalist


    def _find_classes(self):

        #カテゴリリストtxtはカテゴリごとのディレクトリのパスです。
        "/dir/class1/
         /dir/class2/
         .
         .
         .
        " 
      
        Class_Serect_List_Path = 'categorylist.txt'
        with open(Class_Serect_List_Path) as f:
            datas= f.read()
            
        directorys = datas.split('\n')
        directorys.pop()

        class_to_idx = {}
        
        for idx,directory in enumerate(directorys):
            category = directory.split('/')[-1]
            class_to_idx[category] = idx

        return class_to_idx

    def __len__(self):
        return len(self.filenamelist)


    def __getitem__(self,index):
        filepath = self.filenamelist[index]

        img = self.loader(filepath)
        if self.transform is not None:
            img = self.transform(img)
        
        return img
