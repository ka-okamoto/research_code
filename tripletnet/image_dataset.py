from PIL import Image
import os
import os.path

import torch.utils.data
import torchvision.transforms as transforms

def default_image_loader(path):
    return Image.open(path).convert('RGB')


class TripletCategory(torch.utils.data.Dataset):
    def __init__(self, filenames_txt, transform=None,
                 loader=default_image_loader):
        

        self.category = []
        self.filenamelist = []
        self.labels = []
        self.class_to_idx = self._find_classes()

        for line in open(filenames_txt):
            line = line.rstrip('\n')
            if line == '':
                continue
            else:
                datas = line.split(',')
                p_path,p_label = self._label_name(datas[0])
                n_path,n_label = self._label_name(datas[2])
                self.filenamelist.append(datas)
                self.labels.append([p_label,n_label])

                
        self.loader = loader
        self.transform = transform
        
    def _label_name(self,line):

        label = line.split('/')[-2]
        return line , label


    def _find_classes(self):
        #カテゴリリストtxtはカテゴリごとのディレクトリのパスです。
        "/dir/class1/
         /dir/class2/
         .
         .
         .
        "
        Class_Serect_List_Path = 'categorylist.txt'
        
        nofood_num = 106

        with open(Class_Serect_List_Path) as f:
            datas= f.read()
            
        directorys = datas.split('\n')
        directorys.pop()

        class_to_idx = {}
        
        for idx,directory in enumerate(directorys):
            category = directory.split('/')[-1]
            class_to_idx[category] = idx

        class_to_idx['nonfood'] = nofood_num

        return class_to_idx

    
    def __getitem__(self, index):

        p_label = self.class_to_idx[self.labels[index][0]]
        n_label = self.class_to_idx[self.labels[index][1]]
        
        query = self.loader(self.filenamelist[index][0])        
        positive = self.loader(self.filenamelist[index][1])    
        negative = self.loader(self.filenamelist[index][2])

        if self.transform is not None:
            query = self.transform(query)
            positive = self.transform(positive)
            negative = self.transform(negative)


        return query,positive,negative,p_label,p_label,n_label

    def __len__(self):
        return len(self.filenamelist)

