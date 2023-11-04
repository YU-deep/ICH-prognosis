import numpy
import torch
import torch.utils.data
import SimpleITK as sitk
import re
from Utils.Text_Buider import get_patient_text,text_to_tensor


class ich_dataset(torch.utils.data.Dataset):
    """
    the dataset for Nets,
    tips: all the methods for pretreatment of Data are designed for our private dataset exclusively
    YOU NEED TO DESIGN ACCORDING PRETREATMENT METHOD FOR YOUR DATASET !!!
    """
    def __init__(self, data_path: str = None):
        self.data_list = []
        self.list = open(data_path, 'r')
        for item in self.list:
            item_list = item.split(" ")
            path='./Data/'+item_list[0]
            text = get_patient_text(path)
            label = int(item_list[1])
            self.data_list.append([path, text, label])


    def __getitem__(self, index):
        # calculate by all Data(after preprocess)
        # mean (2.4908)
        # std  (6.2173)
        # already normalized before this step
        # vision
        vision = sitk.ReadImage(self.data_list[index][0])
        vision = sitk.GetArrayFromImage(vision)
        # for i in range(32):
        #     for j in range(512):
        #         for k in range(512):
        #             nii[i][j][k]=(nii[i][j][k]-2.4908)/6.2173
        vision = torch.Tensor(vision)
        vision_tensor = torch.unsqueeze(vision, 0)

        text_tensor = text_to_tensor(self.data_list[index][1])

        label = self.data_list[index][2]
        label_tensor = torch.from_numpy(numpy.array(label)).long()

        return vision_tensor, text_tensor, label_tensor # self.data_list[index][0], self.data_list[index][1], label_tensor#  #

    def __len__(self):
        return len(self.data_list)




