import torch
import torch.nn as nn
import SimpleITK as sitk
import re
from Utils.Text_Buider import text_to_tensor,get_patient_text


class Our_Net(nn.Module):
    """"
    Please palace the .pth files in according location for loading
    """
    def __init__(self):
        super(Our_Net, self).__init__()
        self.text_encoder = torch.load("trained_net/best/text_encoder.pth")
        self.vision_encoder = torch.load("trained_net/best/vision_encoder.pth")
        self.text_trt = torch.load("trained_net/best/text_trt.pth")
        self.vision_trt = torch.load("trained_net/best/vision_trt.pth")
        self.self_attn = torch.load("trained_net/best/self_attention.pth")
        self.cross_attn = torch.load("trained_net/best/cross_attention.pth")
        self.classifier = torch.load("trained_net/best/classifier.pth")
        print("Net is available!")

    def forward(self ,vision, text):
        vision_feature = self.vision_encoder(vision)
        text_feature = self.text_encoder(text)

        vision_feature = self.vision_trt(vision_feature)
        text_feature = self.text_trt(text_feature)
        cross_feature = self.cross_attn(text_feature, vision_feature)
        feature = self.self_attn(cross_feature)

        output = self.classifier(feature)
        return output




