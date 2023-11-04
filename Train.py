import argparse

import torch.utils.data

from Data.dataset import ich_dataset
from Nets.Api import run_net
from Nets.Joint_Attention_Fusion import Cross_Modal_Attention_Fusion, Multi_Head_Self_Attention_Fusion,Text_Representation_Transformation,Vision_treatment_Net
from Nets.Vision_Encoder import get_pretrained_Vision_Encoder,get_Vision_Encoder
from Nets.Text_Encoder import get_Text_Encoder
from Nets.Classification_Header import get_Classification_Header,get_pretrained_Classification_Header



def prepare_to_train_net(batch_size, epochs, learning_rate, use_gpu, gpu_id, vision_pre, classification_pre,
                         loss_function_id, fusion_structure_id , parameters = [512, 256, 256, 256]):
    torch.cuda.empty_cache()
    text_encoder = get_Text_Encoder()
    if vision_pre == True:
        vision_encoder = get_pretrained_Vision_Encoder()
    else:
        vision_encoder = get_Vision_Encoder()
    text_trt = Text_Representation_Transformation(64,128,32,32,3)
    vision_trt = Vision_treatment_Net()
    cross_atten = Cross_Modal_Attention_Fusion(parameters[0] , parameters[1], 1 ,256,[256,256,256])
    self_atten = Multi_Head_Self_Attention_Fusion(16, parameters[3], parameters[4], 0)
    if classification_pre ==True:
        classify_head = get_pretrained_Classification_Header()
    else:
        classify_head = get_Classification_Header()

    params =[p for p in text_encoder.parameters() if p.requires_grad]+[p for p in vision_encoder.parameters() if p.requires_grad]+\
    [p for p in text_trt.parameters() if p.requires_grad]+[p for p in vision_trt.parameters() if p.requires_grad]+[p for p in cross_atten.parameters() if p.requires_grad]+[p for p in self_atten.parameters() if p.requires_grad]+\
    [p for p in classify_head.parameters() if p.requires_grad]
    text_encoder_params = [p for p in text_encoder.parameters() if p.requires_grad]
    print("step1: " + str(len(text_encoder_params)))
    vision_encoder_params = [p for p in vision_encoder.parameters() if p.requires_grad]
    print("step1: " + str(len(vision_encoder_params)))

    text_trt_params = [p for p in text_trt.parameters() if p.requires_grad]
    print("step2: " + str(len(text_trt_params)))
    vision_trt_params = [p for p in vision_trt.parameters() if p.requires_grad]
    print("step2: " + str(len(vision_trt_params)))
    cross_atten_params = [p for p in cross_atten.parameters() if p.requires_grad]
    print("step2: " + str(len(cross_atten_params)))
    self_atten_params = [p for p in self_atten.parameters() if p.requires_grad]
    print("step2: " + str(len(self_atten_params)))

    classify_params = [p for p in classify_head.parameters() if p.requires_grad]
    print("step3: " + str(len(classify_params)))


    data_set = ich_dataset("Data/all.txt")

    data_loader = torch.utils.data.DataLoader(dataset=data_set, batch_size=batch_size, shuffle=True)
    print("datasets loaded, dataset_sum: %d" % (len(data_set)))

    optimizer = torch.optim.Adam(params, lr=learning_rate)

    if use_gpu:
        device = torch.device(('cuda:' + gpu_id) if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    print("use device {}".format(device))
    print("prepare completed! launch training!\U0001F680")
    print("Use full Nets to classify")
    run_net(vision_encoder=vision_encoder, text_encoder=text_encoder, text_trt=text_trt, vision_trt=vision_trt,
            self_attn=self_atten, cross_attn=cross_atten, classify=classify_head, device=device, epochs=epochs,
            data_loader=data_loader, optimizer=optimizer, loss_function_id=loss_function_id,
            fusion_structure_id=fusion_structure_id, batch_size=batch_size)



if __name__ == "__main__":
    # Adding necessary input arguments
    parser = argparse.ArgumentParser(description="add arguments to test")
    parser.add_argument("--batch_size", default=4, help="the batch_size of training")
    parser.add_argument("--epochs", default=300, help="the epochs of training", type=int)
    parser.add_argument("--use_gpu", default=True, help="device choice, if cuda isn't available program will warn",
                        type=bool)
    parser.add_argument("--learning_rate", default=0.0001, help="learning_rate", type=float)
    parser.add_argument("--gpu_id", default="0", type=str, help="the gpu id")
    parser.add_argument("--classification_pre", help="whether use pre-trained classification header", action='store_true')
    parser.add_argument("--vision_pre", help="whether use pre-trained vision encoder", action='store_true')
    parser.add_argument("--loss_function_id", help="the ID of loss function you use", default=0, type=int)
    parser.add_argument("--fusion_structure_id", help="the ID of fusion structure you use", default=0, type=int)
    args = parser.parse_args()

    print(args)
    prepare_to_train_net(batch_size=int(args.batch_size), epochs=args.epochs, use_gpu=args.use_gpu,
                                 learning_rate=args.learning_rate, gpu_id=args.gpu_id, vision_pre=args.vision_pre,
                                 classification_pre=args.classification_pre, loss_function_id=int(args.loss_function_id),
                                 fusion_structure_id=int(args.fusion_structure_id))