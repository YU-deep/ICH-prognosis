import os
import sys
import time
import torchmetrics
import numpy
import torch
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from Nets.Loss_Function import *
from sklearn.model_selection import KFold

current_time = "{0:%Y-%m-%d_%H-%M-%S}".format(datetime.now())


def delta(a, b, tau=1e-8):
    a = a.permute(2, 1, 0)
    return torch.bmm(a, b) / tau

def get_bar_delta(bar):
    t2t, v2v, t2v, v2t = 0.0, 0.0, 0.0, 0.0
    for i, (vision, text, label) in enumerate(bar):
        t2t += delta(text, text)
        v2v += delta(vision, v2v)
        t2v += delta(text, vision)
        v2t += delta(vision, text)
    return t2t, v2v, t2v, v2t

def run_net(vision_encoder, text_encoder, cross_attn, self_attn, text_trt, vision_trt, classify, device, epochs, data_loader,
            optimizer, loss_function_id, fusion_structure_id, batch_size):
    """
        This is the function of training module
    """

    if not os.path.exists("./logs/ich"):
        os.mkdir("./logs/ich")
    os.mkdir("./logs/ich/" + current_time)
    train_summary = SummaryWriter(log_dir="./logs/ich/" + current_time + "/summary")
    log = open("./logs/ich/" + current_time + "/training_log.txt", 'w')

    test_acc = torchmetrics.Accuracy()
    test_recall = torchmetrics.Recall(average='none', num_classes=2)
    test_precision = torchmetrics.Precision(average='none', num_classes=2)
    test_auc = torchmetrics.AUROC(average="macro", num_classes=2)

    vision_encoder = vision_encoder.to(device)
    text_encoder = text_encoder.to(device)
    self_attn = self_attn.to(device)
    cross_attn = cross_attn.to(device)
    vision_trt = vision_trt.to(device)
    text_trt = text_trt.to(device)
    classify = classify.to(device)


    print("start training")
    for epoch in range(epochs):
        start_time = time.time()
        # train
        print("Epoch: {}/{}".format(epoch + 1, epochs))
        vision_encoder.train()
        text_encoder.train()
        self_attn.train()
        cross_attn.train()
        vision_trt.train()
        text_trt.trian()
        classify.train()


        t2t, v2v, t2v,v2t = get_bar_delta(data_loader)
        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        for train_index, val_index in kf.split(data_loader):
            train_fold = torch.utils.data.dataset.Subset(data_loader, train_index)
            val_fold = torch.utils.data.dataset.Subset(data_loader, val_index)
            train_loader = torch.utils.data.DataLoader(dataset=train_fold, batch_size=batch_size, shuffle=True)
            val_loader = torch.utils.data.DataLoader(dataset=val_fold, batch_size=batch_size, shuffle=True)
            train_bar = tqdm(train_loader, leave=True, file=sys.stdout)
            for i, ( vision , text, label) in enumerate(train_bar):
                vision = vision.to(device)
                text = text.to(device)
                label = label.to(device)

                optimizer.zero_grad()

                vision_feature = vision_encoder(vision)
                text_feature = text_encoder(text)

                text_feature = text_trt(text_feature)
                vision_feature = vision_trt(vision_feature)

                if fusion_structure_id == 0:
                    #ours fusion structure
                    cross_feature, _, _ = cross_attn(text_feature, vision_feature)
                    feature = self_attn(cross_feature)

                elif fusion_structure_id == 1:
                    cross_feature = self_attn(torch.cat(text_feature, vision_feature))
                    feature = self_attn(cross_feature)

                elif fusion_structure_id == 2:
                    _, text_feature, vision_feature = cross_attn(text_feature, vision_feature)
                    feature, _, _ = cross_attn(text_feature, vision_feature)

                elif fusion_structure_id == 3:
                    text_feature = self_attn(text_feature)
                    vision_feature = self_attn(vision_feature)
                    feature, _, _ = cross_attn(text_feature, vision_feature)


                outputs = classify(feature)

                if loss_function_id == 0:
                    # ours VTMF Loss
                    IMIMA = Intra_Modality_and_Inter_Modality_Alignment_Loss(t2t, v2v, t2v,v2t)
                    imima_loss = IMIMA(vision_feature, text_feature)
                    SDM = Similarity_Distribution_Matching_Loss(len(outputs))
                    sdm_loss = SDM(vision_feature, text_feature, label)
                    MLM = Masked_Language_Modeling_Loss()
                    mlm_loss = MLM(text_feature)
                    loss = imima_loss + sdm_loss + mlm_loss

                elif loss_function_id == 1:
                    loss = Cross_Entropy_Loss(outputs , label)

                elif loss_function_id == 2:
                    Blend = blend_Loss(len(outputs))
                    loss = Blend(outputs, label)

                elif loss_function_id == 3:
                    loss = SM_Loss(text_feature, vision_feature, label)

                elif loss_function_id == 4:
                    loss = cmpm_Loss(text_feature, vision_feature, label)

                elif loss_function_id == 5:
                    IMIMA = Intra_Modality_and_Inter_Modality_Alignment_Loss(t2t, v2v, t2v,v2t)
                    loss = IMIMA(vision_feature, text_feature)

                elif loss_function_id == 6:
                    SDM = Similarity_Distribution_Matching_Loss(len(outputs))
                    loss = SDM(vision_feature, text_feature, label)

                elif loss_function_id == 7:
                    MLM = Masked_Language_Modeling_Loss()
                    loss = MLM(text_feature)

                elif loss_function_id == 8:
                    IMIMA = Intra_Modality_and_Inter_Modality_Alignment_Loss(t2t, v2v, t2v, v2t)
                    imima_loss = IMIMA(vision_feature, text_feature)
                    SDM = Similarity_Distribution_Matching_Loss(len(outputs))
                    sdm_loss = SDM(vision_feature, text_feature, label)
                    loss = imima_loss + sdm_loss

                elif loss_function_id == 9:
                    IMIMA = Intra_Modality_and_Inter_Modality_Alignment_Loss(t2t, v2v, t2v, v2t)
                    imima_loss = IMIMA(vision_feature, text_feature)
                    MLM = Masked_Language_Modeling_Loss()
                    mlm_loss = MLM(text_feature)
                    loss = imima_loss + mlm_loss

                elif loss_function_id == 10:
                    CMAF = CMAF_Loss()
                    loss = CMAF(text_feature, vision_feature, label, outputs)

                loss.backward()

                optimizer.step()


            with torch.no_grad():
                vision_encoder.eval()
                text_encoder.eval()
                vision_trt.eval()
                text_trt.eval()
                self_attn.eval()
                cross_attn.eval()
                classify.eval()

                val_bar = tqdm(val_loader, leave=True, file=sys.stdout)
                for i, (vision, text, label) in enumerate(val_bar):
                    vision = vision.to(device)
                    text = text.to(device)
                    label = label.to(device)

                    vision_feature = vision_encoder(vision)
                    text_feature = text_encoder(text)

                    text_feature = text_trt(text_feature)
                    vision_feature = vision_trt(vision_feature)

                    if fusion_structure_id == 0:
                        # ours fusion structure
                        cross_feature, _, _ = cross_attn(text_feature, vision_feature)
                        feature = self_attn(cross_feature)

                    elif fusion_structure_id == 1:
                        cross_feature = self_attn(torch.cat(text_feature, vision_feature))
                        feature = self_attn(cross_feature)

                    elif fusion_structure_id == 2:
                        _, text_feature, vision_feature = cross_attn(text_feature, vision_feature)
                        feature, _, _ = cross_attn(text_feature, vision_feature)

                    elif fusion_structure_id == 3:
                        text_feature = self_attn(text_feature)
                        vision_feature = self_attn(vision_feature)
                        feature, _, _ = cross_attn(text_feature, vision_feature)

                    outputs = classify(feature)

                    predictions = torch.max(outputs.data, dim=1)
                test_acc.update(predictions.argmax(1), label)
                test_auc.update(predictions, label)
                test_recall.update(predictions.argmax(1), label)
                test_precision.update(predictions.argmax(1), label)

        total_acc = test_acc.compute()
        total_recall = test_recall.compute()
        total_precision = test_precision.compute()
        total_auc = test_auc.compute()

        train_summary.add_scalar("val_acc", total_acc, epoch)
        train_summary.add_scalar("val_recall", total_recall, epoch)
        train_summary.add_scalar("val_precision", total_precision, epoch)
        train_summary.add_scalar("val_auc", total_auc, epoch)

        if total_precision >= best_accuracy:
            best_accuracy = total_acc
            best_epoch = epoch
            torch.save(vision_encoder, "./logs/ich/" + current_time + "/vision_encoder.pth")
            torch.save(text_encoder, "./logs/ich/" + current_time + "/text_encoder.pth")
            torch.save(vision_trt, "./logs/ich/" + current_time + "/vision_trt.pth")
            torch.save(text_trt, "./logs/ich/" + current_time + "/text_trt.pth")
            torch.save(classify, "./logs/ich/" + current_time + "/classifier.pth")
            torch.save(self_attn, "./logs/ich/" + current_time + "/self_attention.pth")
            torch.save(cross_attn, "./logs/ich/" + current_time + "/cross_attention.pth")
            log_info = "-------\n" + "Epoch %d:\n" % (epoch + 1) + "Training" \
                   + "Val Accuracy: %4.2f%%  || " % (total_acc * 100) + \
                   "using time: %4.2f s\n" % (time.time() - start_time) + \
                   "best accuracy : %4.2f%%" % (best_accuracy * 100) \
                   + " produced @epoch %3d\n" % (best_epoch + 1)
            log.write(log_info)
            print(log_info)
        test_precision.reset()
        test_acc.reset()
        test_recall.reset()
        test_auc.reset()



