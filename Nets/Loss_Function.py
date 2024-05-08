import torch
import torch.nn as nn
import torch.nn.functional as F


class Intra_Modality_and_Inter_Modality_Alignment_Loss(nn.Module):
    """
    Intra Modality and Inter Modality Alignment Loss (IMIMA) Loss,
    """

    def __init__(self, t2t, v2v, t2v, v2t):
        super(Intra_Modality_and_Inter_Modality_Alignment_Loss, self).__init__()
        self.t2t = t2t
        self.v2v = v2v
        self.t2v = t2v
        self.v2t = v2t
        self.w = nn.Parameter(torch.ones(2))

    def delta(self, a, b, tau):
        a = a.permute(2, 1, 0)
        return torch.bmm(a, b) / tau

    def forward(self, vision_features, text_features):
        w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
        f_t2t = self.delta(text_features, text_features)
        f_v2v = self.delta(vision_features, vision_features)
        f_t2v = self.delta(text_features, vision_features)
        f_v2t = self.delta(vision_features, text_features)

        t2t_loss = - torch.log(f_t2t / (f_t2t + w1 * self.t2t))
        v2v_loss = - torch.log(f_v2v / (f_v2v + w1 * self.v2v))
        t2v_loss = - torch.log(f_t2v / (f_t2v + w2 * self.t2v))
        v2t_loss = - torch.log(f_v2t / (f_v2t + w2 * self.v2t))

        return t2t_loss + v2v_loss + t2v_loss + v2t_loss


class Similarity_Distribution_Matching_Loss(nn.Module):
    """
    Similarity Distribution Matching (SDM) Loss,
    Adapted from: https://github.com/anosorae/IRRA
    """

    def __init__(self, length):
        super(Similarity_Distribution_Matching_Loss, self).__init__()
        self.length = length

    def forward(self, vision_fetures, text_fetures, labels, epsilon=1e-8):
        logit_scale = self.length
        labels = labels - labels.t()
        labels = (labels == 0).float()

        image_norm = vision_fetures / vision_fetures.norm(dim=1, keepdim=True)
        text_norm = text_fetures / text_fetures.norm(dim=1, keepdim=True)

        t2i_cosine_theta = text_norm @ image_norm.t()
        i2t_cosine_theta = t2i_cosine_theta.t()

        text_proj_image = logit_scale * t2i_cosine_theta
        vision_proj_text = logit_scale * i2t_cosine_theta

        # normalize the true matching distribution
        labels_distribute = labels / labels.sum(dim=1)

        i2t_pred = F.softmax(vision_proj_text, dim=1)
        i2t_loss = i2t_pred * (F.log_softmax(vision_proj_text, dim=1) - torch.log(labels_distribute + epsilon))
        t2i_pred = F.softmax(text_proj_image, dim=1)
        t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_distribute + epsilon))

        loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

        return loss


class Masked_Language_Modeling_Loss(nn.Module):
    """
    Masked Language Modeling (MLM) Loss
    """

    def __init__(self):
        super(Masked_Language_Modeling_Loss, self).__init__()
        self.criterion = nn.NLLLoss(ignore_index=0)

    def forward(self, datas):
        loss = 0.0
        for i in range(datas):
            next_sent_output, mask_lm_output = torch.eq(datas[i + 1], datas[i])
            next_loss = self.criterion(next_sent_output, datas[i + 1])
            mask_loss = self.criterion(mask_lm_output.transpose(1, 2), datas[i])
            loss += (next_loss + mask_loss)
        return loss


def Cross_Entropy_Loss(scores, labels):
    """
    Cross Entropy Loss
    :param scores:
    :param labels:
    :return:
    """

    ce = nn.CrossEntropyLoss(ignore_index=0)
    return ce(scores, labels)


class blend_Loss(nn.Module):
    """
    blend Loss
    :param scores:
    :param labels:
    :return:
    """

    def __init__(self, length):
        super(blend_Loss, self).__init__()
        self.weight = nn.Parameter(torch.ones(length))

    def forward(self, scores, labels):
        ce = nn.CrossEntropyLoss(ignore_index=0)
        return self.weight * ce(scores, labels)


def SM_Loss(text_embeddings, vision_embeddings, labels):
    """
    SM_Loss
    :param text_embeddings:
    :param vision_embeddings:
    :param labels:
    :return:
    """

    text_probs = torch.nn.functional.log_softmax(text_embeddings, dim=1)
    text_loss = -(labels * text_probs).sum() / text_embeddings.shape[0]

    vision_loss = nn.functional.binary_cross_entropy_with_logits(vision_embeddings, labels)
    return (text_loss + vision_loss) / 2



def cmpm_Loss(image_embeddings, text_embeddings, labels, epsilon=1e-8):
    """
    Cross-Modal Projection Matching Loss(CMPM)
    :param image_embeddings: Tensor with dtype torch.float32
    :param text_embeddings: Tensor with dtype torch.float32
    :param labels: Tensor with dtype torch.int32
    :return:
        i2t_loss: cmpm loss for image projected to text
        t2i_loss: cmpm loss for text projected to image
        pos_avg_sim: average cosine-similarity for positive pairs
        neg_avg_sim: averate cosine-similarity for negative pairs
    """

    batch_size = image_embeddings.shape[0]
    labels_reshape = torch.reshape(labels, (batch_size, 1))
    labels_dist = labels_reshape - labels_reshape.t()
    labels_mask = (labels_dist == 0).float()

    image_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
    text_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
    image_proj_text = torch.matmul(image_embeddings, text_norm.t())
    text_proj_image = torch.matmul(text_embeddings, image_norm.t())

    # normalize the true matching distribution
    labels_mask_norm = labels_mask / labels_mask.norm(dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_mask_norm + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_mask_norm + epsilon))

    cmpm_loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    return cmpm_loss





def Triplet_Loss(inputs, labels):
    """
    a single component loss of CMAF Loss 
    :param inputs: 
    :param labels: 
    :return: 
    """""
    n = inputs.size(0)  # batch_size

    # Compute pairwise distance, replace by the official when merged
    dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
    dist = dist + dist.t()
    dist.addmm_(1, -2, inputs, inputs.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

    # For each anchor, find the hardest positive and negative
    mask = labels.expand(n, n).eq(labels.expand(n, n).t())
    dist_ap, dist_an = [], []
    for i in range(n):
        dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
        dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
    dist_ap = torch.cat(dist_ap)
    dist_an = torch.cat(dist_an)

    # Compute ranking hinge loss
    y = torch.ones_like(dist_an)
    ranking_loss = nn.MarginRankingLoss()
    loss = ranking_loss(dist_an, dist_ap, y)
    return loss


def Cross_Modal_Affinity_Loss(text_features, vision_features, labels, alpha=0.01):
    """
    a single component loss of CMAF Loss
    :param text_features:
    :param vision_features:
    :param labels:
    :param alpha:
    :return:
    """
    text_features_plus = []
    text_features_sub = []
    vision_features_plus = []
    vision_features_sub = []

    for i in range(labels):
        if labels[i] == 0:
            text_features_sub.append(text_features[i])
            vision_features_sub.append(vision_features[i])
        else:
            text_features_plus.append(text_features[i])
            vision_features_plus.append(vision_features[i])

    image_norm_plus = vision_features_plus / vision_features_plus.norm(dim=1, keepdim=True)
    text_norm_plus = text_features_plus / text_features_plus.norm(dim=1, keepdim=True)
    image_norm_sub = vision_features_sub / vision_features_sub.norm(dim=1, keepdim=True)
    text_norm_sub = text_features_sub / text_features_sub.norm(dim=1, keepdim=True)

    t2i_cosine_theta_plus = text_norm_plus @ image_norm_plus.t()
    i2t_cosine_theta_plus = t2i_cosine_theta_plus.t()
    t2i_cosine_theta_sub = text_norm_sub @ image_norm_sub.t()
    i2t_cosine_theta_sub = t2i_cosine_theta_sub.t()
    return max(0, alpha - t2i_cosine_theta_plus - i2t_cosine_theta_sub) + max(0,
                                                                              alpha - i2t_cosine_theta_plus - t2i_cosine_theta_sub)


class CMAF_Loss(nn.Module):
    """
    CMAF Loss
    :param text_features:
    :param vision_features:
    :param labels:
    :param scores:
    :return:
    """

    def __init__(self):
        super(CMAF_Loss, self).__init__()
        self.w = nn.Parameter(torch.ones(3))

    def forward(self, text_features, vision_features, labels, scores):
        w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
        w3 = torch.exp(self.w[2]) / torch.sum(torch.exp(self.w))
        cross_entropy_loss = Cross_Entropy_Loss(scores=scores, labels=labels)
        triplet_loss = Triplet_Loss(text_features, labels) + Triplet_Loss(vision_features, labels)
        cross_modal_affinity_loss = Cross_Modal_Affinity_Loss(text_features=text_features,
                                                              vision_features=vision_features, labels=labels)
        return w1 * cross_entropy_loss + w2 * triplet_loss + w3 * cross_modal_affinity_loss
