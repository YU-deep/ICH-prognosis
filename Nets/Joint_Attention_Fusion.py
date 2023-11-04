import torch
import torch.nn as nn
import math
from Nets.Vision_Encoder import conv1x1x1,conv3x3x3
from torch.autograd import Function
import torch.nn.functional as F
from torch.nn.modules.utils import _triple, _pair, _single
from SoftPool import soft_pool3d, SoftPool3d
import softpool_cuda



class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2))
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Vision_treatment_Net(nn.Module):
    def __init__(self, input_size, output_size, middle_size):
        super(Vision_treatment_Net, self).__init__()
        self.fc = nn.Linear(input_size, middle_size[0])
        self.up1 = Up(middle_size[0], middle_size[1])
        self.up2 = Up(middle_size[1], middle_size[2])
        self.up3 = Up(middle_size[2], middle_size[3])
        self.up4 = Up(middle_size[3], middle_size[4])

    def forward(self, x):
        x = self.fc(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        return x


class Text_Representation_Transformation(nn.Module):
    def __init__(self, input_size, output_size, height, width, num_channels):
        """
        Text_Representation_Transformation (TRT) Module
        """
        super(Text_Representation_Transformation, self).__init__()
        self.height = height
        self.width = width
        self.num_channels = num_channels
        self.linear1 = nn.Linear(input_size, output_size)

    def reshape_2d_to_3d(self, x):
        batch_size, num_features = x.shape
        """ The input Data does not match the target dimension """
        assert num_features == self.num_channels * self.height * self.width
        reshaped_data = x.view(batch_size, self.num_channels, self.height, self.width)
        return reshaped_data

    def forward(self, x):
        x_T = x.permute(2,1,0)
        x = torch.bmm(x, x_T)
        x = self.linear1(x)
        x = self.reshape_2d_to_3d(x)
        return x


class Multi_Head_Self_Attention_Fusion(nn.Module):
    def __init__(self, num_attention_heads, input_size, hidden_size, hidden_dropout_prob, output_size):
        """
        Multi_Head_Self_Attention_Fusion (MHSAF) Module and a 3×3×3 Conv
        """
        super(Multi_Head_Self_Attention_Fusion, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size

        self.query = nn.Linear(input_size, self.all_head_size)
        self.key = nn.Linear(input_size, self.all_head_size)
        self.value = nn.Linear(input_size, self.all_head_size)

        self.attn_dropout = nn.Dropout()

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)
        self.conv = conv3x3x3(hidden_size, output_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return self.conv(hidden_states)


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """
        Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class Cross_Modal_Attention_Fusion(nn.Module):
    expansion = 1
    C_hat = 4

    def __init__(self, input_size, hidden_size, planes, middle_size ,size):
        """
              Cross_Modal_Attention_Fusion (CMAF) Module
        """
        super(Cross_Modal_Attention_Fusion, self).__init__()
        self.text_fc1 = nn.Linear(input_size, hidden_size)
        self.vision_fc1 = nn.Linear(input_size, hidden_size)

        self.text_query = conv1x1x1(planes, self.expansion)
        self.text_key = conv1x1x1(planes, self.expansion)
        self.text_value = conv1x1x1(planes, self.expansion)

        self.vision_query = conv1x1x1(planes, self.expansion)
        self.vision_key = conv1x1x1(planes, self.expansion)
        self.vision_value = conv1x1x1(planes, self.expansion)

        self.text_fc2 = nn.Linear(hidden_size, middle_size)
        self.vision_fc2 = nn.Linear(hidden_size, middle_size)

        self.softpool_x1 = SoftPool3d(kernel_size=(self.C_hat, 2, 2), stride=(2, 2, 2))
        self.softpool_x2 = SoftPool3d(kernel_size=(self.C_hat, 2, 2), stride=(2, 2, 2))
        self.softpool_y1 = SoftPool3d(kernel_size=(self.C_hat, 2, 2), stride=(2, 2, 2))
        self.softpool_y2 = SoftPool3d(kernel_size=(self.C_hat, 2, 2), stride=(2, 2, 2))

        self.softmax1 = nn.Softmax(dim=-1)
        self.softmax2 = nn.Softmax(dim=-1)

        self.w_t = nn.Parameter(torch.Tensor(size[0],size[1],size[2]))
        self.w_v = nn.Parameter(torch.Tensor(size[0],size[1],size[2]))
        self.w = nn.Parameter(torch.ones(2))
    def get_vt(self, a, b):
        o = torch.bmm(a, b)
        o = torch.exp(o)
        sum_o = torch.sum(o)
        return o / sum_o

    def forward(self, x, y):
        w_t_query = torch.exp(self.w_t[0]) / torch.sum(torch.exp(self.w_t))
        w_t_key = torch.exp(self.w_t[1]) / torch.sum(torch.exp(self.w_t))
        w_t_value = torch.exp(self.w_t[2]) / torch.sum(torch.exp(self.w_t))

        w_v_query = torch.exp(self.w_v[0]) / torch.sum(torch.exp(self.w_v))
        w_v_key = torch.exp(self.w_v[1]) / torch.sum(torch.exp(self.w_v))
        w_v_value = torch.exp(self.w_v[2]) / torch.sum(torch.exp(self.w_v))

        w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))

        raw_x = x.clone()

        x_softpool = self.softpool_x2(x)
        y_softpool = self.softpool_x2(y)

        x = self.text_fc1(x)
        y = self.vision_fc1(y)

        query_t = self.text_query(x)
        key_t = self.text_key(x)
        value_t = self.text_value(x)

        query_v = self.vision_query(y)
        key_v = self.vision_key(y)
        value_v = self.vision_value(y)

        query_t = query_t.permute(2,1,0)
        query_v = query_v.permute(2,1,0)

        k_v_q_t = self.get_vt(w_v_key * key_v, w_t_query * query_t)
        k_t_q_v = self.get_vt(w_t_key * key_t, w_v_query * query_v)

        ox = torch.bmm(w_v_value * value_v, k_v_q_t)
        oy = torch.bmm(w_t_value * value_t, k_t_q_v)

        ox = self.softpool_x1(ox)
        oy = self.softpool_y1(oy)

        x_hat = ox + x_softpool
        y_hat = oy + y_softpool

        x_hat = self.vision_fc2(x_hat)
        y_hat = self.text_fc2(y_hat)

        x_hat = x_hat + raw_x

        x_hat = self.softmax1(x_hat)
        y_hat = self.softmax2(y_hat)

        feature = torch.cat(w1 * x_hat, w2 * y_hat)
        # return feature  for the fusion structure experiment
        return feature, x_hat, y_hat



class CUDA_SOFTPOOL3d(Function):
    """
    Code of SoftPool
    This section of SoftPool is referenced from:
        @inproceedings
        {
            stergiou2021refining,
            title={Refining activation downsampling with SoftPool},
            author={Stergiou, Alexandros, Poppe, Ronald and Kalliatakis Grigorios},
            booktitle={International Conference on Computer Vision (ICCV)},
            year={2021},
            pages={10357-10366},
            organization={IEEE}
        }
    """

    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, kernel=2, stride=None):
        # Create contiguous tensor (if tensor is not contiguous)
        no_batch = False
        if len(input.size()) == 3:
            no_batch = True
            input.unsqueeze_(0)
        B, C, D, H, W = input.size()
        kernel = _triple(kernel)
        if stride is None:
            stride = kernel
        else:
            stride = _triple(stride)
        oD = (D - kernel[0]) // stride[0] + 1
        oH = (H - kernel[1]) // stride[1] + 1
        oW = (W - kernel[2]) // stride[2] + 1
        output = input.new_zeros((B, C, oD, oH, oW))
        softpool_cuda.forward_3d(input.contiguous(), kernel, stride, output)
        ctx.save_for_backward(input)
        ctx.kernel = kernel
        ctx.stride = stride
        if no_batch:
            return output.squeeze_(0)
        return output

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        # Create contiguous tensor (if tensor is not contiguous)
        grad_input = torch.zeros_like(ctx.saved_tensors[0])
        saved = [grad_output.contiguous()] + list(ctx.saved_tensors) + [ctx.kernel, ctx.stride] + [grad_input]
        softpool_cuda.backward_3d(*saved)
        # Gradient underflow
        saved[-1][torch.isnan(saved[-1])] = 0
        return saved[-1], None, None
