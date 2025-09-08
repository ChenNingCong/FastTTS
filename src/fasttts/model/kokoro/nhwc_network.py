import torch
from torch.utils.cpp_extension import load
from torch import Tensor
from torch.nn.utils.parametrizations import weight_norm
from einops import rearrange
import torch.nn as nn
from .istftnet import init_weights, get_padding, weight_norm
from typing import List
import os

def init_nchw_network():
    global custom_conv_ext
    global cudnnDataType_t
    global cudnnMathType_t
    os.environ["CUDNN_LOGERR_DBG"] = "1"
    custom_conv_ext = load(
        name="custom_conv_ext",
        sources=[os.path.join(os.path.dirname(__file__),"convolution_kernel.cpp")],
        extra_cflags=['-O2'],
        extra_cuda_cflags=['-O2'],
        with_cuda=True,
        extra_ldflags=['-lcudnn'],
        is_python_module=True,
        verbose=True
    )

    cudnnDataType_t = custom_conv_ext.cudnnDataType_t
    cudnnMathType_t = custom_conv_ext.cudnnMathType_t
    

def my_custom_convolution2d_forward(
    input_tensor: torch.Tensor,
    weight_tensor: torch.Tensor,
    padding: List[int],
    stride: List[int],
    upscale: List[int],
    groups: int,
    allow_tf32: bool,
    dataType: 'cudnnDataType_t',
    mathType: 'cudnnMathType_t',
    outputDataType: 'cudnnDataType_t'
):
    return custom_conv_ext.my_custom_convolution2d_forward(
    input_tensor,
    weight_tensor,
    padding,
    stride,
    upscale,
    groups,
    allow_tf32,
    dataType,
    mathType,
    outputDataType
)

def my_custom_convolution2d_forward_with_bias(
    input_tensor: torch.Tensor,
    weight_tensor: torch.Tensor,
    bias_tensor : torch.Tensor,
    padding: List[int],
    stride: List[int],
    upscale: List[int],
    groups: int,
    allow_tf32: bool,
    dataType: 'cudnnDataType_t',
    mathType: 'cudnnMathType_t',
    outputDataType: 'cudnnDataType_t'
):
    return custom_conv_ext.my_custom_convolution2d_forward_with_bias(
    input_tensor,
    weight_tensor,
    bias_tensor,
    padding,
    stride,
    upscale,
    groups,
    allow_tf32,
    dataType,
    mathType,
    outputDataType
)

@torch.library.custom_op("custom_conv_ext::conv1d", mutates_args=())
def conv1d(
    input: Tensor,
    weight: Tensor,
    bias: Tensor,
    stride: int,
    padding: int,
    dilation: int,
) -> Tensor:
    input_tensor = input.unsqueeze(1)
    weight_tensor = weight.unsqueeze(1)
    bias_tensor = bias
    assert bias_tensor.dim() == 1
    assert input.is_contiguous()
    assert weight_tensor.is_contiguous()
    assert input.is_cuda
    assert weight.is_cuda
    if input.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16:
        dataType=cudnnDataType_t.CUDNN_DATA_FLOAT
        mathType=cudnnMathType_t.CUDNN_TENSOR_OP_MATH
        outputDataType=cudnnDataType_t.CUDNN_DATA_BFLOAT16
    elif input.dtype == torch.float32 and weight.dtype == torch.float32:
        dataType=cudnnDataType_t.CUDNN_DATA_FLOAT
        mathType=cudnnMathType_t.CUDNN_TENSOR_OP_MATH
        outputDataType=cudnnDataType_t.CUDNN_DATA_FLOAT
    else:
        assert False
    custom_output = my_custom_convolution2d_forward_with_bias(
        input_tensor,
        weight_tensor,
        bias_tensor,
        [0, padding],
        [1, stride],
        [1, dilation],
        groups=1,
        allow_tf32=True,
        dataType=dataType,
        mathType=mathType,
        outputDataType=outputDataType,
    )
    return custom_output.squeeze(1)

class NHWCConv1d(nn.Conv1d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int | tuple[int], stride: int | tuple[int] = 1, padding: str | int | tuple[int] = 0, dilation: int | tuple[int] = 1, groups: int = 1, bias: bool = True, padding_mode: str = "zeros", device=None, dtype=None) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
    def apply_fix(self):
        self.weight = nn.Parameter(rearrange(self.weight, "c_out c_in h->c_out h c_in").contiguous())
        assert len(self.kernel_size) == 1
    @torch.no_grad
    def forward(self, input: Tensor) -> Tensor:
        result = conv1d(input, self.weight, self.bias, self.stride[0], self.padding[0], self.dilation[0])
        return result


def instance_norm_channel_last(x, epsilon=1e-5):
    """
    Applies instance normalization to a channel-last input tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (N, H, W, C).
        epsilon (float): A small value added to the variance to avoid division by zero.

    Returns:
        torch.Tensor: The instance-normalized tensor.
    """
    # Check if the input tensor has the correct number of dimensions
    if x.dim() != 3 and x.dim() != 4:
        raise ValueError("Input tensor must be 3D (N, L, C) or 4D (N, H, W, C).")

    # Get the number of channels
    num_channels = x.size(-1)

    # Reshape the tensor to (N, H*W, C) to perform calculations more easily
    # This groups the spatial dimensions (H, W) for each batch and channel
    x_reshaped = x.view(x.size(0), -1, num_channels)

    # Calculate the mean and standard deviation along the spatial dimensions
    # The mean and std will have a shape of (N, 1, C)
    mean = x_reshaped.mean(dim=1, keepdim=True)
    std = x_reshaped.std(dim=1, keepdim=True)

    # Normalize the tensor using the calculated mean and std
    normalized_x = (x_reshaped - mean) / (std + epsilon)

    # Reshape the normalized tensor back to its original shape (N, H, W, C)
    if x.dim() == 4:
        normalized_x = normalized_x.view(x.size(0), x.size(1), x.size(2), num_channels)

    return normalized_x

class NHWCInstanceNorm1d(nn.InstanceNorm1d):
    def __init__(self, num_features: int, eps: float = 0.00001, momentum: float = 0.1, affine: bool = False, track_running_stats: bool = False, device=None, dtype=None) -> None:
        super().__init__(num_features, eps, momentum, affine, track_running_stats, device, dtype)
    def forward(self, x):
        return instance_norm_channel_last(x)
    
class NHWCAdaIN1d(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        # affine should be False, however there's a bug in the old torch.onnx.export (not newer dynamo) that causes the channel dimension to be lost if affine=False. When affine is true, there's additional learnably parameters. This shouldn't really matter setting it to True, since we're in inference mode
        self.norm = NHWCInstanceNorm1d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)
        # self.fc is splitted into two linear maps
        # self.fc will be deleted during runtime
        self.fc1 = nn.Linear(style_dim, num_features)
        self.fc2 = nn.Linear(style_dim, num_features)

    def forward(self, x, s):
        gamma = self.fc1(s).unsqueeze(1)
        beta = self.fc2(s).unsqueeze(1)
        return (1 + gamma) * self.norm(x) + beta


class NCHWAdaINResBlock1(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5), style_dim=64):
        super(NCHWAdaINResBlock1, self).__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(NHWCConv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                                  padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(NHWCConv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                                  padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(NHWCConv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                                  padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)
        self.convs2 = nn.ModuleList([
            weight_norm(NHWCConv1d(channels, channels, kernel_size, 1, dilation=1,
                                  padding=get_padding(kernel_size, 1))),
            weight_norm(NHWCConv1d(channels, channels, kernel_size, 1, dilation=1,
                                  padding=get_padding(kernel_size, 1))),
            weight_norm(NHWCConv1d(channels, channels, kernel_size, 1, dilation=1,
                                  padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)
        self.adain1 = nn.ModuleList([
            NHWCAdaIN1d(style_dim, channels),
            NHWCAdaIN1d(style_dim, channels),
            NHWCAdaIN1d(style_dim, channels),
        ])
        self.adain2 = nn.ModuleList([
            NHWCAdaIN1d(style_dim, channels),
            NHWCAdaIN1d(style_dim, channels),
            NHWCAdaIN1d(style_dim, channels),
        ])
        self.alpha1 = nn.ParameterList([nn.Parameter(torch.ones(1, channels, 1)) for i in range(len(self.convs1))])
        self.alpha2 = nn.ParameterList([nn.Parameter(torch.ones(1, channels, 1)) for i in range(len(self.convs2))])
    def apply_fix(self):
        for i, v in enumerate(self.alpha1):
            self.alpha1[i] = rearrange(v, "b c h -> b h c")
        for i, v in enumerate(self.alpha2):
            self.alpha2[i] = rearrange(v, "b c h -> b h c")
    @torch.no_grad
    def forward(self, x, s):
        return self.forward_helper(x, s)
    @torch.no_grad
    def forward_helper(self, x : torch.Tensor, s):
        # note the magic number here!
        x = rearrange(x, "b c h -> b h c").contiguous()
        for i in range(3):
            c1, c2, n1, n2, a1, a2 = self.convs1[i], self.convs2[i], self.adain1[i], self.adain2[i], self.alpha1[i], self.alpha2[i]
            xt = n1(x, s)
            xt = xt + (1 / a1) * (torch.sin(a1 * xt) ** 2)  # Snake1D
            xt = c1(xt)
            xt = n2(xt, s)
            xt = xt + (1 / a2) * (torch.sin(a2 * xt) ** 2)  # Snake1D
            xt = c2(xt)
            x = xt + x
        x = rearrange(x, "b h c -> b c h")
        return x

def fix_conv1d(m : nn.Module):
    if isinstance(m, NCHWAdaINResBlock1):
        m.apply_fix()
    if isinstance(m, NHWCConv1d):
        m.apply_fix()

def modify_istftnet_ada1n(model : torch.nn.Module):
    if isinstance(model, NHWCAdaIN1d):
        with torch.no_grad():
            fc = model.fc
            fc1 = model.fc1
            fc2 = model.fc2
            # the matrix is transposed, if we want to split output dim
            # then we need to split the row of underlying weight matrix
            assert fc.weight.shape[0] % 2 == 0
            num_features = fc.weight.shape[0] // 2
            # Split the weight tensor and copy to the new layers
            fc1.weight.copy_(fc.weight[:num_features, :])
            fc2.weight.copy_(fc.weight[num_features:, :])

            # Split the bias tensor (if it exists) and copy
            if fc.bias is not None:
                fc1.bias.copy_(fc.bias[:num_features])
                fc2.bias.copy_(fc.bias[num_features:])
            del model.fc
from .model import KModel
from .fix import af
def fix(model : KModel):
    # let's load the network first
    init_nchw_network()
    model.apply(af)
    model.apply(fix_conv1d)
    model.apply(modify_istftnet_ada1n)
    # TODO : disable force_earge one day
    torch.compiler.set_stance("force_eager")