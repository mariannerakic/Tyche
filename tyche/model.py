from dataclasses import dataclass
from os import device_encoding
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import einops as E
import torch
from torch import nn

from .nn.cross_conv import CrossConv2d
from .nn.init import reset_conv2d_parameters
from .nn.vmap import Vmap, vmap
from .validation import (
    Kwargs,
    as_2tuple,
    size2t,
    validate_arguments,
    validate_arguments_init,
)


def get_nonlinearity(nonlinearity: Optional[str]) -> nn.Module:
    if nonlinearity is None:
        return nn.Identity()
    if nonlinearity == "Softmax":
        # For Softmax, we need to specify the channel dimension
        return nn.Softmax(dim=1)
    if hasattr(nn, nonlinearity):
        return getattr(nn, nonlinearity)()
    raise ValueError(f"nonlinearity {nonlinearity} not found")


@validate_arguments_init
@dataclass(eq=False, repr=False)
class ConvOp(nn.Sequential):

    in_channels: int
    out_channels: int
    kernel_size: size2t = 3
    nonlinearity: Optional[str] = "LeakyReLU"
    init_distribution: Optional[str] = "kaiming_normal"
    init_bias: Union[None, float, int] = 0.0

    def __post_init__(self):
        super().__init__()
        self.conv = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            padding_mode="zeros",
            bias=True,
        )

        if self.nonlinearity is not None:
            self.nonlin = get_nonlinearity(self.nonlinearity)

        reset_conv2d_parameters(
            self, self.init_distribution, self.init_bias, self.nonlinearity
        )


@validate_arguments_init
@dataclass(eq=False, repr=False)
class CrossOp(nn.Module):

    in_channels: size2t
    out_channels: int
    kernel_size: size2t = 3
    nonlinearity: Optional[str] = "LeakyReLU"
    init_distribution: Optional[str] = "kaiming_normal"
    init_bias: Union[None, float, int] = 0.0

    def __post_init__(self):
        super().__init__()

        self.cross_conv = CrossConv2d(
            in_channels=as_2tuple(self.in_channels),
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
        )

        if self.nonlinearity is not None:
            self.nonlin = get_nonlinearity(self.nonlinearity)


        reset_conv2d_parameters(
            self, self.init_distribution, self.init_bias, self.nonlinearity
        )

    def forward(self, target, support):
        interaction = self.cross_conv(target, support).squeeze(dim=1)

        if self.nonlinearity is not None:
            interaction = vmap(self.nonlin, interaction)

        new_target = interaction.mean(dim=1, keepdims=True)

        return new_target, interaction
    

@validate_arguments_init
@dataclass(eq=False, repr=False)
class CrossOpTarget(nn.Module):

    in_channels: size2t
    out_channels: int
    kernel_size: size2t = 3
    nonlinearity: Optional[str] = "LeakyReLU"
    init_distribution: Optional[str] = "kaiming_normal"
    init_bias: Union[None, float, int] = 0.0

    def __post_init__(self):
        super().__init__()

        self.cross_conv = CrossConv2d(
            in_channels=as_2tuple(self.in_channels),
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
        )

        if self.nonlinearity is not None:
            self.nonlin = get_nonlinearity(self.nonlinearity)


        reset_conv2d_parameters(
            self, self.init_distribution, self.init_bias, self.nonlinearity
        )

    def forward(self, target, support):
        interaction = self.cross_conv(target, support).squeeze(dim=1)

        if self.nonlinearity is not None:
            interaction = vmap(self.nonlin, interaction)

        return interaction
    

class Residual(nn.Module):
    @validate_arguments
    def __init__(
        self, module, in_channels: int, out_channels: int,
    ):
        super().__init__()
        self.main = module
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            # TODO do we want to init these to 1, like controlnet's zeroconv
            # TODO do we want to initialize these like the other conv layers
            self.shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, bias=False
            )
            reset_conv2d_parameters(self.shortcut, "kaiming_normal", 0.0)

    def forward(self, input):
        return self.main(input) + self.shortcut(input)


class VResidual(Residual):
    def forward(self, input):
        return self.main(input) + vmap(self.shortcut, input)


@validate_arguments_init
@dataclass(eq=False, repr=False)
class CrossBlockTarget(nn.Module):

    in_channels: size2t
    cross_features: int
    conv_features: Optional[int] = None
    cross_kws: Optional[Dict[str, Any]] = None
    conv_kws: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        super().__init__()

        conv_features = self.conv_features or self.cross_features
        cross_kws = self.cross_kws or {}
        conv_kws = self.conv_kws or {}

        self.cross = CrossOp(self.in_channels, self.cross_features, **cross_kws)

        if isinstance(self.in_channels, tuple): 
            mean_convs = self.in_channels[0] + self.cross_features
        else:
            mean_convs = self.in_channels + self.cross_features


        self.meanconv  = Vmap(ConvOp(mean_convs, self.conv_features, **self.conv_kws or {}))
        
        self.target = Vmap(ConvOp(self.cross_features, conv_features, **conv_kws))
        self.support = Vmap(ConvOp(self.cross_features, conv_features, **conv_kws))

    def forward(self, target, support):
        mean_img  = target.mean(dim=1, keepdims=True)#[:, None, ...]

        mean_img, support = self.cross(mean_img, support)

        K = target.shape[1]

        mean_img = E.repeat(mean_img, 'B 1 C H W -> B K C H W', K=K)

        target = torch.cat([target, mean_img], dim=2)
        target = self.meanconv(target) 

        target = self.target(target)
        support = self.support(support)
        return target, support


@validate_arguments_init
@dataclass(eq=False, repr=False)
class TysegXC(nn.Module):

    encoder_blocks: List[size2t]
    decoder_blocks: Optional[List[size2t]] = None
    cross_relu: bool = True

    def __post_init__(self):
        super().__init__()

        self.downsample = nn.MaxPool2d(2, 2)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        self.enc_blocks = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()

        encoder_blocks = list(map(as_2tuple, self.encoder_blocks))
        decoder_blocks = self.decoder_blocks or encoder_blocks[-2::-1]
        decoder_blocks = list(map(as_2tuple, decoder_blocks))

        if self.cross_relu:
            block_kws = dict(cross_kws=dict(nonlinearity="LeakyReLU"))
        else:
            block_kws = dict(cross_kws=dict(nonlinearity=None))

        in_ch = (2, 2)
        out_channels = 1 #3 outputs to compare to SAM
        out_activation = None

        # Encoder
        skip_outputs = []
        for (cross_ch, conv_ch) in encoder_blocks:
            block = CrossBlockTarget(in_ch, cross_ch,  conv_ch, **block_kws)
            in_ch = conv_ch
            self.enc_blocks.append(block)
            skip_outputs.append(in_ch)

        # Decoder
        skip_chs = skip_outputs[-2::-1]
        for (cross_ch, conv_ch), skip_ch in zip(decoder_blocks, skip_chs):
            block = CrossBlockTarget(in_ch + skip_ch, cross_ch, conv_ch, **block_kws)
            in_ch = conv_ch
            self.dec_blocks.append(block)

        self.out_conv = ConvOp(
            in_ch, out_channels, kernel_size=1, nonlinearity=out_activation,
        )

    def get_model_inputs(self, x, sx, sy, target_size, aug=False):
        """
        Gather all the input for the model and put them in the right shape.
        """
        x= self.format_target(x, target_size)
        noise = torch.randn_like(x)

        sx, sy = [self.format_support(i) for i in [sx, sy]]

        return {'support_images': sx, 'support_labels': sy,
                                    'target_image': x, 'noise_image': noise}
        

    def format_target(self, x, target_size):
        """
        This is meant exclusively in an inference setting with batch size of 1.
        If target, should have shape: (1 1 H W) or (1, H, W).
        For model input, it needs to have shape: (B, K, X, H, W) with B=1.
        """
        x_shape = x.shape

        if x_shape[0]>1: 
            if len(x_shape)==5: # B 1 1 H W
                assert x_shape[2]==1, x_shape
                assert x_shape[1]==1, x_shape
                x = E.repeat(x, 'B 1 1 H W -> B K 1 H W', K=target_size)
                return x
            
            elif len(x_shape)==4: # B 1 H W
                x = E.repeat(x, 'B 1 H W -> B K H W', K=target_size)
                return x[:, :, None, :, :]
            
            elif len(x_shape)==3: # B H W 
                x = E.rearrange(x, "B H W -> B 1 1 H W")
                x = E.repeat(x, 'B 1 1 H W -> B K 1 H W', K=target_size)
                return x
            
            else:
                return 0
            
            
        assert 3<=len(x_shape)<=4, 'Input should have shape 1, 1, H, W or 1, H, W.'

        if len(x_shape)==3:
            x = x[None]

        x = E.repeat(x, '1 1 H W -> K 1 H W', K=target_size)
        x = E.rearrange(x, "K 1 H W -> 1 K 1 H W")    
        return x

    def format_support(self, sx,):
        """
        This is meant exclusively in an inference setting with batch size of 1.
        If support, should have shape: (S 1 H W) or (1, S, 1, H, W).
        For model input, it needs to have shape: (B, S, 1, H, W) with B=1.        
        """
        sx_shape = sx.shape
        if len(sx_shape)==4:
            assert sx_shape[1]==1, 'Support should have shape (1, S, 1, H, W), or (S, 1, H, W).'
            sx = sx[None]

        elif len(sx_shape)==5:
            assert sx_shape[2] == 1, 'Support should have shape (S, 1, H, W) or (1, S, 1, H, W)'

        return sx

    def format_pred(self, yhat):
        assert len(yhat.shape)==5
        assert yhat.shape[2] == 1, 'Prediction should have shape (1, K, 1, H, W)' 
        return yhat[:, :, 0]
    
    def format_label(self, y, target_size):
        '''
        Go back from (1, K, 1, H, W) to (1, K, H, W)
        '''
        y_shape = y.shape
        assert y_shape[0]==1, 'Input should have batch size and channel of 1.'
        assert 3<=len(y_shape)<=4, 'Input should have shape 1, 1, H, W or 1, H, W.'

        if len(y_shape)==3:
            y = y[None]

        y = E.repeat(y, '1 1 H W -> 1 K H W', K=target_size)
        
        return y
    
    def pred_ged_stats(self, m_inputs, sigmoid=True):
        model_inputs = self.get_model_inputs(**m_inputs)
        yhat_tmp = self.forward(**model_inputs)
        yhat_tmp = self.format_pred(yhat_tmp)

        if sigmoid:
            yhat_tmp = torch.sigmoid(yhat_tmp)

        return yhat_tmp

    def forward(self, support_images, support_labels, target_image, noise_image):

        
        target = torch.cat([target_image, noise_image], dim=2)
        support = torch.cat([support_images, support_labels], dim=2)

        B, K, _, _, _ = target.shape

        pass_through = []

        for i, encoder_block in enumerate(self.enc_blocks):
            target, support = encoder_block(target, support)
            if i == len(self.encoder_blocks) - 1:
                break
            pass_through.append((target, support))
            target = vmap(self.downsample, target)
            support = vmap(self.downsample, support)

        for decoder_block in self.dec_blocks:
            target_skip, support_skip = pass_through.pop()
            target = torch.cat([vmap(self.upsample, target), target_skip], dim=2)
            support = torch.cat([vmap(self.upsample, support), support_skip], dim=2)
            target, support = decoder_block(target, support)

        target = E.rearrange(target, "B K C H W  -> (B K) C H W")
        target = self.out_conv(target)

        target = E.rearrange(target, "(B K) C H W  -> B K C H W", B=B, K=K)
        
        return target


@validate_arguments
def tychets(version: Literal["v1"] = "v1", pretrained: bool = False) -> nn.Module:
    weights = {
            "v1": "https://github.com/mariannerakic/Tyche/releases/download/weights/tyche_v1_model_weights_CVPR.pt"
            }
    if version == "v1":
        model = TysegXC(encoder_blocks=[64, 64, 64, 64])

    if pretrained:
       state_dict = torch.hub.load_state_dict_from_url(weights[version])
       model.load_state_dict(state_dict)

    return model
