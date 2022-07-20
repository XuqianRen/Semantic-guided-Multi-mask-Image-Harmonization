from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
import torchvision
from basicsr.utils.registry import ARCH_REGISTRY
import functools

class Identity(nn.Module):
    def forward(self, x):
        return x

def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

@ARCH_REGISTRY.register()
class UnetGeneratorRGB(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf, norm_type, use_dropout=False, use_attention=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_type      -- normalization type

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        norm_layer = get_norm_layer(norm_type)
        super(UnetGeneratorRGB, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_attention=use_attention)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_attention=use_attention)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_attention=use_attention)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

        #initialization
        
    def forward(self, x):
        """Standard forward"""
       
        output = self.model(x)
        
        return output



class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, use_attention=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                            stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = UpsampleConvLayer(inner_nc*2, outer_nc, kernel_size=3, stride=1, upsample=2)      
            # upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2,padding=1, bias=use_bias)
            down = [downconv]
            up = [uprelu, upconv]
            model = down + [submodule] + up
        

        elif innermost:
            upconv = UpsampleConvLayer(inner_nc, outer_nc, kernel_size=3, stride=1, upsample=2)
            # upconv = nn.ConvTranspose2d(inner_nc, outer_nc,kernel_size=4, stride=2,padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = UpsampleConvLayer(inner_nc*2, outer_nc, kernel_size=3, stride=1, upsample=2)
            # upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,kernel_size=4, stride=2,padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up
        self.use_attention = use_attention
        if use_attention:
            attention_conv = nn.Conv2d(input_nc+outer_nc, input_nc+outer_nc, kernel_size=1)
            attention_relu = nn.Sigmoid()
            self.attention = nn.Sequential(*[attention_conv, attention_relu])
        self.model = nn.Sequential(*model)
        self.relu = nn.ReLU()
    def forward(self, x):
        if self.outermost:
            x = self.model(x)
  
            return x
            
        else:       
            ret = torch.cat([x, self.model(x)], 1)
            if self.use_attention:
                return self.attention(ret) * ret
            return ret 
        
class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample_layer = torch.nn.Upsample(scale_factor=upsample,mode='nearest')
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = self.upsample_layer(x_in)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out

