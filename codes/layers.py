from torch import nn

import utils_torch_json as torch_util

class denseLayer(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 activation,
                 is_bias = True,
                 *args,
                 **kwargs,):
        super().__init__(*args, **kwargs)

        self.f_activation   = torch_util.get_activation(activation)
        self.linear         = nn.Linear(in_features     = in_features,
                                        out_features    = out_features,
                                        bias            = is_bias)
        
    def forward(self, inputs):
        return self.f_activation( self.linear(inputs))
    
    def forward_with_preactivation(self, inputs,):
        pre_act = self.linear(inputs)
        outputs = self.f_activation(pre_act)
        return outputs, pre_act
    

class lazyDenseLayer(nn.Module):
    def __init__(self,
                 out_features,
                 activation,
                 is_bias = True,
                 *args,
                 **kwargs,):
        super().__init__(*args, **kwargs)

        self.f_activation   = torch_util.get_activation(activation)
        self.linear         = nn.LazyLinear(out_features    = out_features,
                                            bias            = is_bias)
        
    def forward(self, inputs):
        return self.f_activation( self.linear(inputs))
    
    def forward_with_preactivation(self, inputs,):
        pre_act = self.linear(inputs)
        outputs = self.f_activation(pre_act)
        return outputs, pre_act

class convLayer2d(nn.Module):
    def __init__(   self,
                    in_channels,
                    out_channels,
                    kernel_size,
                    strides,
                    activation,
                    groups              = 1,
                    dilation            = 1,
                    padding             = 'valid',
                    padding_mode        = 'zeros',
                    is_bias             = True,
                    dtype               = None,
                    *args,
                    **kwargs,):
        super().__init__(*args, **kwargs)

        self.f_activation   = torch_util.get_activation(activation)

        self.conv2d = nn.Conv2d(    in_channels     = in_channels,
                                    out_channels    = out_channels,
                                    kernel_size     = kernel_size,
                                    stride          = strides,
                                    padding         = padding,
                                    dilation        = dilation,
                                    groups          = groups,
                                    bias            = is_bias,
                                    padding_mode    = padding_mode,
                                    dtype           = dtype,)
        
    def forward(self, inputs):
        return self.f_activation( self.conv2d(inputs))
    
    def forward_with_preactivation(self, inputs):
        pre_act     = self.conv2d(inputs)
        outputs     = self.f_activation(pre_act)
        return outputs, pre_act
    
class convTransposeLayer2d(nn.Module):
    def __init__(   self,
                    in_channels,
                    out_channels,
                    kernel_size,
                    strides,
                    activation,
                    groups          = 1,
                    dilation        = 1,
                    padding         = 0,
                    padding_mode    = 'zeros',
                    output_padding  = 0,
                    is_bias         = True,
                    dtype           = None,
                    *args,
                    **kwargs,
                    ):
        super().__init__(*args, **kwargs)

        self.f_activation   = torch_util.get_activation(activation)

        self.convTrans2d = nn.ConvTranspose2d(  in_channels     = in_channels,
                                                out_channels    = out_channels,
                                                kernel_size     = kernel_size,
                                                stride          = strides,
                                                padding         = padding,
                                                output_padding  = output_padding,
                                                groups          = groups,
                                                bias            = is_bias,
                                                dilation        = dilation,
                                                padding_mode    = padding_mode,
                                                dtype           = dtype,
                                                )
    def forward(self, inputs):
        return self.f_activation( self.convTrans2d(inputs))
    
    def forward_with_preactivation(self, inputs):
        pre_act     = self.convTrans2d(inputs)
        outputs     = self.f_activation(pre_act)
        return outputs, pre_act