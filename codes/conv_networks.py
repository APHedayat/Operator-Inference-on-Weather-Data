import os
import json
import numpy as np

import torch
from torch import nn



from layers import denseLayer, lazyDenseLayer, convLayer2d,\
    convTransposeLayer2d

from utils_torch_json import NpEncoder

class convAutoencoder2dV1Settings:
    def __init__(   self,
                    n_layers_conv,
                    kernel_size,
                    strides,
                    filters_encoder,
                    filters_decoder,
                    activation_conv,
                    n_layers_hidden_dense,
                    n_nodes_hidden_dense,
                    inputs_shape,
                    inputs_channels,
                    outputs_shape,
                    outputs_channels,
                    activation_dense,
                    groups              = 1,
                    dilation            = 1,
                    padding             = 'valid',
                    padding_mode_conv   = 'zeros',
                    is_layer_norm_conv  = False,
                    is_layer_norm_dense = False,
                    is_linear_output    = True,
                    is_bias_conv        = True,
                    is_bias_dense       = True,
                    flatten_dim         = None,
                    expand_dim          = None,):
        '''Settings for 2d convolutional autoencoder'''

        opt = locals().copy()
        del opt['self']
        self.args = opt

        # check args
        assert len(filters_encoder) == n_layers_conv
        assert len(filters_decoder) == n_layers_conv
        assert len(kernel_size) == n_layers_conv
        assert len(strides) == n_layers_conv

        # ensure output channels are correct
        filters_decoder[-1] = outputs_channels

        self.n_layers_conv          = n_layers_conv
        self.kernel_size_encoder    = kernel_size
        self.kernel_size_decoder    = kernel_size.copy()
        self.kernel_size_decoder.reverse() #decoder is mirror of encoder
        self.strides_encoder        = strides
        self.strides_decoder        = strides.copy()
        self.strides_decoder.reverse() # decoder is mirror of encoder
        self.filters_encoder        = filters_encoder
        self.filters_decoder        = filters_decoder
        self.activation_conv        = activation_conv
        self.n_layers_hidden_dense  = n_layers_hidden_dense
        self.n_nodes_hidden_dense   = n_nodes_hidden_dense
        self.inputs_shape           = inputs_shape
        self.inputs_channels        = inputs_channels
        self.outputs_shape          = outputs_shape
        self.outputs_channels       = outputs_channels
        self.activation_dense       = activation_dense
        self.groups                 = groups
        self.dilation               = dilation
        self.padding                = padding
        self.padding_mode_conv      = padding_mode_conv
        self.is_layer_norm_conv     = is_layer_norm_conv
        self.is_layer_norm_dense    = is_layer_norm_dense
        self.is_linear_output       = is_linear_output
        self.is_bias_conv           = is_bias_conv
        self.is_bias_dense          = is_bias_dense

        self.flatten_dim            = flatten_dim
        self.expand_dim             = expand_dim


class convEncoder2d(nn.Module):
    def __init__(self,
                 opt,
                 encoder_shapes,
                 *args,
                 **kwargs,):
        super().__init__(*args, **kwargs)
        self.opt            = opt
        self.encoder_shapes = encoder_shapes
        self.build_model()


    def build_model(self,):
        layers = []
        # build conv encoder
        for iLayer in np.arange(self.opt.n_layers_conv):
            if iLayer == 0:
                in_channels     = self.opt.inputs_channels
                out_channels    = self.opt.filters_encoder[iLayer]
                
            else:
                in_channels     = self.opt.filters_encoder[iLayer-1]
                out_channels    = self.opt.filters_encoder[iLayer]

            cur_layer = convLayer2d(
                in_channels         = in_channels,
                out_channels        = out_channels,
                kernel_size         = self.opt.kernel_size_encoder[iLayer],
                strides             = self.opt.strides_encoder[iLayer],
                activation          = self.opt.activation_conv,
                groups              = self.opt.groups,
                dilation            = self.opt.dilation,
                padding             = self.opt.padding,
                padding_mode        = self.opt.padding_mode_conv,
                is_bias             = self.opt.is_bias_conv,
                )
            layers.append(cur_layer)

            # optional layer norm
            if self.opt.is_layer_norm_conv:
                norm_layer = nn.LayerNorm(normalized_shape = self.encoder_shapes[iLayer])
                layers.append(norm_layer)

        self.layers = layers
        self.net    = nn.Sequential(*self.layers)

    def forward(self, inputs):
        return self.net(inputs)
    

class convDecoder2d(nn.Module):
    def __init__(self,
                 opt,
                 decoder_shapes,
                 *args,
                 **kwargs,):
        super().__init__(*args, **kwargs)
        self.opt            = opt
        self.decoder_shapes = decoder_shapes
        # conv2d uses string, while convTranspose2d uses int or tuple of int
        if self.opt.padding == 'valid':
            self.opt.padding_arg = 0
        self.build_model()

    def build_model(self,):
        layers = []
        # build conv decoder
        for iLayer in np.arange(self.opt.n_layers_conv):
            if iLayer == 0:
                in_channels     = self.opt.filters_decoder[iLayer]
                out_channels    = self.opt.filters_decoder[iLayer]
                
            else:
                in_channels     = self.opt.filters_decoder[iLayer-1]
                out_channels    = self.opt.filters_decoder[iLayer]

            if iLayer == self.opt.n_layers_conv -1 and self.opt.is_linear_output:
                activ = 'linear'
            else:
                activ = self.opt.activation_conv

            cur_layer = convTransposeLayer2d(
                in_channels         = in_channels,
                out_channels        = out_channels,
                kernel_size         = self.opt.kernel_size_decoder[iLayer],
                strides             = self.opt.strides_decoder[iLayer],
                activation          = activ,
                groups              = self.opt.groups,
                dilation            = self.opt.dilation,
                padding             = self.opt.padding_arg,
                padding_mode        = self.opt.padding_mode_conv,
                output_padding      = 0,
                is_bias             = self.opt.is_bias_conv,
            )
            layers.append(cur_layer)

            # optional layer norm
            if self.opt.is_layer_norm_conv:
                norm_layer = nn.LayerNorm(normalized_shape = self.decoder_shapes[iLayer])
                layers.append(norm_layer)

        self.layers = layers
        self.net    = nn.Sequential(*self.layers)

    def forward(self, inputs):
        return self.net(inputs)
             
class convAutoencoder2dV1(nn.Module):
    def __init__(   self, 
                    opt,
                    *args,
                    **kwargs,):
        super().__init__(*args, **kwargs)
        self.opt            = opt
        
        # self.opt is updated with flattened and expanded dimensions
        # model_shapes tracks throughout entire model, while encoder_shapes
        # and decoder_shapes track on the hidden dimension through the conv
        # layers, for use with layer normalization 
        self.model_shapes, self.encoder_shapes, self.decoder_shapes =\
            self.compute_intermediate_dimensions()

        self.build_model()

    def save_opt(self,):
        '''Save the parameters needed to build the model to model_path/model_settings.json
        '''

        opt_dict = self.opt.__dict__.copy()
        del opt_dict['args']
        json.dump( opt_dict, open(self.fn_opt, 'w'), cls = NpEncoder, indent = 4, )

    def build_model(self,):
        self.convEncoder = convEncoder2d(   opt             = self.opt,
                                            encoder_shapes  = self.encoder_shapes,)
        self.flatten = nn.Flatten()
        self.build_dense_layers()
        self.convDecoder = convDecoder2d(   opt             = self.opt,
                                            decoder_shapes  = self.decoder_shapes,)

    def build_dense_layers(self,):
        layers = []

        for iLayer in np.arange(self.opt.n_layers_hidden_dense):
            if iLayer == 0:
                in_features     = self.opt.flatten_dim
            else:
                in_features     = self.opt.n_nodes_hidden_dense[iLayer-1]
                
            out_features    = self.opt.n_nodes_hidden_dense[iLayer]
            cur_layer = denseLayer( in_features     = in_features,
                                    out_features    = out_features,
                                    activation      = self.opt.activation_dense,
                                    is_bias         = self.opt.is_bias_dense,)
                
            layers.append(cur_layer)    

            # optionally include layer normalization except for after output layer
            if self.opt.is_layer_norm_dense and not (iLayer == self.opt.n_layers_dense - 1):
                norm_layer  = nn.LayerNorm( normalized_shape = self.opt.n_nodes_dense[iLayer]),
                layers.append(norm_layer)

        self.dense_layers = layers
        self.dense_net    = nn.Sequential(*self.dense_layers)

    def compute_intermediate_dimensions(self,):
        '''Track the hidden state shapes through the network'''

        shapes, conv_encoder_hidden, conv_decoder_hidden = [], [], []
        # conv encoder
        shapes.append((self.opt.inputs_channels, *self.opt.inputs_shape))
        if self.opt.padding == 'valid':
            padding = [(0,0),]*self.opt.n_layers_conv
        for iLayer, (k1,k2) in enumerate(self.opt.kernel_size_encoder):
            d1 = self.conv_layer_dim_change(    d = shapes[-1][1],
                                                k = k1,
                                                p = padding[iLayer][0],
                                                s = self.opt.strides_encoder[iLayer][0],
                                                )
            d2 = self.conv_layer_dim_change(    d = shapes[-1][2],
                                                k = k2,
                                                p = padding[iLayer][1],
                                                s = self.opt.strides_encoder[iLayer][1],
                                                )
            cur_shape = (self.opt.filters_encoder[iLayer], d1, d2)
            shapes.append( cur_shape )
            conv_encoder_hidden.append( cur_shape )

        conv_out_shape = tuple(shapes[-1])
        # dense layers
        flatten_dim   = shapes[-1][0]  * shapes[-1][1] * shapes[-1][2]
        
        shapes.append(flatten_dim)

        # update self.opt.n_nodes_hidden_dense with appropriate last layer size
        # to match the decoder
        expand_dim      = self.opt.filters_decoder[0] * conv_out_shape[1] * conv_out_shape[2]
        self.opt.n_nodes_hidden_dense[-1] = expand_dim

        for iLayer, nodes in enumerate(self.opt.n_nodes_hidden_dense):
            shapes.append(nodes)
        
        self.decoder_input_shape = (self.opt.filters_decoder[0], conv_out_shape[1], 
                        conv_out_shape[2],)
        shapes.append( self.decoder_input_shape )

        # dconv decoder
        for iLayer, (k1,k2) in enumerate(self.opt.kernel_size_decoder):
            d1 = self.transposed_conv_layer_dim_change( d = shapes[-1][1],
                                                        k = k1,
                                                        p = padding[iLayer][0],
                                                        s = self.opt.strides_decoder[iLayer][0],
                                                )
            d2 = self.transposed_conv_layer_dim_change( d = shapes[-1][2],
                                                        k = k2,
                                                        p = padding[iLayer][1],
                                                        s = self.opt.strides_decoder[iLayer][1],
                                                )
            cur_shape   = (self.opt.filters_decoder[iLayer], d1, d2)
            shapes.append( cur_shape )
            conv_decoder_hidden.append(cur_shape)

        assert shapes[-1][0] == self.opt.outputs_channels
        assert shapes[-1][1] == self.opt.outputs_shape[0]
        assert shapes[-1][2] == self.opt.outputs_shape[1]

        self.opt.flatten_dim    = flatten_dim
        self.opt.expand_dim     = expand_dim

        return shapes, conv_encoder_hidden, conv_decoder_hidden
        
    def conv_layer_dim_change(self, d, k, p, s):
        '''Compute convolutional layer output dimension 
        Args:
            d (int)     : incoming dimension
            k (int)     : kernel size
            p (int)     : padding size
            s (int)     : stride
        
        Returns:
            dnew (int)  : new dimension
        '''
        dnew = int(np.floor( ((d - k + 2*p) / s) +1 ))
        return dnew
    
    def transposed_conv_layer_dim_change(self, d, k, p, s):
        '''Compute transposed convolutional layer output dimension 
        Args:
            d (int)     : incoming dimension
            k (int)     : kernel size
            p (int)     : padding size
            s (int)     : stride
        
        Returns:
            dnew (int)  : new dimension
        '''
        dnew = int(np.floor( ((d-1)*s) - (2*p) + k))
        return dnew
    
    def forward(self, inputs):
        h = self.convEncoder(inputs)
        h = self.flatten(h)
        h = self.dense_net(h)
        h = torch.reshape(h, (-1, *self.decoder_input_shape))
        h = self.convDecoder(h)
        return h
    

def dev_autoencoder():
    opt = convAutoencoder2dV1Settings(
        n_layers_conv           = 4,
        kernel_size             = [(2,2),]*4,
        strides                 = [(2,2),]*4,
        filters_encoder         = [50,]*4,
        filters_decoder         = [50,]*4,
        activation_conv         = 'swish',
        n_layers_hidden_dense   = 2,
        n_nodes_hidden_dense    = [100,-1], # use for last layer to determine automatically
        inputs_shape            = (128, 128),
        inputs_channels         = 4,
        outputs_shape           = (128,128),
        outputs_channels        = 4,
        activation_dense        = 'swish',
        groups                  = 1,
        dilation                = 1,
        padding                 = 'valid',
        padding_mode_conv       = 'zeros',
        is_layer_norm_conv      = True,
        is_layer_norm_dense     = False,
        is_linear_output        = True,
        is_bias_conv            = True,
        is_bias_dense           = True,)
    
    base_path   = os.path.join( os.environ['SCIFM_PATH'], 'Results')
    name        = 'convAETest'
    model_path  = os.path.join( base_path, name)

    if not os.path.exists(model_path):
        os.mkdir(model_path)

    model = convAutoencoder2dV1(opt         = opt,
                                model_path  = model_path,)
    print(model)
    inp = torch.zeros((1,4,128,128))
    out = model(inp)
    test = 1


if __name__ == '__main__':
    dev_autoencoder()