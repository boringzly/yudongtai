'''
Unet
UnetPlusPlus
MAnet
Linknet
FPN
PSPNet
DeepLabV3
DeepLabV3Plus
PAN
custom_model
'''


def model_interface(opt):
    model_params = {}

    if opt.custom_model:

        if opt.model == 'Unet':
            model_params = {
                'arch': opt.model,
                'encoder_name': opt.encoder_name,
                'encoder_depth': opt.encoder_depth,
                'encoder_weights': None,
                'encoder_weights_offline': None,
                'encoder_weights_offline_path': None,
                'decoder_use_batchnorm': opt.decoder_use_batchnorm,
                'decoder_channels': opt.decoder_channels,
                'decoder_attention_type': opt.decoder_attention_type,
                'in_channels': opt.inchannel,
                'classes': opt.num_classes,
                'activation': opt.activation,
                'aux_params': opt.aux_params,
            }

        if opt.model == 'UnetPlusPlus':
            model_params = {
                'arch': opt.model,
                'encoder_name': opt.encoder_name,
                'encoder_depth': opt.encoder_depth,
                'encoder_weights': None,
                'encoder_weights_offline': None,
                'encoder_weights_offline_path': None,
                'decoder_use_batchnorm': opt.decoder_use_batchnorm,
                'decoder_channels': opt.decoder_channels,
                'decoder_attention_type': opt.decoder_attention_type,
                'in_channels': opt.inchannel,
                'classes': opt.num_classes,
                'activation': opt.activation,
                'aux_params': opt.aux_params,
            }

        if opt.model == 'MAnet':
            model_params = {
                'arch': opt.model,
                'encoder_name': opt.encoder_name,
                'encoder_depth': opt.encoder_depth,
                'encoder_weights': None,
                'encoder_weights_offline': None,
                'encoder_weights_offline_path': None,
                'decoder_use_batchnorm': opt.decoder_use_batchnorm,
                'decoder_channels': opt.decoder_channels,
                'decoder_pab_channels': opt.decoder_pab_channels,
                'in_channels': opt.inchannel,
                'classes': opt.num_classes,
                'activation': opt.activation,
                'aux_params': opt.aux_params,
            }

        if opt.model == 'Linknet':
            model_params = {
                'arch': opt.model,
                'encoder_name': opt.encoder_name,
                'encoder_depth': opt.encoder_depth,
                'encoder_weights': None,
                'encoder_weights_offline': None,
                'encoder_weights_offline_path': None,
                'decoder_use_batchnorm': opt.decoder_use_batchnorm,
                'in_channels': opt.inchannel,
                'classes': opt.num_classes,
                'activation': opt.activation,
                'aux_params': opt.aux_params,
            }

        if opt.model == 'FPN':
            model_params = {
                'arch': opt.model,
                'encoder_name': opt.encoder_name,
                'encoder_depth': opt.encoder_depth,
                'encoder_weights': None,
                'encoder_weights_offline': None,
                'encoder_weights_offline_path': None,
                'decoder_pyramid_channels': opt.decoder_pyramid_channels,
                'decoder_segmentation_channels': opt.decoder_segmentation_channels,
                'decoder_merge_policy': opt.decoder_merge_policy,
                'decoder_dropout': opt.decoder_dropout,
                'in_channels': opt.inchannel,
                'classes': opt.num_classes,
                'activation': opt.activation,
                'upsampling': opt.upsampling,
                'aux_params': opt.aux_params,
            }

        if opt.model == 'PSPNet':
            model_params = {
                'arch': opt.model,
                'encoder_name': opt.encoder_name,
                'encoder_depth': opt.encoder_depth,
                'encoder_weights': None,
                'encoder_weights_offline': None,
                'encoder_weights_offline_path': None,
                'psp_out_channels': opt.psp_out_channels,
                'psp_use_batchnorm': opt.psp_use_batchnorm,
                'psp_dropout': opt.decoder_dropout,
                'in_channels': opt.inchannel,
                'classes': opt.num_classes,
                'activation': opt.activation,
                'upsampling': opt.upsampling,
                'aux_params': opt.aux_params,
            }

        if opt.model == 'DeepLabV3':
            model_params = {
                'arch': opt.model,
                'encoder_name': opt.encoder_name,
                'encoder_depth': opt.encoder_depth,
                'encoder_weights': None,
                'encoder_weights_offline': None,
                'encoder_weights_offline_path': None,
                'decoder_channels': opt.decoder_channels,
                'decoder_attention_type': opt.decoder_attention_type,
                'in_channels': opt.inchannel,
                'classes': opt.num_classes,
                'activation': opt.activation,
                'upsampling': opt.upsampling,
                'aux_params': opt.aux_params,
            }

        if opt.model == 'DeepLabV3Plus':
            model_params = {
                'arch': opt.model,
                'encoder_name': opt.encoder_name,
                'encoder_depth': opt.encoder_depth,
                'encoder_weights': None,
                'encoder_weights_offline': None,
                'encoder_weights_offline_path': None,
                'encoder_output_stride': opt.encoder_output_stride,
                'decoder_channels': opt.decoder_channels,
                'decoder_atrous_rates': opt.decoder_atrous_rates,
                'in_channels': opt.inchannel,
                'classes': opt.num_classes,
                'activation': opt.activation,
                'upsampling': opt.upsampling,
                'aux_params': opt.aux_params,
            }

        if opt.model == 'PAN':
            model_params = {
                'arch': opt.model,
                'encoder_name': opt.encoder_name,
                'encoder_weights': None,
                'encoder_weights_offline': None,
                'encoder_weights_offline_path': None,
                'encoder_dilation': opt.encoder_dilation,
                'decoder_channels': opt.decoder_channels,
                'in_channels': opt.inchannel,
                'classes': opt.num_classes,
                'activation': opt.activation,
                'upsampling': opt.upsampling,
                'aux_params': opt.aux_params,
            }

        # CLSeg
        if opt.model == 'ResUnet':
            model_params = {
                'arch': opt.model,
                'encoder_name': opt.encoder_name,
                'encoder_depth': opt.encoder_depth,
                'encoder_weights': None,
                'encoder_weights_offline': None,
                'encoder_weights_offline_path': None,
                'decoder_use_batchnorm': opt.decoder_use_batchnorm,
                'decoder_channels': opt.decoder_channels,
                'decoder_attention_type': opt.decoder_attention_type,
                'decoder_res_type': opt.decoder_res_type,
                'in_channels': opt.inchannel,
                'classes': opt.num_classes,
                'activation': opt.activation,
                'aux_params': opt.aux_params,
            }

    else:

        model_params = {
            'arch': opt.model,
            'encoder_name': opt.encoder_name,
            'encoder_weights': None,
            'encoder_weights_offline': None,
            'encoder_weights_offline_path': None,
            'in_channels': opt.inchannel,
            'classes': opt.num_classes,
        }

    return model_params


def model_v1_interface(opt):
    model_params = {}

    if opt.model == 'RSPSPNet':
        model_params = {
            'inchannel': opt.inchannel,
            'num_classes': opt.num_classes,
            'basic_net': opt.basic_net,
            'psp_sizes': opt.psp_sizes,
            'drop_rate': opt.drop_rate,
            'norm_name': opt.norm_name
        }
    if opt.model == 'RSResUNet':
        model_params = {
            'inchannel': opt.inchannel,
            'num_classes': opt.num_classes,
            'basic_net': opt.basic_net,
            'drop_rate': opt.drop_rate,
            'norm_name': opt.norm_name
        }
    if opt.model == 'RSEffUNet':
        model_params = {
            'inchannel': opt.inchannel,
            'num_classes': opt.num_classes,
            'basic_net': opt.basic_net,
            'drop_rate': opt.drop_rate,
            'norm_name': opt.norm_name
        }

    return model_params