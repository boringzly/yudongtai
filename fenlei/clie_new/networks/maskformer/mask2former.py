'''
Function:
    Implementation of MaskFormer
Author:
    Zhenchao Jin
'''
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from networks.utils.bricks import BuildActivation, BuildNormalization
from .transformers import MultiScaleMaskedTransformerDecoder
from networks.backbone import Build_Backbone
from networks.utils.ASPP import ASPPModule

cfg = {
        'decoder': {
        'mask': {'in_channels': 512, 'out_channels': 256},
        'predictor': {
            'in_channels': 2048,
            'mask_classification': True,
            'hidden_dim': 256,
            'num_queries': 40,
            'nheads': 8,
            'dropout': 0.0,
            'dim_feedforward': 2048,
            # 'enc_layers': 0,
            'dec_layers': 6,
            'pre_norm': False,
            # 'deep_supervision': True,
            'mask_dim': 256,
            'enforce_input_project': False,
            # 'norm_cfg': {'type': 'layernorm', 'opts': {}},
            # 'act_cfg': {'type': 'relu', 'opts': {'inplace': True}},
        },
        'matcher': {'cost_class': 1.0, 'cost_mask': 20.0, 'cost_dice': 1.0},
    },
    'auxiliary': None,
        'matcher': {'cost_class': 1.0, 'cost_mask': 20.0, 'cost_dice': 1.0},
    'lateral': {
        'in_channels_list': [256, 512, 1024],
        'out_channels': 512,
    },
    'fpn': {
        'in_channels_list': [512, 512, 512],
        'out_channels': 512,
    },
}


'''Mask2Former'''
class Mask2Former(nn.Module):
    def __init__(self, backbone_name, pretrained, num_band, num_class=1, mode='seg', pretrained_path='', **kwargs):
        super(Mask2Former, self).__init__()
        self.align_corners = True
        act_cfg = {'type': 'relu', 'opts': {'inplace': True}}
        # build backbone
        if mode == 'change':
            num_band = num_band*2
        self.backbone, self.channels_blocks, self.do_upsample = Build_Backbone(
            backbone_name, pretrained, num_band, pretrained_path=pretrained_path
        )
        # set channels
        mid_channels = 256
        cfg['lateral']['in_channels_list'] = self.channels_blocks[::-1][0:-1]
        cfg['lateral']['out_channels'] = mid_channels
        cfg['fpn']['in_channels_list'] = [mid_channels]*len(cfg['lateral']['in_channels_list'])
        cfg['fpn']['out_channels'] = mid_channels

        # build pyramid pooling module
        self.ppm_net = ASPPModule(self.channels_blocks[0], mid_channels, norm="layernorm")
        # build lateral convs
        act_cfg_copy = copy.deepcopy(act_cfg)
        if 'inplace' in act_cfg_copy['opts']: act_cfg_copy['opts']['inplace'] = False
        lateral_cfg = cfg['lateral']
        self.lateral_convs = nn.ModuleList()
        for in_channels in lateral_cfg['in_channels_list']:
            self.lateral_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, lateral_cfg['out_channels'], kernel_size=1, stride=1, padding=0, bias=False),
                    BuildNormalization('layernorm', (lateral_cfg['out_channels'], {})),
                    torch.nn.ReLU(inplace=True),
                )
            )
        # build fpn convs
        fpn_cfg = cfg['fpn']
        self.fpn_convs = nn.ModuleList()
        for in_channels in fpn_cfg['in_channels_list']:
            self.fpn_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, fpn_cfg['out_channels'], kernel_size=3, stride=1, padding=1, bias=False),
                    BuildNormalization('layernorm', (fpn_cfg['out_channels'], {})),
                    torch.nn.ReLU(inplace=True),
                )
            )
        # build decoder
        cfg['decoder']['mask']['in_channels'] = mid_channels
        self.decoder_mask = nn.Sequential(
            nn.Conv2d(cfg['decoder']['mask']['in_channels'], cfg['decoder']['mask']['out_channels'], kernel_size=3, stride=1, padding=1)
        )
        cfg['decoder']['predictor']['num_classes'] = num_class
        cfg['decoder']['predictor']['in_channels'] = mid_channels
        self.decoder_predictor = MultiScaleMaskedTransformerDecoder(**cfg['decoder']['predictor'])
        
    '''forward'''
    def forward(self, x, targets=None, losses_cfg=None):
        img_size = x.size(2), x.size(3)
        # feed to backbone network
        backbone_outputs = self.backbone(x)
        # feed to pyramid pooling module
        ppm_out = self.ppm_net(backbone_outputs[-1])
        # apply fpn
        inputs = backbone_outputs[:-1]
        lateral_outputs = [lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)]
        lateral_outputs.append(ppm_out)
        p1, p2, p3, p4 = lateral_outputs
        fpn_out = F.interpolate(p4, size=p3.shape[2:], mode='bilinear', align_corners=self.align_corners) + p3
        fpn_out = self.fpn_convs[0](fpn_out)
        fpn_out = F.interpolate(fpn_out, size=p2.shape[2:], mode='bilinear', align_corners=self.align_corners) + p2
        fpn_out = self.fpn_convs[1](fpn_out)
        fpn_out = F.interpolate(fpn_out, size=p1.shape[2:], mode='bilinear', align_corners=self.align_corners) + p1
        fpn_out = self.fpn_convs[2](fpn_out)
        # feed to decoder
        mask_features = self.decoder_mask(fpn_out)
        predictions_for_loss = self.decoder_predictor(lateral_outputs[:-1], mask_features)
        # # forward according to the mode
        mask_cls_results = predictions_for_loss['pred_logits']
        mask_pred_results = predictions_for_loss['pred_masks']
        mask_pred_results = F.interpolate(mask_pred_results, size=img_size, mode='bilinear', align_corners=self.align_corners)
        predictions = []
        for mask_cls, mask_pred in zip(mask_cls_results, mask_pred_results):
            mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
            mask_pred = mask_pred.sigmoid()
            semseg = torch.einsum('qc,qhw->chw', mask_cls, mask_pred)
            predictions.append(semseg.unsqueeze(0))
        predictions = torch.cat(predictions, dim=0)
        return predictions

    '''return all layers'''
    def alllayers(self):
        all_layers = {
            'ppm_net': self.ppm_net,
            'lateral_convs': self.lateral_convs,
            'fpn_convs': self.fpn_convs,
            'decoder_mask': self.decoder_mask,
            'decoder_predictor': self.decoder_predictor,
        }
        tmp_layers = []
        for key, value in self.backbone_net.zerowdlayers().items():
            tmp_layers.append(value)
        all_layers.update({'backbone_net_zerowd': nn.Sequential(*tmp_layers)})
        tmp_layers = []
        for key, value in self.backbone_net.nonzerowdlayers().items():
            tmp_layers.append(value)
        all_layers.update({'backbone_net_nonzerowd': nn.Sequential(*tmp_layers)})
        return all_layers