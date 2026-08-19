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
from ..utils.ppm import PyramidPoolingModule
from ..utils.bricks import BuildActivation, BuildNormalization
from .transformers import StandardTransformerDecoder, SetCriterion, Transformer, HungarianMatcher
from networks.backbone import Build_Backbone

cfg = {
        'decoder': {
        'mask': {'in_channels': 512, 'out_channels': 256},
        'predictor': {
            'in_channels': 2048,
            'mask_classification': True,
            'hidden_dim': 256,
            'num_queries': 128,
            'nheads': 8,
            'dropout': 0.0,
            'dim_feedforward': 2048,
            'enc_layers': 0,
            'dec_layers': 6,
            'pre_norm': False,
            'deep_supervision': True,
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


'''MaskFormer'''
class MaskFormer(nn.Module):
    def __init__(self, backbone_name, pretrained, num_band, num_class=1, mode='seg', pretrained_path='', **kwargs):
        super(MaskFormer, self).__init__()
        self.align_corners = True
        act_cfg = {'type': 'relu', 'opts': {'inplace': True}}
        # build backbone
        if mode == 'change':
            num_band = num_band*2
        self.backbone, self.channels_blocks, self.do_upsample = Build_Backbone(
            backbone_name, pretrained, num_band, pretrained_path=pretrained_path
        )
        self.num_class = num_class
        # set channels
        mid_channels = 256
        cfg['lateral']['in_channels_list'] = [self.channels_blocks[3], self.channels_blocks[2], self.channels_blocks[1]]
        cfg['lateral']['out_channels'] = mid_channels
        cfg['fpn']['in_channels_list'] = [mid_channels, mid_channels, mid_channels]
        cfg['fpn']['out_channels'] = mid_channels

        # build pyramid pooling module
        self.ppm_net = PyramidPoolingModule(self.channels_blocks[0], mid_channels, norm="layernorm")
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
        cfg['decoder']['predictor']['in_channels'] = self.channels_blocks[0]
        self.decoder_predictor = StandardTransformerDecoder(**cfg['decoder']['predictor'])
        # matcher = HungarianMatcher(**cfg['decoder']['matcher'])
        # weight_dict = {'loss_ce': cfg['decoder']['matcher']['cost_class'], 'loss_mask': cfg['decoder']['matcher']['cost_mask'], 'loss_dice': cfg['decoder']['matcher']['cost_dice']}
        # if cfg['decoder']['predictor']['deep_supervision']:
        #     dec_layers = cfg['decoder']['predictor']['dec_layers']
        #     aux_weight_dict = {}
        #     for i in range(dec_layers - 1):
        #         aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        #     weight_dict.update(aux_weight_dict)
        # self.criterion = SetCriterion(num_class, matcher=matcher, weight_dict=weight_dict, eos_coef=0.1, losses=['labels', 'masks'])
        
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
        predictions_for_loss = self.decoder_predictor(backbone_outputs[-1], mask_features)
        # # forward according to the mode
        mask_cls_results = predictions_for_loss['pred_logits']
        mask_pred_results = predictions_for_loss['pred_masks']
        mask_pred_results = F.interpolate(mask_pred_results, size=img_size, mode='bilinear', align_corners=self.align_corners)
        predictions = []
        for mask_cls, mask_pred in zip(mask_cls_results, mask_pred_results):
            # semseg = self.general_inference(mask_cls, mask_pred)
            mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
            mask_pred = mask_pred.sigmoid()
            semseg = torch.einsum('qc,qhw->chw', mask_cls, mask_pred)#爱因斯坦求和，同样的维度相乘，输出省略的维度求和
            predictions.append(semseg.unsqueeze(0))
        predictions = torch.cat(predictions, dim=0)
        return predictions


    def semantic_inference(self, mask_cls, mask_pred):
        
        ####1####
        # mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        # mask_pred = mask_pred.sigmoid()
        # semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)

        ####2####
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)

        mask_pred = mask_pred.sigmoid()

        mask_cls = F.softmax(mask_cls[keep], dim=-1)[..., :-1]
        mask_pred = mask_pred[keep]
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)

        ####3####
        # scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        # keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)

        # mask_pred = mask_pred.sigmoid()

        # mask_cls = F.softmax(mask_cls[keep][:, :-1], dim=-1)
        # mask_pred = mask_pred[keep]
        # semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        #import ipdb;ipdb.set_trace()
        return semseg

    def general_inference(self, mask_cls, mask_pred):
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)#取128个msak中最可能的类别以及类别概率
        mask_pred = mask_pred.sigmoid()

        keep = labels.ne(self.num_class) & (scores > 0.8)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]

        cur_prob_masks = torch.einsum('q,qhw->qhw', cur_scores, cur_masks)#如果最大score都是0.9，那这里是不是没用了？

        h, w = cur_masks.shape[-2:]
        semseg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask 😞
            return semseg
        else:
        # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)#当前位置最可能属于哪个mask
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()#当前位置最可能属于的mask对应的类别
                mask = cur_mask_ids == k#画出mask的范围
                mask_area = mask.sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()

                if mask_area > 0 and original_area > 0:
                    if mask_area / original_area < 0.8:
                        continue
                semseg[mask] = int(pred_class)#把类别给到mask，得到语义分割结果

        return semseg

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


"Siam-MaskFormer"
class SiamMaskFormer(MaskFormer):
    def __init__(self, backbone_name, pretrained, num_band, num_class=1, mode='seg', pretrained_path='', **kwargs):
        super(SiamMaskFormer, self).__init__(backbone_name, pretrained, num_band, num_class, mode, pretrained_path, **kwargs)
        self.backbone, self.channels_blocks, self.do_upsample = Build_Backbone(
            backbone_name, pretrained, num_band, pretrained_path=pretrained_path
        )
        self.num_band = num_band
        #backbone
        self.fuse_blocks = nn.ModuleList([])
        for channel in self.channels_blocks:
            self.fuse_blocks.append(
                nn.Sequential(
                    nn.Conv2d(channel*2, channel, kernel_size=1, stride=1, padding=0, bias=False),
                    BuildNormalization('layernorm', (channel, {})),
                    torch.nn.ReLU(inplace=True),
                )
            )
        mid_channels = 256
        self.fuse_ppm = nn.Sequential(
                    nn.Conv2d(mid_channels*2, mid_channels, kernel_size=1, stride=1, padding=0, bias=False),
                    BuildNormalization('layernorm', (mid_channels, {})),
                    torch.nn.ReLU(inplace=True),
                )

    '''forward'''
    def forward(self, x):
        img_size = x.size(2), x.size(3)
        x1 = x[:,0:self.num_band,::]
        x2 = x[:,self.num_band:,::]
        layers1 = self.backbone(x1)
        layers2 = self.backbone(x2)
        ppm1 = self.ppm_net(layers1[-1])
        ppm2 = self.ppm_net(layers2[-1])
        # feed to backbone network
        backbone_outputs = [self.fuse_blocks[len(layers1)-i-1](torch.cat((layers1[i], layers2[i]), 1)) for i in range(len(layers1))]
        # feed to pyramid pooling module
        ppm_out = self.fuse_ppm(torch.cat((ppm1, ppm2), 1))
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
        predictions_for_loss = self.decoder_predictor(backbone_outputs[-1], mask_features)
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
        return [predictions, predictions_for_loss]