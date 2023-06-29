from collections import OrderedDict
from functools import partial
from typing import Any, Dict, Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from network.utils import IntermediateLayerGetter
from network.mobilenet_v3 import mobilenet_v3_large, MobileNetV3
from network.mynn import initialize_weights, Upsample


__all__ = ["LRASPP", "LRASPP_MobileNet_V3_Large_Weights", "lraspp_mobilenet_v3_large"]


class LRASPP(nn.Module):
    """
    Implements a Lite R-ASPP Network for semantic segmentation from
    `"Searching for MobileNetV3"
    <https://arxiv.org/abs/1905.02244>`_.

    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "high" for the high level feature map and "low" for the low level feature map.
        low_channels (int): the number of channels of the low level features.
        high_channels (int): the number of channels of the high level features.
        num_classes (int, optional): number of output classes of the model (including the background).
        inter_channels (int, optional): the number of channels for intermediate computations.
    """

    def __init__(
        self, args, backbone: nn.Module, backb, low_channels: int, high_channels: int, num_classes: int, inter_channels: int = 128,
        skip_num=48, criterion=None, criterion_aux=None, cont_proj_head=None, 
        wild_cont_dict_size=None, variant='D16', skip='m1') -> None:
        super().__init__()
        
        self.backbone = backbone
        self.backb = backb
        self.classifier = LRASPPHead(low_channels, high_channels, num_classes, inter_channels)
        
        self.args = args

        # loss
        self.criterion = criterion
        self.criterion_aux = criterion_aux
        self.criterion_kl = nn.KLDivLoss(reduction='batchmean').cuda()

        # create the wild-content dictionary
        self.cont_proj_head = cont_proj_head
        if wild_cont_dict_size > 0:
            if cont_proj_head > 0:
                self.cont_dict = {}
                self.cont_dict['size'] = wild_cont_dict_size
                self.cont_dict['dim'] = self.cont_proj_head

                self.register_buffer("wild_cont_dict", torch.randn(self.cont_dict['dim'], self.cont_dict['size']))
                self.wild_cont_dict = nn.functional.normalize(self.wild_cont_dict, p=2, dim=0) # C X Q
                self.register_buffer("wild_cont_dict_ptr", torch.zeros(1, dtype=torch.long))
                self.cont_dict['wild'] = self.wild_cont_dict.cuda()
                self.cont_dict['wild_ptr'] = self.wild_cont_dict_ptr
            else:
                raise 'dimension of wild-content dictionary is zero'
                
        self.layer0 = self.backbone.features[0]
        self.layer1, self.layer2, self.layer3 = self.backbone.features[1], self.backbone.features[2], self.backbone.features[3]

        if self.cont_proj_head > 0:
            self.proj = nn.Sequential(
                nn.Linear(256, 256, bias=True),
                nn.ReLU(inplace=False),
                nn.Linear(256, self.cont_proj_head, bias=True))
            initialize_weights(self.proj)

        self.dsn = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )
        initialize_weights(self.dsn)
            
        # Setting the flags
        self.eps = 1e-5
        self.whitening = False

        
    def forward(self, input: Tensor, gts=None, aux_gts=None, x_w=None, apply_fs=False) -> Dict[str, Tensor]:
        
        x_size = input.size()
        
        # features = self.backbone(input)
        
        # encoder
        x = self.layer0[0](input)
        if self.training & apply_fs:
            with torch.no_grad():
                x_w = self.layer0[0](x_w)
                
        x = self.layer0[1](x)
        if self.training & apply_fs:
            x_sw = self.layer0[1](x, x_w) # feature stylization
            with torch.no_grad(): 
                x_w = self.layer0[1](x_w)
                
        x = self.layer0[2](x)
        if self.training & apply_fs:
            with torch.no_grad():
                x_w = self.layer0[2](x_w)
            x_sw = self.layer0[2](x_sw)
        
        if self.training & apply_fs:
            x_tuple = self.layer1([x, x_w, x_sw])
            low_level = x_tuple[0]
            low_level_w = x_tuple[1]
            low_level_sw = x_tuple[2]
        else:
            x_tuple = self.layer1([x])
            low_level = x_tuple[0]
        
        x_tuple = self.layer2(x_tuple)
        x_tuple = self.layer3(x_tuple)
        aux_out = x_tuple[0]
        if self.training & apply_fs:
            aux_out_w = x_tuple[1]
            aux_out_sw = x_tuple[2]
        
        for i in range(4,17):
            x_tuple = self.backbone.features[i](x_tuple)
        
        x = x_tuple[0]
        if self.training & apply_fs:
            x_w = x_tuple[1]
            x_sw = x_tuple[2]
            
        out, out_proj = self.classifier(self.backb(input))
        main_out = F.interpolate(out, size=input.shape[-2:], mode="bilinear", align_corners=False)
        
        if self.training:
            # compute original semantic segmentation loss
            loss_orig = self.criterion(main_out, gts)
            aux_out = self.dsn(aux_out)
            if aux_gts.dim() == 1:
                aux_gts = gts
            aux_gts = aux_gts.unsqueeze(1).float()
            aux_gts = nn.functional.interpolate(aux_gts, size=aux_out.shape[2:], mode='nearest')
            aux_gts = aux_gts.squeeze(1).long()
            loss_orig_aux = self.criterion_aux(aux_out, aux_gts)

            return_loss = [loss_orig, loss_orig_aux]

            if apply_fs:
                out_sw, out_proj_sw = self.classifier(self.backb(input))
                main_out_sw = F.interpolate(out_sw, size=input.shape[-2:], mode="bilinear", align_corners=False)
                
                with torch.no_grad():
                    out_w, out_proj_w = self.classifier(self.backb(input))
                    main_out_w = F.interpolate(out_w, size=input.shape[-2:], mode="bilinear", align_corners=False)
                
                if self.args.use_cel:
                    # projected features
                    assert (self.cont_proj_head > 0)
                    proj2 = self.proj(out_proj.permute(0,2,3,1)).permute(0,3,1,2)
                    proj2_sw = self.proj(out_proj_sw.permute(0,2,3,1)).permute(0,3,1,2)
                    with torch.no_grad():
                        proj2_w = self.proj(out_proj_w.permute(0,2,3,1)).permute(0,3,1,2)

                    # compute content extension learning loss
                    loss_cel = get_content_extension_loss(proj2, proj2_sw, proj2_w, gts, self.cont_dict)

                    return_loss.append(loss_cel)
                
                if self.args.use_sel:
                    # compute style extension learning loss
                    loss_sel = self.criterion(main_out_sw, gts)
                    aux_out_sw = self.dsn(aux_out_sw)
                    loss_sel_aux = self.criterion_aux(aux_out_sw, aux_gts)
                    return_loss.append(loss_sel)
                    return_loss.append(loss_sel_aux)
                
                if self.args.use_scr:
                    # compute semantic consistency regularization loss
                    loss_scr = torch.clamp((self.criterion_kl(nn.functional.log_softmax(main_out_sw, dim=1), 
                                                              nn.functional.softmax(main_out, dim=1)))/(
                                   torch.prod(torch.tensor(main_out.shape[1:]))), min=0)
                    loss_scr_aux = torch.clamp((self.criterion_kl(nn.functional.log_softmax(aux_out_sw, dim=1),
                                                                  nn.functional.softmax(aux_out, dim=1)))/(
                                       torch.prod(torch.tensor(aux_out.shape[1:]))), min=0)
                    return_loss.append(loss_scr)
                    return_loss.append(loss_scr_aux)

            return return_loss
        
        else:
            return main_out, self.backb(input)

class LRASPPHead(nn.Module):
    def __init__(self, low_channels: int, high_channels: int, num_classes: int, inter_channels: int) -> None:
        super().__init__()
        self.cbr = nn.Sequential(
            nn.Conv2d(high_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
        )
        self.scale = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(high_channels, inter_channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.low_classifier = nn.Conv2d(low_channels, num_classes, 1)
        self.high_classifier = nn.Conv2d(inter_channels, num_classes, 1)

    def forward(self, input: Dict[str, Tensor]) -> Tensor:
        low = input["low"]
        high = input["high"]

        x = self.cbr(high)
        s = self.scale(high)
        x = x * s
        x = F.interpolate(x, size=low.shape[-2:], mode="bilinear", align_corners=False)

        return [self.low_classifier(low) + self.high_classifier(x), x]


def _lraspp_mobilenetv3(args, backbone: MobileNetV3, num_classes: int, criterion=None, criterion_aux=None, 
                        cont_proj_head=None, wild_cont_dict_size=None,
                        variant='D16', skip='m1') -> LRASPP:
    backb = backbone.features
    # Gather the indices of blocks which are strided. These are the locations of C1, ..., Cn-1 blocks.
    # The first and last blocks are always included because they are the C0 (conv1) and Cn.
    stage_indices = [0] + [i for i, b in enumerate(backb) if getattr(b, "_is_cn", False)] + [len(backb) - 1]
    low_pos = stage_indices[-4]  # use C2 here which has output_stride = 8
    high_pos = stage_indices[-1]  # use C5 which has output_stride = 16
    low_channels = backb[low_pos].out_channels
    high_channels = backb[high_pos].out_channels
    backb = IntermediateLayerGetter(backb, return_layers={str(low_pos): "low", str(high_pos): "high"})

    return LRASPP(args, backbone, backb, low_channels, high_channels, num_classes, criterion=criterion, criterion_aux=criterion_aux, 
                  cont_proj_head=cont_proj_head, wild_cont_dict_size=wild_cont_dict_size, variant='D16', skip='m1')



def lraspp_mobilenet_v3_large(
    *,
    args,
    criterion, 
    criterion_aux, 
    cont_proj_head, 
    wild_cont_dict_size,
    pretrained: bool = True,
    progress: bool = True,
    num_classes: Optional[int] = 1,
    **kwargs: Any) -> LRASPP:
    
    """Constructs a Lite R-ASPP Network model with a MobileNetV3-Large backbone from
    `Searching for MobileNetV3 <https://arxiv.org/abs/1905.02244>`_ paper.

    .. betastatus:: segmentation module

    Args:
        weights (:class:`~torchvision.models.segmentation.LRASPP_MobileNet_V3_Large_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.segmentation.LRASPP_MobileNet_V3_Large_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        num_classes (int, optional): number of output classes of the model (including the background).
        aux_loss (bool, optional): If True, it uses an auxiliary loss.
        weights_backbone (:class:`~torchvision.models.MobileNet_V3_Large_Weights`, optional): The pretrained
            weights for the backbone.
        **kwargs: parameters passed to the ``torchvision.models.segmentation.LRASPP``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/lraspp.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.segmentation.LRASPP_MobileNet_V3_Large_Weights
        :members:
    """
    
    if kwargs.pop("aux_loss", False):
        raise NotImplementedError("This model does not use auxiliary loss")

    elif num_classes is None:
        num_classes = 1

    backbone = mobilenet_v3_large(pretrained=pretrained, dilated=True, fs_layer=args.fs_layer)
    model = _lraspp_mobilenetv3(args, backbone, num_classes=1000, criterion=criterion, criterion_aux=criterion_aux, 
                                cont_proj_head=cont_proj_head, wild_cont_dict_size=wild_cont_dict_size,
                                variant='D16', skip='m1')

    return model