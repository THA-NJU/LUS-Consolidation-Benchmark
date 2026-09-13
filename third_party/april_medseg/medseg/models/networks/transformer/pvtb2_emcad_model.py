"""PVTv2-B2 + EMCAD: complete end-to-end segmentation network.
    PVTv2-B 2 + EMCAD: complete end-to-end 分割 网络。

中文: PVTv2-B2编码器 + EMCAD解码器的完整分割网络。

Bundles the PVTv2-B2 backbone (via timm) with the EMCAD decoder
(Efficient Multi-scale Convolutional Attention Decoding, CVPR 2024)
to provide a convenient single-architecture entry point equivalent to
the ``timm_pvt_v2_b2`` + ``emcad`` combination in ``configs/combinations/``.

Architecture overview:
    - Encoder: PVTv2-B2 producing 4 multi-scale features at strides
      {4, 8, 16, 32} with channels {64, 128, 320, 512}.
    - Decoder: EMCAD with LGAG gating + MSCB + MSCAM attention.
    - Final: bilinear upsample + 1x1 segmentation head.

Reference:
    EMCAD: https://github.com/SLDGroup/EMCAD (CVPR 2024)
    PVTv2: https://github.com/whai362/PVT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm

from medseg.models.networks.sam.sam_base import load_with_ssl_fallback


class PVTB2EMCAD(nn.Module):
    """PVTv2-B2 + EMCAD end-to-end segmentation network.
        PVTv2-B 2 + EMCAD end-to-end 分割 网络。

    中文: PVTv2-B2编码器 + EMCAD解码器医学图像分割网络。

    Args:
        in_channels: number of input image channels.
        num_classes: number of segmentation classes.
        img_size: nominal input spatial size (not enforced; forward is
            fully convolutional).
    """

    def __init__(self, in_channels=3, num_classes=2, img_size=224,
                 pretrained=True, deep_supervision=False, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.img_size = img_size
        self.deep_supervision = deep_supervision

        # - - - 骨干网络: PVTv2-B 2 via timm - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - / --- Backbone: PVTv2-B2 via timm -----------------------------------
        def _create_backbone(pretrained):
            return timm.create_model(
                'pvt_v2_b2',
                features_only=True,
                pretrained=pretrained,
                in_chans=in_channels,
            )

        self.backbone = load_with_ssl_fallback(_create_backbone, pretrained=pretrained)

        enc_channels = self.backbone.feature_info.channels()
        if len(enc_channels) != 4:
            raise RuntimeError(
                f'Expected 4 encoder features, got {len(enc_channels)}: {enc_channels}'
            )
        # PVTv2-B2: c1=64(/4), c2=128(/8), c3=320(/16), c4=512(/32)
        skip_channels = enc_channels[:-1]   # [64, 128, 320]
        bottleneck_ch = enc_channels[-1]    # 512

        # --- EMCAD Decoder --------------------------------------------------
        from medseg.models.decoders.cascade.emcad_decoder import EMCADDecoder
        self.decoder = EMCADDecoder(
            encoder_channels=skip_channels,
            bottleneck_channels=bottleneck_ch,
            deep_supervision=deep_supervision,
        )

        out_ch = self.decoder.out_channels  # shallowest skip = 64
        self.head = nn.Conv2d(out_ch, num_classes, kernel_size=1)

        # 深度 supervision heads / Deep supervision heads
        if deep_supervision and self.decoder.ds_channels:
            self.ds_heads = nn.ModuleList([
                nn.Conv2d(ch, num_classes, 1) for ch in self.decoder.ds_channels
            ])

    def forward(self, x):
        H, W = x.shape[-2:]
        features = self.backbone(x)  # [f1(/4), f2(/8), f3(/16), f4(/32)]

        skip_features = features[:-1]   # [f1, f2, f3]
        bottleneck = features[-1]       # f4

        dec_out = self.decoder(bottleneck, skip_features)

        if self.training and self.deep_supervision and isinstance(dec_out, tuple):
            d, intermediates = dec_out
            logits = self.head(d)
            logits = F.interpolate(logits, size=(H, W), mode='bilinear',
                                   align_corners=False)
            aux = [
                F.interpolate(h(f), size=logits.shape[2:],
                              mode='bilinear', align_corners=False)
                for f, h in zip(intermediates, self.ds_heads)
            ]
            return [logits] + aux

        logits = self.head(dec_out)
        logits = F.interpolate(logits, size=(H, W), mode='bilinear',
                               align_corners=False)
        return logits
