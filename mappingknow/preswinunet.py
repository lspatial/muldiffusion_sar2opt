import torch
import torch.nn as nn
from .swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys
import copy

class preSwinUnet(nn.Module):

    def __init__(self, config,inchannel=None, num_classes=1,shortcut=True):

        super(preSwinUnet, self).__init__()
        self.num_classes = num_classes
        self.config = config
        self.swin_unet = SwinTransformerSys(img_size=config.DATA.IMG_SIZE,
                                            patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                            in_chans=config.MODEL.SWIN.IN_CHANS,
                                            num_classes=self.num_classes,
                                            embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                            depths=config.MODEL.SWIN.DEPTHS,
                                            num_heads=config.MODEL.SWIN.NUM_HEADS,
                                            window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                            qk_scale=config.MODEL.SWIN.QK_SCALE,
                                            drop_rate=config.MODEL.DROP_RATE,
                                            drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                            ape=config.MODEL.SWIN.APE,
                                            patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                            use_checkpoint=config.TRAIN.USE_CHECKPOINT)
        self.inalign = torch.nn.Conv2d(inchannel, config.MODEL.SWIN.IN_CHANS,
                                    kernel_size=3, padding=1, bias=True)
        self.in_batch_norm = nn.BatchNorm2d(config.MODEL.SWIN.IN_CHANS)
        self.in_act = nn.PReLU()
        self.shortcut = shortcut

    def forward(self, x):
        x = self.inalign(x)
        x = self.in_batch_norm(x)
        x = self.in_act(x)
        logits = self.swin_unet(x)
        if self.shortcut:
            logits += torch.sum(x, axis=1, keepdim=True)
        return logits

    def parameters2opt(self, mode):
        reparameters = (list(self.inalign.parameters()) + list(self.in_batch_norm.parameters())+
                        list(self.in_act.parameters()))
        if mode == 'all':
            reparameters += list(self.swin_unet.parameters())
        for param in reparameters: param.requires_grad = True
        total_params = sum(p.numel() for p in reparameters)
        print(f"Total parameters to be optimized: {total_params}")
        return reparameters

    def load_from(self, pretrained_path, device):
        """
         Load pretrained weights for Swin UNet with advanced handling.

            This method supports:
            - Loading from checkpoint files
            - Mapping layer names between different model architectures
            - Handling potential shape mismatches in model layers
            - Selective weight loading with non-strict state dict update

            Args:
                pretrained_path (str): Path to the pretrained model checkpoint
                device (torch.device): Device to load the model weights onto (CPU/CUDA)

            Key Steps:
            1. Load pretrained weights from file
            2. Handle different checkpoint formats
            3. Perform layer name remapping
            4. Filter out incompatible layers
            5. Update model state dictionary
        """
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model" not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]: v for k, v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.swin_unet.load_state_dict(pretrained_dict, strict=False)
                # print(msg)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.swin_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3 - int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k: v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k, v.shape, model_dict[k].shape))
                        del full_dict[k]
            msg = self.swin_unet.load_state_dict(full_dict, strict=False)
            # print(msg)
        else:
            print("none pretrain")
