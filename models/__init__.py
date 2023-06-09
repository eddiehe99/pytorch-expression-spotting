from models.soft_net import SOFTNet

# from models.soft_net_cbam import SOFTNetCBAM
from models.vision_transformer import ViT
from models.sl_vision_transformer import SLViT
from models.swin_transformer import SwinTransformer
from models.l_swin_transformer import LSwinTransformer
from models.s_swin_transformer import SSwinTransformer
from models.sl_swin_transformer import SLSwinTransformer


def load_model(model_name):
    if model_name == "SOFTNet":
        model = SOFTNet()
    # elif model_name == "SOFTNetCBAM":
    #     model = SOFTNetCBAM()
    elif model_name == "ViT-B":
        model = ViT(
            img_size=42,
            patch_size=6,
            num_hiddens=768,
            mlp_num_hiddens=3072,
            num_heads=12,
            num_blks=12,
            emb_dropout=0.1,
            blk_dropout=0.1,
            num_classes=1,
        )
    elif model_name == "SL-ViT-B":
        model = SLViT(
            image_size=42,
            patch_size=6,
            num_classes=1,
            dim=768,
            depth=12,
            heads=12,
            mlp_dim_ratio=4,
            channels=3,
            dim_head=16,
            dropout=0.1,
            emb_dropout=0.1,
            stochastic_depth=0.0,
            is_LSA=True,
            is_SPT=True,
        )
    elif model_name == "Swin-T":
        model = SwinTransformer(
            patch_size=6,
            window_size=7,
            embed_dim=96,
            depths=(2, 2, 6, 2),
            num_heads=(3, 6, 12, 24),
            num_classes=1,
        )
    elif model_name == "Swin-S":
        model = SwinTransformer(
            patch_size=6,
            window_size=7,
            embed_dim=96,
            depths=(2, 2, 18, 2),
            num_heads=(3, 6, 12, 24),
            num_classes=1,
        )
    elif model_name == "L-Swin-T":
        model = LSwinTransformer(
            image_size=42,
            patch_size=6,
            window_size=7,
            embed_dim=96,
            depths=(2, 2, 6, 2),
            num_heads=(3, 6, 12, 24),
            num_classes=1,
        )
    elif model_name == "S-Swin-T":
        model = SSwinTransformer(
            image_size=42,
            patch_size=6,
            window_size=7,
            embed_dim=96,
            depths=(2, 2, 6, 2),
            num_heads=(3, 6, 12, 24),
            num_classes=1,
        )
    elif model_name == "SL-Swin-T":
        model = SLSwinTransformer(
            image_size=42,
            patch_size=6,
            window_size=7,
            embed_dim=96,
            depths=(2, 2, 6, 2),
            num_heads=(3, 6, 12, 24),
            num_classes=1,
        )
    elif model_name == "SL-Swin-S":
        model = SLSwinTransformer(
            image_size=42,
            patch_size=6,
            window_size=7,
            embed_dim=96,
            depths=(2, 2, 18, 2),
            num_heads=(3, 6, 12, 24),
            num_classes=1,
        )
    return model
