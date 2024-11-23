from .dino_v2 import DinoVisionTransformer
from .reins_dinov2 import ReinsDinoVisionTransformer
from .reins_eva_02 import ReinsEVA2
from .reins_resnet import ReinsResNetV1c
from .my_rein_dinov2 import MyReinsDinoVisionTransformer
from .myreinstoken import MyReinsToken
from .rein_token_divo2 import ReinsTokenDinoVisionTransformer
from .myrein_tonken_mlp import MyReinsTokenMlp
from .my_rein_token_mlp_dinov2 import MyReinTokenDinoVisionTransformer
from .loracacheadapter import LoRACacheAdapter
from .vitadapter_dinov2 import ViTAdapter
from .cnnadapter import CNNAdapter
from .cnnadapter_dinov2 import CNNAdapterDinoVisionTransformer
from .pmaaadapter import PMAAAdapter
from .pmaaadapter_dinov2 import PMAAAdapterDinoVisionTransformer
from .cloud_adapter import CloudAdapter
from .cloud_adapter_dinov2 import CloudAdapterDinoVisionTransformer
try:
    from .reins_convnext import ReinsConvNeXt
except:
    print('Fail to import ReinsConvNeXt, if you need to use it, please install mmpretrain')
from .clip import CLIPVisionTransformer
from .reins_sam_vit import ReinsSAMViT
from .sam_vit import SAMViT
from .reins_clip import ReinsCLIPVisionTransformer
from .convnext_dinov2 import ConvnextDinoVisionTransformer
from .loracacheadapter_dinov2 import LoRACacheAdapterDinoVisionTransformer
from .cloud_adapter_sam import CloudAdapterSamVisionTransformer