from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset


@DATASETS.register_module()
class CLOUDSEN12HIGHL1CDataset(BaseSegDataset):
    METAINFO = dict(
        classes=("clear", "thick cloud", "thin cloud", "cloud shadow"),
        palette=[
            [0, 0, 0],
            [255, 255, 255],
            [170, 170, 170],
            [85, 85, 85],
        ],
    )

    def __init__(
        self,
        img_suffix=".png",
        seg_map_suffix=".png",
        reduce_zero_label=False,
        **kwargs
    ) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs
        )
