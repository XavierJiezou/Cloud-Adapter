from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset


@DATASETS.register_module()
class L8BIOMEDataset(BaseSegDataset):
    METAINFO = dict(
        classes=("Clear", "Cloud Shadow", "Thin Cloud", "Cloud"),
        palette=[
            [0, 0, 0],
            [85, 85, 85],
            [170, 170, 170],
            [255, 255, 255],
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
