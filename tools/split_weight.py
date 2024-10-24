import torch

weight_path = "work_dirs/head_dinov2_l_mask2former_gf2/best_mIoU_iter_8000.pth"

weight:dict = torch.load(weight_path)['state_dict']

print(weight.keys())
key_word = "decode_head."


decode_state_dict = dict()
for key,val in weight.items():
    key:str = key
    if key.startswith(key_word):
        decode_state_dict[key] = val

torch.save(decode_state_dict,"decode_head.pt")