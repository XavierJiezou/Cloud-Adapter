import torch

weight_path = "checkpoints/dinov2_l_gf2_head.pth"

weight:dict = torch.load(weight_path,map_location="cuda:1")['state_dict']

# print(weight.keys())
# key_word = "decode_head."


# decode_state_dict = dict()
# for key,val in weight.items():
#     key:str = key
#     if key.startswith(key_word):
#         decode_state_dict[key] = val

torch.save(decode_state_dict,"checkpoints/dinov2_l_gf2_head.pth")