import torch
import jittor as jt
import pickle

state_dict = torch.load('./pack/superglue_cyberfeat_1800_nd_100.pth')
dict_numpy = {k: v.cpu().numpy() for k, v in state_dict.items()}
# torch.save(dict_numpy, './pack/superglue_cyberfeat_1800_nd_100.pkl')
with open('./pack/superglue_cyberfeat_1800_nd_100.pkl', 'wb') as f:
    pickle.dump(dict_numpy, f)
a = jt.load('./pack/superglue_cyberfeat_1800_nd_100.pkl')

state_dict2 = jt.load('./pack/superglue_cyberfeat_1800_79.pkl')

temp1 = [x for x in a.keys() if x not in state_dict2.keys()]
temp2 = [x for x in state_dict2.keys() if x not in a.keys()]
print(temp1, temp2)