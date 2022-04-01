import pickle
with open('./pack/superglue_cyberfeat_1800_nd_100.pkl', 'rb') as fin:
    state_dict = pickle.load(fin)

for k, v in state_dict.items():
    if len(v.shape) == 3 and v.shape[2] == 1:
        state_dict[k] = v[:, :, 0]

with open('./pack/superglue_cyberfeat_1800_nd_100_convopt1.pkl', 'wb') as fout:
    pickle.dump(state_dict, fout)