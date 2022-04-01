from pathlib import Path
from pprint import pformat
from hloc import extract_features_aslfeat, match_features_jittor

dataset = Path('/inloc/')  # change this if your dataset is somewhere else

pairs = Path('pairs/inloc/')
loc_pairs = pairs / 'pairs-query-netvlad40.txt'  # top 40 retrieved by NetVLAD

outputs = Path('/home/cscg/HL_output/inloc_aslfeat/')  # where everything will be saved


result_name = 'aslfeat+sg_netvlad40_jittor_nd100.txt'
results = outputs / result_name  # the result file

# list the standard configurations available
print(f'Configs for feature extractors:\n{pformat(extract_features_aslfeat.confs)}')
print(f'Configs for feature matchers:\n{pformat(match_features_jittor.confs)}')

# pick one of the configurations for extraction and matching
# you can also simply write your own here!
feature_conf = extract_features_aslfeat.confs['aslfeat_inloc']
matcher_conf = match_features_jittor.confs['superglue-aslfeat']

features0 = feature_conf['output']

feature_file0 = f"{features0}.h5"
match_file0 = f"{features0}_{matcher_conf['output']}_{loc_pairs.stem}.h5"

extract_features_aslfeat.main(feature_conf, dataset, outputs)

match_features_jittor.main(matcher_conf, loc_pairs, features0, outputs, dataset_path=dataset)


