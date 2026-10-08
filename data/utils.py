from . import TRANSFORMS


def get_transforms(cfg, train, cfg_transforms):
	transform_list = []
	for t in cfg_transforms:
		t = {k: v for k, v in t.items()}
		t_type = t.pop('type')
		t_tran = TRANSFORMS.get_module(t_type)(**t)
		transform_list.extend(t_tran) if isinstance(t_tran, list) else transform_list.append(t_tran)
	transform_out = TRANSFORMS.get_module('Compose')(transform_list)

	return transform_out
