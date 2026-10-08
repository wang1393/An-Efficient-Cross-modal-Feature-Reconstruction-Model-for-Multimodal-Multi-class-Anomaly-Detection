"""AdamW with the existing bias/norm weight-decay grouping."""
from torch.optim import AdamW

def check_keywords_in_name(name, keywords=()):
	isin = False
	for keyword in keywords:
		if keyword in name:
			isin = True
	return isin


def add_weight_decay(model, weight_decay=1e-5, skip_list=(), skip_keywords=()):
	decay = []
	no_decay = []
	for name, param in model.named_parameters():
		if not param.requires_grad:
			continue  # frozen weights
		if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or check_keywords_in_name(name, skip_keywords):
			no_decay.append(param)
		else:
			decay.append(param)
	return [
		{'params': no_decay, 'weight_decay': 0.},
		{'params': decay, 'weight_decay': weight_decay}]


def get_optim(optim_kwargs, net, lr, betas=None, filter_bias_and_bn=True):
    kwargs = dict(optim_kwargs)
    name = kwargs.pop('name').lower()
    if name != 'adamw':
        raise ValueError(f'ECFR uses AdamW; got {name!r}')
    if kwargs.get('weight_decay') and filter_bias_and_bn:
        skip = net.no_weight_decay() if hasattr(net, 'no_weight_decay') else ()
        keywords = net.no_weight_decay_keywords() if hasattr(net, 'no_weight_decay_keywords') else ()
        params = add_weight_decay(net, kwargs['weight_decay'], skip, keywords)
        kwargs['weight_decay'] = 0.
    else:
        params = net.parameters()
    if kwargs.get('betas') and betas:
        kwargs['betas'] = betas
    return AdamW(params, lr=lr, **kwargs)
