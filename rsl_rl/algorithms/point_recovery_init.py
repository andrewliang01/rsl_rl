"""Explicit recovery-only startup; never import learned control/AMP state."""
from pathlib import Path
import copy, json, os
import torch


def validate_recovery_only(path):
    saved=torch.load(Path(path),map_location='cpu',weights_only=False)
    required={'architecture','model_cfg','kinematic_contract','state_dict'}
    if saved.get('architecture')!='evidence' or not required.issubset(saved):
        raise ValueError('Recovery-only initialization requires a standalone evidence recovery checkpoint')
    forbidden={'actor_state_dict','critic_state_dict','critics_state_dict','point_surface_contract','point_control_contract','iter'}
    if forbidden.intersection(saved) or any('optimizer_state' in k or k.startswith('amp_') for k in saved):
        raise ValueError('Recovery-only startup may not contain Actor/critic/AMP/optimizer state; use explicit resume instead')
    if not saved['state_dict'] or not all(isinstance(v,torch.Tensor) for v in saved['state_dict'].values()):
        raise ValueError('Recovery weights must be a nonempty tensor state dictionary')
    if not all(torch.isfinite(v).all() for v in saved['state_dict'].values()):
        raise ValueError('Recovery weights contain nonfinite values')


def _clone(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:_clone(v) for k,v in value.items()}
    if isinstance(value,list):return [_clone(v) for v in value]
    if isinstance(value,tuple):return tuple(_clone(v) for v in value)
    return copy.deepcopy(value)


def _equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(_equal(v,b[k]) for k,v in a.items())
    if isinstance(a,(tuple,list)):return type(a)is type(b) and len(a)==len(b) and all(_equal(x,y) for x,y in zip(a,b))
    return a==b


def snapshot_core(algorithm):
    # Call AMP's core serializer directly, before any perception is ready.
    # This includes all critics, normalizers, AMP and both optimizers.
    from .amp_ppo import AMPPPO
    return _clone(AMPPPO.save(algorithm))


def verify_recovery_only_initialization(algorithm,before,source):
    after=snapshot_core(algorithm)
    if not _equal(before,after):raise RuntimeError('Recovery installation changed initial control/critic/AMP/optimizer state')
    optimizers={k:len(v['state']) for k,v in before.items() if k.endswith('optimizer_state_dict')}
    if not optimizers or any(optimizers.values()):raise RuntimeError('From-scratch control requires empty initial optimizer states')
    receipt=dict(source=str(Path(source).resolve()),recovery_only=True,actor_critic_optimizer_amp_loaded=False,
        initial_core_exactly_preserved=True,initial_optimizer_state_lengths=optimizers,
        perception_enabled=bool(algorithm.actor.perception_enabled),
        recovery_parameters=sum(p.numel() for p in algorithm.perception.parameters()),
        scope='Only pretrained frozen perception weights loaded. Actor/critics/normalizers/AMP freshlyconstructed; optimizer states empty. This is fromscratchCONTROL, not an all-networks-random experiment.')
    directory=os.environ.get('CAMP_POINT_INIT_AUDIT_DIR')
    if directory:
        folder=Path(directory);folder.mkdir(parents=True,exist_ok=True)
        rank=os.environ.get('RANK','0')
        target=folder/f'initial_core_rank{rank}.pt'
        if target.exists():raise RuntimeError('Initial-state audit already exists; do not overwrite a previous experiment')
        torch.save(before,target)
        (folder/f'initialization_rank{rank}.json').write_text(json.dumps(receipt,indent=2))
    return receipt
