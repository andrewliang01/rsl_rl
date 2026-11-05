"""Explicit perception LR overrides must survive an actual optimizer resume."""
from copy import deepcopy

import pytest
import torch

from rsl_rl.algorithms.range_task_bridge_amp_ppo import RangeTaskBridgeAMPPPO
from test_range_jepa_amp_ppo import TinyEnv
from test_range_task_bridge import configuration, collect, assert_tree_equal


def construct(request='absent'):
    config=configuration()
    if request!='absent':config['algorithm']['perception_cfg']['learning_rate']=request
    env=TinyEnv()
    alg=RangeTaskBridgeAMPPPO.construct_algorithm(env.get_observations(),env,config,'cpu')
    return env,alg


@pytest.mark.parametrize('value',[0.,-1.,float('nan'),float('inf'),True,False,'0.0005'])
def test_invalid_explicit_perception_rate_is_rejected(value):
    with pytest.raises(ValueError,match='learning rate'):
        construct(value)


@pytest.mark.parametrize('requested_lr',['absent',None,.0005])
def test_resume_preserves_all_adam_state_except_explicit_lr(requested_lr):
    torch.manual_seed(9)
    env,source=construct()
    collect(env,source);source.update()
    source.perception_optimizer.param_groups[0]['lr']=.00037
    saved=deepcopy(source.save())
    before=deepcopy(saved['perception_optimizer_state_dict'])
    assert before['state'], 'Test requires real nonempty Adam moments.'
    _,target=construct(requested_lr)
    target.load(saved)
    actual=deepcopy(target.perception_optimizer.state_dict())
    expected=deepcopy(before)
    if requested_lr==.0005:expected['param_groups'][0]['lr']=.0005
    assert_tree_equal(actual,expected)
    assert_tree_equal(saved['perception_optimizer_state_dict'],before)
    assert_tree_equal(target.optimizer.state_dict(),saved['optimizer_state_dict'])
    assert_tree_equal(target.actor.state_dict(),saved['actor_state_dict'])
    assert_tree_equal(target.perception.state_dict(),saved['perception_state_dict'])
    if requested_lr==.0005:
        assert target.last_perception_receipt['perception_learning_rate_source']=='explicit_launch_override_after_optimizer_load'


def test_new_saved_rate_is_inherited_without_an_override():
    env,source=construct()
    collect(env,source);source.update()
    _,changed=construct(.0005)
    changed.load(deepcopy(source.save()))
    saved=deepcopy(changed.save())
    assert saved['perception_cfg']['learning_rate']==.0005
    _,resumed=construct()
    resumed.load(saved)
    assert resumed.perception_optimizer.param_groups[0]['lr']==.0005
    assert_tree_equal(resumed.perception_optimizer.state_dict(),saved['perception_optimizer_state_dict'])
