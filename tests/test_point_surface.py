import torch
from tensordict import TensorDict
from rsl_rl.models.point_surface import PointSurfaceProjector,DilatedSurfaceCompleter,axis_rotation
from rsl_rl.models.point_surface_policy import PointSurfaceActor,PointSurfacePolicy,compact_points


def example():
    prop=torch.zeros(2,96);prop[:,5]=-1.
    xy=torch.cartesian_prod(torch.linspace(-1.1,1.1,16),torch.linspace(-1.1,1.1,16))
    level=torch.cat((xy,torch.full((len(xy),1),-1.23)),1)
    calibration=PointSurfaceProjector().calibration
    native=(level@calibration).expand(2,-1,-1).clone()
    return TensorDict(dict(policy=prop,range_obs=native,range_capture_id=torch.zeros(2,1,dtype=torch.long),
        range_episode_id=torch.zeros(2,1,dtype=torch.long),camp_terrain_targets=torch.randn(2,3,2,32,32),
        range_map=torch.randn(2,2,40,96)),batch_size=[2])


def test_gravity_and_waist_fk_preserve_metric_plane():
    projector=PointSurfaceProjector()
    prop=torch.zeros(4,96)
    angles=torch.tensor([0.,.1,-.15,.2])
    body=axis_rotation(angles,0)@axis_rotation(angles*.7,1)@axis_rotation(angles*2,2)
    up=torch.tensor([0.,0.,1.]).expand(4,-1)
    prop[:,3:6]=-(body.transpose(1,2)@up[...,None])[...,0]
    prop[:,18:21]=torch.tensor([[0.,0.,0.],[.2,.1,-.1],[-.3,-.1,.2],[.5,.2,.15]])
    q=prop[:,18:21]
    sensor=body@axis_rotation(q[:,0],2)@axis_rotation(q[:,1],0)@axis_rotation(q[:,2],1)@projector.calibration
    xy=torch.cartesian_prod(torch.linspace(-1.1,1.1,20),torch.linspace(-1.1,1.1,20))
    world=torch.cat((xy,torch.full((len(xy),1),-1.23)),1).expand(4,-1,-1)
    output=projector(world@sensor,prop)
    assert output[:,1].sum()>1000
    assert (output[:,0][output[:,1].bool()]+1.23).abs().max()<1e-5


def test_actor_cannot_read_privileged_targets_and_held_capture_does_not_advance():
    obs=example()
    compact=compact_points(obs,torch.zeros(2,3072))
    actor=PointSurfaceActor(compact,{'actor':['policy','range_features']},'actor',29)
    policy=PointSurfacePolicy(actor,DilatedSurfaceCompleter())
    first=policy(obs);cache=policy.features.clone()
    obs['camp_terrain_targets'].fill_(float('nan'));obs['range_map'].fill_(100000.)
    obs['range_obs']*=.8  # same capture identity: must retain original capture
    second=policy(obs)
    torch.testing.assert_close(first,second,rtol=0,atol=0)
    torch.testing.assert_close(cache,policy.features,rtol=0,atol=0)
    obs['range_episode_id'][0]+=1
    policy(obs)
    assert not torch.equal(cache[0],policy.features[0])
    torch.testing.assert_close(cache[1],policy.features[1],rtol=0,atol=0)


def test_measurements_survive_all_completion_parameters_and_partial_reset():
    obs=example();projector=PointSurfaceProjector();surface=projector(obs['range_obs'],obs['policy'])
    net=DilatedSurfaceCompleter()
    with torch.no_grad():
        for p in net.parameters():p.fill_(.1)
    output=net(surface);known=surface[:,1].bool()
    torch.testing.assert_close(output['height'][:,0][known],surface[:,0][known],rtol=0,atol=0)
    compact=compact_points(obs,torch.zeros(2,3072))
    actor=PointSurfaceActor(compact,{'actor':['policy','range_features']},'actor',29)
    policy=PointSurfacePolicy(actor,net);policy(obs);second=policy.features[1].clone()
    policy.reset(torch.tensor([True,False]))
    assert not policy.features[0].any()
    torch.testing.assert_close(second,policy.features[1],rtol=0,atol=0)
    assert torch.isfinite(net(torch.zeros_like(surface))['height']).all()


def test_nearest_projection_retains_subcell_location_and_matches_actual_anchors():
    from rsl_rl.models.point_surface import sensor_to_level
    from rsl_rl.models.point_evidence import PointEvidenceReducer
    prop=torch.zeros(2,96);prop[:,5]=-1.
    reducer=PointEvidenceReducer()
    level=torch.tensor([[[1.000,0.,-1.23],[1.035,0.,-1.08],[0.,0.,0.]],
                        [[1.000,0.,-1.23],[1.000,0.,-1.08],[float('nan'),0.,0.]]])
    rotation=sensor_to_level(prop,reducer.projector.calibration)
    native=level@rotation
    nearest=PointSurfaceProjector(projection='nearest')(native,prop)
    maximum=PointSurfaceProjector()(native,prop)
    evidence=reducer(native,prop)
    torch.testing.assert_close(nearest[:,0],evidence['anchors'][:,2,:,:,2])
    torch.testing.assert_close(nearest[:,1].bool(),evidence['observed'])
    torch.testing.assert_close(maximum[:,1],nearest[:,1])
    assert abs(float(nearest[0,0,16,29])+1.23)<1e-5
    assert abs(float(maximum[0,0,16,29])+1.08)<1e-5
    assert torch.isfinite(nearest).all()
    empty=PointSurfaceProjector(projection='nearest')(native[:,:0],prop)
    assert empty.shape==(2,2,32,32) and not empty.any()


def test_nearest_cached_policy_uses_no_privileged_observations():
    obs=example()
    compact=compact_points(obs,torch.zeros(2,3072))
    actor=PointSurfaceActor(compact,{'actor':['policy','range_features']},'actor',29)
    policy=PointSurfacePolicy(actor,DilatedSurfaceCompleter(),projection='nearest')
    first=policy(obs)
    obs['camp_terrain_targets'].fill_(float('nan'));obs['range_map'].fill_(float('nan'))
    torch.testing.assert_close(first,policy(obs),rtol=0,atol=0)
    known=policy.features.reshape(2,3,32,32)[:,1]>.5
    assert known.any()
    torch.testing.assert_close(policy.features.reshape(2,3,32,32)[:,0][known],
                               torch.full_like(policy.features.reshape(2,3,32,32)[:,0][known],-1.23),atol=1e-5,rtol=0)
