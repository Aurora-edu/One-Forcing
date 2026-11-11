"""CPU scientific/protocol tests; synthetic models are never paper results."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from experiments.iclr import progressive as launcher
from experiments.iclr import progressive_protocol as protocol
from experiments.iclr.progressive_core import (GRIDS, flow_step, two_step_target,
    distillation_loss, stage_times, sample_latents)
from experiments.iclr.progressive_export import modes_for, video_path, write_video
from experiments.iclr.protocol import atomic_json, sha256_file


@pytest.mark.parametrize('stage,update',[('4to2',1),('4to2',2),('2to1',1)])
def test_collapsed_target_reaches_two_teacher_step_endpoint(stage,update):
    torch.manual_seed(7)
    x = torch.randn(2,3,4)
    calls = []
    def teacher(x,t):
        calls.append(t)
        return .17*x.sin() + t*.23
    _,times = stage_times(stage,update)
    rng = torch.get_rng_state()
    target,destination = two_step_target(teacher,x,*times)
    assert calls == list(times[:2])
    assert torch.equal(rng,torch.get_rng_state())
    torch.testing.assert_close(flow_step(x,target,times[0],times[2]),destination,atol=1e-6,rtol=1e-6)
    assert not target.requires_grad
    if times[2] != 0:
        assert not torch.allclose(target,teacher(x,times[1]))


@pytest.mark.parametrize('times',[(1,1,0),(1,.5,.5),(.5,.8,0),(1,.5,-1)])
def test_invalid_teacher_times_fail(times):
    with pytest.raises(ValueError): two_step_target(lambda x,t:x,torch.ones(1),*times)


def test_student_has_gradients_and_teacher_is_frozen():
    student,teacher = torch.nn.Linear(3,3),torch.nn.Linear(3,3)
    clean,noise = torch.randn(4,3),torch.randn(4,3)
    loss,_ = distillation_loss(lambda x,t:student(x),lambda x,t:teacher(x),clean,noise,'4to2',1)
    loss.backward()
    assert all(p.grad is None for p in teacher.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in student.parameters())
    assert student.weight.grad.abs().sum() > 0


def test_synthetic_two_stage_optimization_is_finite_and_parent_is_frozen():
    teacher = torch.nn.Linear(3,3)
    torch.manual_seed(2026)
    clean,noise = torch.randn(5,3),torch.randn(5,3)
    for stage in ('4to2','2to1'):
        student = copy.deepcopy(teacher)
        before = copy.deepcopy(teacher.state_dict())
        optimizer = torch.optim.AdamW(student.parameters(),lr=2e-6,betas=(.9,.999),weight_decay=.01)
        for step in range(1,301):
            optimizer.zero_grad()
            loss,_ = distillation_loss(lambda x,t:student(x),lambda x,t:teacher(x),clean,noise,stage,step)
            assert torch.isfinite(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(),10)
            optimizer.step()
        assert all(torch.equal(before[k],v) for k,v in teacher.state_dict().items())
        assert any(not torch.equal(before[k],v) for k,v in student.state_dict().items())
        teacher = copy.deepcopy(student).requires_grad_(False)
        # The next student is explicitly trainable, like the real wrapper build(True).
        teacher.requires_grad_(True)


class ToyPipeline:
    frame_seq_length = 1
    def __init__(self):
        self.calls = []
        self.generator = self
    def eval(self): return self
    def text_encoder(self,**kwargs): return {}
    def _initialize_kv_cache(self,*args): self.kv_cache1 = []
    def _initialize_crossattn_cache(self,*args): self.crossattn_cache = []
    def __call__(self,**kw):
        t = kw['timestep'].item()/1000
        x = kw['noisy_image_or_video']
        self.calls.append((kw['current_start'],t,x.clone()))
        return x, .25*x + .1


@pytest.mark.parametrize('steps',[1,2,4])
def test_uniform_nfe_no_ffe_and_generated_clean_cache(steps):
    pipeline = ToyPipeline()
    noise = torch.tensor([1.,2.,3.]).view(1,3,1)
    rng = torch.get_rng_state()
    result = sample_latents(pipeline,noise,['toy'],steps)
    assert torch.equal(rng,torch.get_rng_state())
    assert len(pipeline.calls) == 3*(steps+1)
    for frame in range(3):
        calls = pipeline.calls[frame*(steps+1):(frame+1)*(steps+1)]
        assert [x[0] for x in calls] == [frame]*(steps+1)
        assert [x[1] for x in calls[:-1]] == pytest.approx(GRIDS[steps][:-1])
        assert calls[-1][1] == 0
        torch.testing.assert_close(calls[-1][2],result[:,frame:frame+1])
    torch.testing.assert_close(result,sample_latents(ToyPipeline(),noise,['toy'],steps))


def test_legacy_sampler_is_only_available_for_source_all4():
    noise = torch.ones(1,2,1)
    rng = torch.get_rng_state()
    sample_latents(ToyPipeline(),noise,['toy'],4,legacy_renoise=True)
    assert not torch.equal(rng,torch.get_rng_state())
    with pytest.raises(ValueError): sample_latents(ToyPipeline(),noise,['toy'],1,legacy_renoise=True)


@pytest.fixture
def source_spec(tmp_path):
    initial = tmp_path/'init.pt'
    torch.save({'generator':{'model.w':torch.ones(2)}},initial)
    manifest = tmp_path/'trajectory_manifest.json'
    atomic_json(manifest,{'generator_checkpoint_sha256':sha256_file(initial),'use_ema':False,
                         'guidance_scale':6.,'latents_shape':[6505,6,21,16,60,104]})
    spec = {'origin':'historical_four_step_cd','initial_checkpoint':str(initial),'trajectory_manifest':str(manifest)}
    cfg = {'seed':2026,'max_steps':300,'batch_size':1,'training_num_frames':9,'lr':2e-6,
           'beta1':.9,'beta2':.999,'weight_decay':.01,'max_grad_norm':10.,'target_ema_decay':.95,'num_frame_per_block':1}
    for arm,intervention in [('original','curved'),('rectified','rectified')]:
        meta = {'experiment':protocol.SOURCE_EXPERIMENT,'intervention':intervention,'seed':2026,
            'training_num_frames':9,'dataset_manifest_sha256':sha256_file(manifest),
            'initial_generator_sha256':sha256_file(initial),'training_objective':'adjacent_state_consistency_l2',
            'target_ema_decay':.95,'world_size':8,'framewise':True,'evaluation_use_ema':False,
            'training_timesteps':[1000,937.5,1000*5/6,625,0]}
        path = tmp_path/f'{arm}.pt'
        torch.save({'step':300,'generator':{'model.w':torch.ones(2)},'experiment_metadata':meta},path)
        intent = tmp_path/f'{arm}.intent.json'
        atomic_json(intent,{**meta,'config':cfg})
        spec[arm] = {'checkpoint':str(path),'sha256':sha256_file(path),'training_record':str(intent)}
    return spec


def test_source_provenance_audit_is_not_a_quality_pass(source_spec):
    audited = protocol.audit_source_pair(source_spec)
    assert audited['quality_status'].startswith('unverified')
    assert set(audited['arms']) == {'original','rectified'}


@pytest.mark.parametrize('mutation',['1000','endpoint','local','ema','initial','world','times','arm'])
def test_unrelated_sources_are_rejected(source_spec,mutation):
    path = Path(source_spec['rectified']['checkpoint'])
    payload = torch.load(path,weights_only=True)
    meta = payload['experiment_metadata']
    if mutation == '1000': payload['step'] = 1000
    if mutation == 'endpoint': meta['training_objective'] = 'adjacent_cd_plus_endpoint_l2'
    if mutation == 'local': meta['rectification_treatment'] = 'high_noise_local_progressive'
    if mutation == 'ema': payload['generator_ema'] = payload.pop('generator')
    if mutation == 'initial': meta['initial_generator_sha256'] = 'bad'
    if mutation == 'world': meta['world_size'] = 4
    if mutation == 'times': meta['training_timesteps'] = [1000,750,500,250,0]
    if mutation == 'arm': meta['intervention'] = 'curved'
    torch.save(payload,path)
    source_spec['rectified']['sha256'] = sha256_file(path)
    with pytest.raises(ValueError): protocol.audit_source_pair(source_spec)


def test_holdout_excludes_duplicate_prompts_and_intervals_are_balanced():
    prompts = [str(i) for i in range(80)] + ['0','60','64']
    rows,cases = protocol.partition_prompts(prompts)
    assert 80 not in rows and 81 not in rows and 82 in rows
    assert len(cases) == 16 and [x['row'] for x in cases] == list(range(0,64,4))
    assert [stage_times('4to2',i)[0] for i in range(1,301)].count(0) == 150
    assert [stage_times('4to2',i)[0] for i in range(1,301)].count(1) == 150


def test_review_requires_all_videos_and_discloses_frame_only_inspection(tmp_path):
    videos = []
    for arm in protocol.ARMS:
        path = tmp_path/f'{arm}.mp4'
        path.write_bytes(b'synthetic receipt fixture, not video')
        videos.append({'path':str(path),'sha256':sha256_file(path)})
    receipt = tmp_path/'previews/source/export.done.json'
    atomic_json(receipt,{'videos':videos})
    review = {'export_sha256':sha256_file(receipt),'dynamic_playback_pending':True,
        'inspected':[{'path':x['path'],'usable':True,'method':'temporal_frames','notes':'synthetic test only'} for x in videos]}
    target = tmp_path/'reviews/source.json'
    atomic_json(target,review)
    protocol.require_review(tmp_path,'source')
    for key,value in [('inspected',review['inspected'][:1]),('dynamic_playback_pending',False),('export_sha256','bad')]:
        atomic_json(target,{**review,key:value})
        with pytest.raises(ValueError): protocol.require_review(tmp_path,'source')
    atomic_json(target,review)
    Path(videos[0]['path']).write_bytes(b'changed')
    with pytest.raises(ValueError): protocol.require_review(tmp_path,'source')


def test_stage_checkpoint_cannot_cross_arm_or_parent(tmp_path):
    m = {'data_sha256':'data','config':OmegaConf.to_container(OmegaConf.load(protocol.CONFIG))}
    meta = {'profile':protocol.PROFILE,'arm':'original','stage':'4to2','seed':2026,'updates':300,
            'parent_sha256':'source','data_sha256':'data','config':m['config'],'use_ema':False}
    path = tmp_path/'stage.pt'
    torch.save({'step':300,'generator':{'w':torch.ones(2)},'experiment_metadata':meta},path)
    protocol.validate_stage_checkpoint(path,m,'original','4to2','source')
    for arm,stage,parent in [('rectified','4to2','source'),('original','2to1','source'),('original','4to2','other')]:
        with pytest.raises(ValueError): protocol.validate_stage_checkpoint(path,m,arm,stage,parent)


def test_commands_and_export_coverage_are_paired_all1(tmp_path):
    args = SimpleNamespace(output=str(tmp_path),action='train',stage='4to2',python='/train python',vbench_python='/vbpy')
    planned = launcher.commands(args,{})
    assert len(planned) == 2 and [x[-1] for x in planned] == list(protocol.ARMS)
    args.action = 'evaluate'
    planned = launcher.commands(args,{'full_info':'/original.json'})
    assert len(planned) == 3 and planned[0][-1] == 'final'
    for cmd in planned[1:]:
        assert cmd[cmd.index('--samples_per_prompt')+1] == '1'
        assert cmd[cmd.index('--mode')+1] == 'vbench_standard'
        assert 'qwen' not in ' '.join(cmd).lower()
    assert modes_for('source') == [('deterministic4',4,False),('legacy4',4,True)]
    assert modes_for('2to1') == modes_for('final') == [('all1',1,False)]
    assert video_path(tmp_path,'final','original','all1',{'filename':'a-0.mp4'}).name == 'a-0.mp4'


def test_synthetic_mp4_roundtrip_validates_all_frames_and_builds_temporal_sheet(tmp_path):
    # Encoding/receipt test only. No Wan visual quality claim.
    pixels = torch.zeros(81,480,832,3,dtype=torch.uint8)
    for i in range(81): pixels[i,:,:,0] = i*3
    path = tmp_path/'synthetic.mp4'
    result = write_video(path,pixels,True)
    assert result == {'frames':81,'fps':16.,'width':832,'height':480}
    assert path.with_suffix('.jpg').is_file()
    with pytest.raises(FileExistsError): write_video(path,pixels,True)


def test_prepare_locks_actual_original_prompts_and_holdout(source_spec,tmp_path,monkeypatch):
    import utils.dataset as dataset
    prompts = [f'holdout or training {i}' for i in range(80)] + ['holdout or training 0']
    class FakeDataset:
        def __init__(self,*args): self.env = SimpleNamespace(close=lambda:None)
        def __len__(self): return len(prompts)
        def __getitem__(self,i): return {'clean_latent':torch.empty(21,16,60,104)}
    monkeypatch.setattr(dataset,'CleanLatentLMDBDataset',FakeDataset)
    monkeypatch.setattr(dataset,'retrieve_row_from_lmdb',lambda env,name,dtype,i:prompts[i])
    spec = tmp_path/'spec.json'; atomic_json(spec,source_spec)
    data = tmp_path/'clean'; data.mkdir(); (data/'data.mdb').write_bytes(b'synthetic data')
    output = tmp_path/'experiment'
    manifest = protocol.prepare(spec,data,output)
    assert manifest['config']['training_num_frames'] == 9
    assert manifest['training_rows'] == list(range(64,80))
    assert len(manifest['vbench']) == 944
    assert manifest['vbench'][0]['seed'] == 0 and manifest['vbench'][-1]['seed'] == 943
    assert protocol.check(output) == manifest
    manifest['vbench'][0]['prompt'] = 'changed'
    atomic_json(output/'manifest.json',manifest)
    with pytest.raises(ValueError,match='manifest changed'): protocol.check(output)


def test_cli_dry_run_never_launches_or_queries_gpus(tmp_path,monkeypatch,capsys):
    import sys
    monkeypatch.setattr(launcher,'check',lambda root:{'full_info':'/original.json'})
    monkeypatch.setattr(launcher.subprocess,'run',lambda *a,**kw:pytest.fail('dry run launched a process'))
    monkeypatch.setattr(sys,'argv',['progressive','evaluate','--output',str(tmp_path/'fresh')])
    launcher.main()
    assert 'DRY RUN' in capsys.readouterr().out
    assert not (tmp_path/'fresh').exists()


def test_training_preflight_refuses_missing_source_review(tmp_path):
    args = SimpleNamespace(action='train',stage='4to2',output=str(tmp_path))
    with pytest.raises(FileNotFoundError): launcher.preflight(args,{})
    assert not (tmp_path/'training').exists()
