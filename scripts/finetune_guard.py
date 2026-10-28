"""Experimental mixed guard fine-tuning. Not an official SL/RL checkpoint format."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=Path.cwd())
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--device', default='cpu')
    p.add_argument('--seed', type=int, default=43)
    p.add_argument('--smoke-test', action='store_true')
    a = p.parse_args()
    if a.output.exists() or a.steps < 1 or a.batch_size < 2 or a.batch_size % 2 or a.lr <= 0:
        p.error('Use a new output directory, positive steps/lr, and even batch size >=2')
    repo = a.repo.resolve()
    sys.path.insert(0, str(repo))
    from holoboxing.training.export import load_policy, check_export_parity
    from holoboxing.training.supervised.statistics import normalized_loss
    from holoboxing.training.motion_prior import state_sha256
    from holoboxing.evaluation.runtime.model_contract import check_model, check_contract, CONTRACT
    import onnx
    import onnxruntime as ort

    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    prepared = Path((repo/'logs/prepared_data_path.txt').read_text().strip())
    manifest_path = prepared/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    fps = float(manifest.get('control_hz', 50))
    provenance = []

    def collect(kind):
        observations, targets = [], []
        for row in manifest['records']:
            name = Path(row.get('source', '')).name
            selected = ((kind == 'guard' and name == 'kangning-003_mirror.npz')
                or (kind == 'guard_val' and name == 'kangning-004_mirror.npz')
                or (kind == 'original' and row.get('dataset') == 'interhuman' and row.get('split') == 'train')
                or (kind == 'original_val' and row.get('dataset') == 'interhuman' and row.get('split') == 'validation'))
            if not selected:
                continue
            expected = 'train' if kind in ('guard','original') else 'validation'
            if row.get('split') != expected:
                raise ValueError(f'Split mismatch: {name}, expected {expected}')
            file = prepared/row['path']
            if digest(file) != row['sha256']:
                raise ValueError(f'Data hash mismatch: {file}')
            with np.load(file,allow_pickle=False) as data:
                obs = data['obs'].astype(np.float32)
                target = data['target_pose'].astype(np.float32)
                mask = np.ones(len(obs),dtype=bool)
                if kind.startswith('guard'):
                    ix = data['sample_index']
                    t = ix[:,0]/fps
                    mask = ix[:,1] == 0
                    if kind == 'guard':
                        mask &= ((t >= 8.82-1e-8)&(t < 9.72-1e-8)) | ((t >= 9.74-1e-8)&(t < 10.72-1e-8))
                    else:
                        mask &= (t >= 11.74-1e-8)&(t < 12.46-1e-8)
                obs,target = obs[mask],target[mask]
                if obs.shape != (len(obs),1097) or target.shape != (len(obs),225) or not np.isfinite(obs).all() or not np.isfinite(target).all():
                    raise ValueError('Invalid arrays')
                observations.append(obs); targets.append(target)
                provenance.append(dict(use=kind,source=name,split=expected,
                    packet_sha256=row['sha256'],samples=len(obs)))
        if not observations or sum(map(len,observations)) == 0:
            raise ValueError(f'No samples for {kind}')
        return torch.from_numpy(np.concatenate(observations)),torch.from_numpy(np.concatenate(targets))

    datasets = {name:collect(name) for name in ['guard','original','guard_val','original_val']}
    checkpoint = repo/'results/boxing-sl/epoch_000020.pt'
    saved = torch.load(checkpoint,map_location='cpu',weights_only=False)
    model = load_policy(checkpoint).to(a.device).eval()
    model.requires_grad_(False)
    model.actor.latent_policy.requires_grad_(True)
    before_decoder = state_sha256(model.actor.decoder)
    before_normalizer = state_sha256(model.normalizer)
    before_actor = state_sha256(model.actor.latent_policy)
    stats = saved['loss_statistics']

    @torch.no_grad()
    def evaluate(name):
        obs,target=datasets[name]
        loss_sum=0.; counts={}; pos_sum=0.
        for start in range(0,len(obs),64):
            o=obs[start:start+64].to(a.device); y=target[start:start+64].to(a.device)
            prediction=model(o)
            loss_sum+=float(normalized_loss(prediction,y,stats)['loss'])*len(o)
            pred=prediction.reshape(-1,5,5,9)
            pos_sum+=float((pred[...,:3]-y.reshape(-1,5,5,9)[...,:3]).norm(dim=-1).mean())*len(o)
            poses=o[:,975:1020].reshape(-1,5,9)
            x,z=poses[:,0,3:6],poses[:,0,6:9]
            rotation=torch.stack([x,torch.cross(z,x,dim=-1),z],dim=-1)
            h=torch.einsum('nij,nkj->nki',rotation,pred[:,0,:,:3])
            feet=h[:,4,0]-h[:,3,0]; hands=h[:,2,0]-h[:,1,0]
            for f,v in zip(feet.tolist(),hands.tolist()):
                label='ambiguous' if min(abs(f),abs(v))<=.02 else ('opposite' if f*v<0 else ('right_lead' if f>0 else 'left_lead'))
                counts[label]=counts.get(label,0)+1
        return dict(samples=len(obs),loss=loss_sum/len(obs),mean_position_error_m=pos_sum/len(obs),heading_counts_20ms=counts)

    a.output.mkdir(parents=True)
    config=dict(format='hbot_guard_mixed_experiment_v1',base_checkpoint_sha256=digest(checkpoint),
        manifest_sha256=digest(manifest_path),data=provenance,seed=a.seed,lr=a.lr,
        batch_size=a.batch_size,steps=1 if a.smoke_test else a.steps,device=a.device,
        guard_batch_fraction=.5,normalization='frozen base SL statistics',
        scope='offline fine-tuning; validation is not closed-loop acceptance',
        validation_caveat='004 candidate not yet manually accepted; metrics are provisional',
        smoke_test=a.smoke_test)
    (a.output/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    baseline={name:evaluate(name) for name in datasets}
    optimizer=torch.optim.Adam(model.actor.latent_policy.parameters(),lr=a.lr)
    with (a.output/'train.jsonl').open('w') as log:
        for step in range(config['steps']):
            batches=[]
            for name in ('guard','original'):
                o,y=datasets[name]; ids=rng.integers(0,len(o),size=a.batch_size//2)
                batches.append((o[ids],y[ids]))
            obs=torch.cat([b[0] for b in batches]).to(a.device)
            target=torch.cat([b[1] for b in batches]).to(a.device)
            optimizer.zero_grad(set_to_none=True)
            loss=normalized_loss(model(obs),target,stats)['loss']
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.actor.latent_policy.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            row=dict(step=step+1,loss=float(loss.detach()),gradient_norm=float(norm))
            log.write(json.dumps(row)+'\n');log.flush()
            if step==0 or (step+1)%10==0: print(json.dumps(row),flush=True)
    if state_sha256(model.actor.decoder)!=before_decoder or state_sha256(model.normalizer)!=before_normalizer:
        raise RuntimeError('Frozen decoder/normalizer changed')
    if state_sha256(model.actor.latent_policy)==before_actor:
        raise RuntimeError('Trainable policy did not change')
    after={name:evaluate(name) for name in datasets}
    report=dict(baseline=baseline,after=after,frozen_components_unchanged=True,
        latent_policy_changed=True,closed_loop_verified=False,exported=False)
    if not a.smoke_test:
        model=model.cpu().eval()
        torch.save(dict(format='hbot_guard_mixed_v1',base_checkpoint=str(checkpoint),
                        config=config,policy_state=model.state_dict(),optimizer=optimizer.state_dict()),a.output/'guard_mixed.pt')
        sample=datasets['guard'][0][:1]
        onnx_file=a.output/'guard_candidate.onnx'
        torch.onnx.export(model,(sample,),onnx_file,input_names=['obs'],output_names=['target_pose'],
                          opset_version=17,dynamo=False,external_data=False)
        graph=onnx.load(onnx_file)
        onnx.helper.set_model_props(graph,{'policy_contract':CONTRACT})
        onnx.save(graph,onnx_file)
        graph,size=check_model(onnx_file);check_contract(graph)
        options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
        session=ort.InferenceSession(str(onnx_file),sess_options=options,providers=['CPUExecutionProvider'])
        sample=torch.cat([v[0][:8] for v in datasets.values()])
        with torch.no_grad(): expected=model(sample).numpy()
        actual=np.concatenate([session.run(['target_pose'],{'obs':r[None].numpy()})[0] for r in sample])
        report['export_parity']=check_export_parity(actual,expected)
        report.update(exported=True,onnx_bytes=size,onnx_sha256=digest(onnx_file))
    (a.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)
    print('Output:',a.output.resolve())


if __name__=='__main__':
    main()
