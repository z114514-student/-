"""Offline guard geometry diagnosis; run on trusted local checkpoints only."""
import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def classify(points, margin):
    feet = points[..., 4, 0] - points[..., 3, 0]
    hands = points[..., 2, 0] - points[..., 1, 0]
    labels = np.where((np.abs(feet) <= margin) | (np.abs(hands) <= margin),
        'ambiguous', np.where(feet * hands < 0, 'opposite',
        np.where(feet > 0, 'right_lead', 'left_lead')))
    return feet, hands, labels


def to_heading(values, rotation):
    points = values.reshape(-1, 5, 5, 9)[..., :3]
    # All methods use the same observation-time pelvis -> heading rotation.
    return np.einsum('nij,nhkj->nhki', rotation, points)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=Path.cwd())
    p.add_argument('--start', type=float, default=8.82)
    p.add_argument('--end', type=float, default=10.72)
    p.add_argument('--margin', type=float, default=.02)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Use a new output directory')
    if not (0 <= a.start <= a.end) or a.margin <= 0:
        p.error('Invalid time range or margin')
    repo = a.repo.resolve()
    sys.path.insert(0, str(repo))
    import torch
    import onnxruntime as ort
    from holoboxing.training.motion_prior import load_prior

    torch.set_num_threads(1)
    prepared = Path((repo/'logs/prepared_data_path.txt').read_text().strip())
    manifest_path = prepared/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    matches = [r for r in manifest['records'] if Path(r.get('source', '')).name == 'kangning-003_mirror.npz']
    if len(matches) != 1:
        raise ValueError(f'Expected one source record, found {len(matches)}')
    record = matches[0]
    packet = prepared/record['path']
    fps = float(manifest.get('control_hz', 50))
    with np.load(packet, allow_pickle=False) as d:
        index = d['sample_index']
        mask = (index[:,1] == 0) & (index[:,0]/fps >= a.start-1e-8) & (index[:,0]/fps <= a.end+1e-8)
        obs = d['obs'][mask].astype(np.float32)
        target = d['target_pose'][mask].astype(np.float32)
        index = index[mask]
        windows = d['window_index'][mask]
    if not len(obs) or obs.shape[1:] != (1097,) or target.shape != (len(obs),225):
        raise ValueError('Missing samples or invalid observation/target shapes')
    if not np.isfinite(obs).all() or not np.isfinite(target).all():
        raise ValueError('Nonfinite source data')
    poses = obs[:,975:1020].reshape(-1,5,9).astype(float)
    x, z = poses[:,0,3:6], poses[:,0,6:9]
    rotation = np.stack([x,np.cross(z,x),z],axis=-1)
    np.testing.assert_allclose(rotation.transpose(0,2,1)@rotation,
                              np.broadcast_to(np.eye(3),rotation.shape),atol=2e-3)

    prior_path = repo/'results/boxing-prior/best.pt'
    onnx_path = repo/'results/boxing-sl-v1.onnx'
    prior, saved = load_prior(prior_path, device='cpu')
    with torch.no_grad():
        reconstructed = prior(torch.from_numpy(target)).numpy()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(onnx_path), sess_options=options,
                                  providers=['CPUExecutionProvider'])
    if [v.name for v in session.get_inputs()] != ['obs'] or [v.name for v in session.get_outputs()] != ['target_pose']:
        raise ValueError('Unexpected ONNX input/output names')
    predicted = np.concatenate([session.run(['target_pose'], {'obs': row[None]})[0] for row in obs],axis=0)
    from holoboxing.training.export import load_policy, check_export_parity
    sl_path = repo/'results/boxing-sl/epoch_000020.pt'
    policy = load_policy(sl_path)
    with torch.no_grad():
        eager = policy(torch.from_numpy(obs)).numpy()
    parity = check_export_parity(predicted, eager)
    sl_saved = torch.load(sl_path, map_location='cpu', weights_only=False)
    expected_prior = sl_saved.get('policy_spec', {}).get('prior_checkpoint_sha256')
    if expected_prior != sha(prior_path):
        raise ValueError('SL checkpoint prior identity does not match tested prior')
    for name, values in [('prior',reconstructed),('onnx',predicted)]:
        if values.shape != target.shape or not np.isfinite(values).all():
            raise ValueError(f'Invalid {name} output')
    methods = {name:to_heading(values,rotation) for name,values in
               [('reference',target),('prior_reconstruction',reconstructed),('onnx_prediction',predicted)]}
    actual_labels = classify(poses[...,:3],a.margin)[2]
    result = {
        'scope':'offline geometry only; no closed-loop or contact verification',
        'frame':'observation-time own heading, not opponent-facing',
        'reference_time':'future 20..100ms; actual observation is current time',
        'source':record['source'], 'samples':len(obs), 'margin_m':a.margin,
        'start_s':a.start, 'end_s':a.end,
        'windows':np.unique(windows,axis=0).tolist(),
        'early_failures':record.get('early_failures'),
        'hashes':{'manifest':sha(manifest_path),'packet':sha(packet),
                  'prior':sha(prior_path),'onnx':sha(onnx_path)},
        'prior_training_groups':saved.get('training_groups'),
        'prior_validation_groups':saved.get('validation_groups'),
        'note':'Measured-reference reconstruction is a diagnostic input and may differ from prior-fitting reference frames. A failure alone does not establish decoder inability.',
        'actual_current_counts':dict(Counter(actual_labels.tolist())),
        'onnx_vs_pytorch_parity':parity,
        'supervised_checkpoint_sha256':sha(sl_path),
        'prior_identity_matches_supervised':True,
        'methods':{}}
    a.output.mkdir(parents=True)
    with (a.output/'frames.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.writer(f)
        writer.writerow(['source_frame','time_s','window_start','window_core','horizon_ms','method','feet_dx_m','hands_dx_m','label'])
        for name, points in methods.items():
            feet,hands,labels=classify(points,a.margin)
            summary=[]
            for h in range(5):
                item={'horizon_ms':(h+1)*20,'counts':dict(Counter(labels[:,h].tolist()))}
                if name != 'reference':
                    distances=np.linalg.norm(points[:,h]-methods['reference'][:,h],axis=-1)
                    item['mean_position_error_m_by_point']=dict(zip(
                        ['pelvis','left_wrist','right_wrist','left_ankle','right_ankle'],distances.mean(axis=0).tolist()))
                summary.append(item)
                for i in range(len(obs)):
                    writer.writerow([int(index[i,0]),float(index[i,0]/fps),
                        int(windows[i,0]),int(windows[i,1]),(h+1)*20,name,
                        float(feet[i,h]),float(hands[i,h]),str(labels[i,h])])
            result['methods'][name]=summary
    (a.output/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    print('Saved:',a.output.resolve())


if __name__ == '__main__':
    main()
