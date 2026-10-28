"""Read-only guard candidate screening; no claim of contact or tracking validity."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def heading_points(p, q):
    norm = np.linalg.norm(q, axis=-1)
    if not np.allclose(norm, 1, atol=1e-3):
        raise ValueError('Pelvis quaternions must be unit WXYZ')
    w, x, y, z = (q / norm[:, None]).T
    yaw = np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    rel = p-p[:, :1]
    out = rel.copy()
    c, s = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
    out[..., 0] = c*rel[..., 0]+s*rel[..., 1]
    out[..., 1] = -s*rel[..., 0]+c*rel[..., 1]
    return out


def classify(feet, hands, margin):
    f = np.where(feet > margin, 1, np.where(feet < -margin, -1, 0))
    h = np.where(hands > margin, 1, np.where(hands < -margin, -1, 0))
    return np.where((f == 0) | (h == 0), 'ambiguous',
                    np.where(f != h, 'opposite',
                             np.where(f == 1, 'right_lead', 'left_lead')))


def intervals(mask, fps, minimum):
    bounds = np.diff(np.r_[False, mask, False].astype(int))
    for a, b in zip(np.flatnonzero(bounds == 1), np.flatnonzero(bounds == -1)):
        if (b-a)/fps >= minimum:
            yield int(a), int(b)


def inspect(path):
    if path.suffix.lower() in ('.h5', '.hdf5'):
        import h5py
        with h5py.File(path, 'r') as f:
            def show(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(name, obj.shape, obj.dtype)
            f.visititems(show)
    elif path.suffix.lower() == '.npz':
        with np.load(path, allow_pickle=False) as d:
            for k in d.files:
                try:
                    print(k, d[k].shape, d[k].dtype)
                except ValueError:
                    print(k, 'object data: not loaded')
    else:
        raise ValueError('Inspect accepts NPZ or HDF5')


def scan(args):
    schema = json.loads(args.schema.read_text())
    names = schema['body_names']
    ids = [names.index(n) for n in ('pelvis', 'left_wrist_yaw_link',
           'right_wrist_yaw_link', 'left_ankle_roll_link', 'right_ankle_roll_link')]
    files = sorted(args.source.rglob('*.npz')) if args.source.is_dir() else [args.source]
    if not files:
        raise ValueError('No NPZ files found')
    args.output.mkdir(parents=True, exist_ok=False)
    summaries = []
    for file_index, path in enumerate(files):
        with np.load(path, allow_pickle=False) as d:
            p = np.asarray(d['body_pos_w'], dtype=float)
            q = np.asarray(d['body_quat_w'], dtype=float)
            fps = float(np.asarray(d['fps']).reshape(-1)[0])
        if (p.ndim != 3 or p.shape[1:] != (len(names), 3)
                or q.shape != (*p.shape[:2], 4) or len(p) < 2
                or not np.isfinite(fps) or fps <= 0
                or not np.isfinite(p).all() or not np.isfinite(q).all()):
            raise ValueError(f'Invalid source schema: {path}')
        points = p[:, ids]
        local = heading_points(points, q[:, ids[0]])
        feet = local[:, 4, 0]-local[:, 3, 0]
        hands = local[:, 2, 0]-local[:, 1, 0]
        labels = classify(feet, hands, args.margin)
        # Finite differences over one real frame; first frame is not qualified.
        speed = np.linalg.norm(np.diff(points, axis=0), axis=-1)*fps
        quiet = np.r_[False, (speed[:, 0] <= args.pelvis_speed)
                      & (speed[:, 1:3].max(axis=1) <= args.wrist_speed)
                      & (speed[:, 3:5].max(axis=1) <= args.ankle_speed)]
        # Quiet means low translation only; not a confirmed guard or no-contact state.
        csv_path = args.output / f'{file_index:02d}_{path.stem}.csv'
        with csv_path.open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['time_s','right_minus_left_ankle_x_m',
                             'right_minus_left_wrist_x_m','geometry','quiet_candidate',
                             'left_ankle_world_z_m','right_ankle_world_z_m'])
            for i in range(len(p)):
                writer.writerow([i/fps, feet[i], hands[i], labels[i],
                                 int(quiet[i]), points[i,3,2], points[i,4,2]])
        events = []
        for label in ('opposite', 'right_lead', 'left_lead', 'ambiguous'):
            for a, b in intervals(quiet & (labels == label), fps, args.minimum):
                events.append(dict(kind=label, start_s=a/fps,
                                   end_exclusive_s=b/fps, duration_s=(b-a)/fps))
        events.sort(key=lambda e: e['start_s'])
        summary = dict(source=str(path.resolve()), frames=len(p), fps=fps,
                       csv=str(csv_path), candidates=events)
        summaries.append(summary)
        print(path.name)
        for e in events:
            print(f"  {e['kind']}: {e['start_s']:.2f}--{e['end_exclusive_s']:.2f}s")
        if not events:
            print('  No continuous quiet candidates at current thresholds')
    report = dict(frame='own pelvis heading, not opponent-facing',
                  scope='reference geometry only; not policy targets or physical tracking',
                  thresholds=dict(margin_m=args.margin, minimum_s=args.minimum,
                     pelvis_speed=args.pelvis_speed, wrist_speed=args.wrist_speed,
                     ankle_speed=args.ankle_speed), files=summaries)
    (args.output/'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Saved:', args.output.resolve())


def self_test():
    p = np.array([[[0.,0.,1.],[.1,.2,1.4],[.2,-.2,1.4],
                   [-.1,.1,.04],[.1,-.1,.04]]])
    q = np.array([[1.,0.,0.,0.]])
    base = heading_points(p,q)
    rot = np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    moved = p@rot.T + np.array([3.,4.,0.])
    actual = heading_points(moved,np.array([[2**-.5,0.,0.,2**-.5]]))
    np.testing.assert_allclose(base,actual,atol=1e-12)
    assert classify(np.array([.1,-.1,.001]),np.array([-.1,-.1,.1]),.02).tolist() == ['opposite','left_lead','ambiguous']
    assert list(intervals(np.array([False,True,True,False]), 2, 1)) == [(1,3)]
    print('PASS: heading rotation/translation invariance, side classification, interval duration')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('self-test')
    check = sub.add_parser('inspect')
    check.add_argument('file', type=Path)
    p = sub.add_parser('scan')
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--schema', type=Path, default=Path('holoboxing/training/assets/robot_schema.json'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--margin', type=float, default=.02)
    p.add_argument('--minimum', type=float, default=.5)
    p.add_argument('--pelvis-speed', type=float, default=.08)
    p.add_argument('--wrist-speed', type=float, default=.15)
    p.add_argument('--ankle-speed', type=float, default=.05)
    args = parser.parse_args()
    if args.command == 'self-test':
        self_test()
    elif args.command == 'inspect':
        inspect(args.file)
    else:
        if min(args.margin,args.minimum,args.pelvis_speed,args.wrist_speed,args.ankle_speed) <= 0:
            parser.error('Thresholds must be positive')
        scan(args)


if __name__ == '__main__':
    main()
