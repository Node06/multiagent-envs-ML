"""Fit bootstrapped structured neural dynamics using image transition labels."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from learned_mpc.model import NeuralDynamics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment', type=Path, default=HERE/'outputs/main_20261014')
    p.add_argument('--members', type=int, default=3)
    p.add_argument('--epochs', type=int, default=160)
    p.add_argument('--checkpoint', default='world.pt')
    p.add_argument('--empirical-init', action='store_true')
    args = p.parse_args(); torch.set_num_threads(1); start = time.perf_counter()
    if (args.experiment/args.checkpoint).exists(): raise RuntimeError('Trained world already frozen')
    data = np.load(args.experiment/'transitions.npz')
    tx, ty = torch.from_numpy(data['x']), torch.from_numpy(data['y'])
    train = np.flatnonzero(data['split'] == 'train'); validation = np.flatnonzero(data['split'] == 'validation')
    test = np.flatnonzero(data['split'] == 'test')
    assert set(data['episode'][train]).isdisjoint(data['episode'][validation])
    assert set(data['episode'][train]).isdisjoint(data['episode'][test])
    assert set(data['episode'][validation]).isdisjoint(data['episode'][test])
    models, histories = [], []
    initialization = []
    if args.empirical_init:
        geometry = data['x'][:, 5:].reshape(-1, 4, 4)
        clear = (geometry[:, :, 0]-geometry[:, :, 3]).min(1) > .09
        for kind in (0, 1):
            index = train[(data['x'][train, 4] == kind) & clear[train]]
            cap = float(np.quantile(np.linalg.norm(data['y'][index], axis=-1), .99))
            slow = index[np.linalg.norm(data['y'][index], axis=-1) < .9*cap]
            a = np.stack((data['x'][slow, :2].ravel(), data['x'][slow, 2:4].ravel()), axis=-1)
            retain, gain = np.linalg.lstsq(a, data['y'][slow].ravel(), rcond=None)[0]
            initialization.append({'cap': cap, 'retention': float(retain), 'gain': float(gain), 'samples': len(slow)})
        print('EMPIRICAL TRAIN-SPLIT INITIALIZATION '+json.dumps(initialization), flush=True)
    for member in range(args.members):
        seed = 5101+member*101; torch.manual_seed(seed); rng = np.random.default_rng(seed)
        model = NeuralDynamics(); opt = torch.optim.Adam(model.parameters(), lr=.001)
        if initialization:
            with torch.no_grad():
                model.cap.copy_(torch.tensor([r['cap'] for r in initialization]))
                model.retention.copy_(torch.tensor([r['retention'] for r in initialization]))
                model.gain.copy_(torch.tensor([r['gain'] for r in initialization]))
        bootstrap = rng.choice(train, len(train), replace=True)
        best = float('inf'); history = []
        for epoch in range(args.epochs):
            for index in np.array_split(rng.permutation(bootstrap), int(np.ceil(len(bootstrap)/512))):
                prediction = model(tx[index])
                # Robust loss reduces the influence of partial-occlusion centroid noise.
                loss = F.smooth_l1_loss(prediction/.02, ty[index]/.02, beta=1.)
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5); opt.step()
            with torch.no_grad(): score = (model(tx[validation])-ty[validation]).square().mean().item()
            if score < best: best = score; best_state = copy.deepcopy(model.state_dict())
            if epoch % 20 == 0 or epoch == args.epochs-1:
                row = {'member': member, 'epoch': epoch+1, 'validation_velocity_mse': score,
                       'gains': model.gain.detach().tolist(), 'caps': model.cap.detach().tolist()}
                history.append(row); print(json.dumps(row), flush=True)
        model.load_state_dict(best_state); model.eval(); models.append(model); histories.append(history)
    with torch.no_grad():
        mean = torch.stack([m(tx) for m in models]).mean(0).numpy()
    errors = np.linalg.norm(mean-data['y'], axis=-1)
    geometry = data['x'][:, 5:].reshape(-1, 4, 4)
    near = (geometry[:, :, 0]-geometry[:, :, 3]).min(1) < .06
    calibration = np.zeros((2, 2))
    for kind in (0, 1):
        for contact in (0, 1):
            index = validation[(data['x'][validation, 4] == kind) & (near[validation] == contact)]
            calibration[kind, contact] = float(np.quantile(errors[index], .9)) if len(index) else float(np.quantile(errors[validation], .9))
    summary = {}
    for split, index in [('train', train), ('validation', validation), ('test', test)]:
        summary[split] = {'body_transitions': len(index), 'mean_next_position_error_pixels': float(errors[index].mean()*.1*400),
                          'p95_next_position_error_pixels': float(np.quantile(errors[index], .95)*.1*400),
                          'mean_velocity_error': float(errors[index].mean())}
    torch.save({'models': [m.state_dict() for m in models], 'calibration': calibration.tolist(), 'dt': .1,
                'architecture': 'learned inertia, actuation and speed limits + 3-layer radial neural contact model',
                'label_source': 'consecutive RGB centers, no environment state', 'split': 'episode-disjoint',
                'initialization_from_training_images': initialization}, args.experiment/args.checkpoint)
    summary.update(calibration_velocity_p90=calibration.tolist(), members=args.members, epochs=args.epochs, seconds=time.perf_counter()-start)
    prefix = '' if args.checkpoint == 'world.pt' else Path(args.checkpoint).stem+'_'
    (args.experiment/(prefix+'fit_history.json')).write_text(json.dumps(histories, indent=2))
    (args.experiment/(prefix+'prediction_metrics.json')).write_text(json.dumps(summary, indent=2))
    print('NEURAL WORLD TRAINING COMPLETE '+json.dumps(summary), flush=True)


if __name__ == '__main__': main()
