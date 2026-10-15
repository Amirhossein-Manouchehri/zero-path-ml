"""Reproduce the published frozen-CNN L=36 metrics without the research solver."""
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
import torch

from src.multisize_cnn import MultiSizeCNN
from src.predict_multisize import predict

ARTIFACTS = Path(__file__).resolve().parents[1] / 'artifacts' / 'multisize_v1'


def evaluate():
    """Verify snapshot integrity, run inference, and return directly measured metrics."""
    manifest = json.loads((ARTIFACTS / 'manifest.json').read_text())
    for name, expected in manifest['files'].items():
        actual = hashlib.sha256((ARTIFACTS / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Artifact checksum mismatch: {name}')
    checkpoint = torch.load(ARTIFACTS / 'model.pt', map_location='cpu', weights_only=True)
    model = MultiSizeCNN().eval()
    model.load_state_dict(checkpoint['state_dict'])
    with np.load(ARTIFACTS / 'L36_test.npz', allow_pickle=False) as data:
        L = json.loads(data['metadata'].item())['L']
        y, p = data['y'], data['p_sample']
        q = predict(data['X'], L, model, checkpoint['calibration'])
    curve_mae = np.mean([abs(float(q[p == value].mean()) - float(y[p == value].mean())) for value in np.unique(p)])
    return {
        'L': L,
        'examples': len(y),
        'majority_class_accuracy': float(max(y.mean(), 1 - y.mean())),
        'accuracy': float(accuracy_score(y, q >= .5)),
        'balanced_accuracy': float(balanced_accuracy_score(y, q >= .5)),
        'roc_auc': float(roc_auc_score(y, q)),
        'curve_mae': float(curve_mae),
    }


def main():
    result = evaluate()
    print(f'Frozen model on {result["examples"]} unseen-size L={result["L"]} realizations')
    for key in ['majority_class_accuracy', 'accuracy', 'balanced_accuracy']:
        print(f'{key}: {100 * result[key]:.2f}%')
    print(f'roc_auc: {result["roc_auc"]:.4f}')
    print(f'curve_mae: {100 * result["curve_mae"]:.2f} percentage points')
    print('Solver-reference agreement; this does not establish precision finite-size scaling.')


if __name__ == '__main__':
    main()
