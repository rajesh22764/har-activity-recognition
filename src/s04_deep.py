"""Step 4 - 1D-CNN on raw inertial signals (128 timesteps x 9 channels).

Early stopping uses 4 held-out TRAIN subjects (never the test subjects).
Three seeds are trained and averaged (probabilities) to report a stable number.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

from common import MODELS, REPORTS, SEED, load_features, load_raw_signals
from sklearn.preprocessing import LabelEncoder

torch.set_num_threads(8)
Xtr, ytr, str_ = load_features("train")
_, yte, _ = load_features("test")
le = LabelEncoder().fit(ytr)
ytr_i, yte_i = le.transform(ytr), le.transform(yte)
Rtr, Rte = load_raw_signals("train"), load_raw_signals("test")
mu, sd = Rtr.mean((0, 1), keepdims=True), Rtr.std((0, 1), keepdims=True)
Rtr, Rte = (Rtr - mu) / sd, (Rte - mu) / sd
np.savez(MODELS / "cnn_norm.npz", mu=mu, sd=sd)

subs = np.sort(str_.unique())
rng = np.random.RandomState(SEED)
val_subj = rng.choice(subs, 4, replace=False)
vm = str_.isin(val_subj).values
print("validation subjects (from train):", sorted(int(v) for v in val_subj))


class CNN(nn.Module):
    def __init__(self, c=9, k=6):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(nn.Conv1d(i, o, 5, padding=2), nn.BatchNorm1d(o), nn.ReLU(),
                                 nn.MaxPool1d(2))
        self.f = nn.Sequential(blk(c, 32), blk(32, 64), blk(64, 128), nn.AdaptiveAvgPool1d(1),
                               nn.Flatten(), nn.Dropout(.4))
        self.h = nn.Linear(128, k)

    def forward(self, x):  # x: (B, T, C)
        return self.h(self.f(x.transpose(1, 2)))


def tt(a):
    return torch.tensor(a)


def run(seed, epochs=40):
    torch.manual_seed(seed)
    m = CNN()
    opt = torch.optim.AdamW(m.parameters(), 2e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 5e-3, total_steps=int(epochs * -(-int((~vm).sum()) // 64)))
    Xa, ya = tt(Rtr[~vm]), tt(ytr_i[~vm]).long()
    Xv, yv = tt(Rtr[vm]), tt(ytr_i[vm]).long()
    best, state = 0, None
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(len(Xa))
        for i in range(0, len(Xa), 64):
            idx = perm[i:i + 64]
            xb = Xa[idx] + .03 * torch.randn_like(Xa[idx])  # light jitter augmentation
            loss = nn.functional.cross_entropy(m(xb), ya[idx], label_smoothing=.05)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        m.eval()
        with torch.no_grad():
            acc = (m(Xv).argmax(1) == yv).float().mean().item()
        if acc >= best:
            best, state = acc, {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(state)
    m.eval()
    with torch.no_grad():
        pr = torch.softmax(m(tt(Rte)), 1).numpy()
    print(f"seed {seed}: val acc {best:.4f} | test acc {accuracy_score(yte_i, pr.argmax(1)):.4f}", flush=True)
    return pr, m


probs, nets = [], []
for s in (0, 1, 2):
    p, m = run(s)
    probs.append(p); nets.append(m)
ens = np.mean(probs, 0)
single = [accuracy_score(yte_i, p.argmax(1)) for p in probs]
res = {"cnn_single_mean": float(np.mean(single)), "cnn_single_std": float(np.std(single)),
       "cnn_ensemble_acc": float(accuracy_score(yte_i, ens.argmax(1))),
       "cnn_ensemble_macro_f1": float(f1_score(yte_i, ens.argmax(1), average="macro"))}
print(res)
np.save(REPORTS / "pred_cnn.npy", ens.argmax(1)); np.save(REPORTS / "proba_cnn.npy", ens)
torch.save([n.state_dict() for n in nets], MODELS / "cnn_ensemble.pt")
import json; (REPORTS / "04_deep.json").write_text(json.dumps(res, indent=2))
