"""
"Tron da dang roi tinh chinh" (diverse-blend-then-refine): giai quyet dung
van de vua phat hien bang thuc nghiem -- VAE hoc truc tiep tu 384 mau F
that (90% tu 1 benh nhan 208) chi hoc duoc "chu ky cua benh nhan 208", KHONG
the tu tao ra da dang cua benh nhan khac (AUC vs test that = 0.81, lech xa;
AUC vs chinh nguon no hoc = 0.57, gan).

Buoc 1 -- nguon DA DANG THAT: tron N that + V that tu NHIEU benh nhan khac
nhau (N gan nhu moi benh nhan, V tu 33 benh nhan -- da dang thuc su, khac
han 384 mau F chi tu ~5 benh nhan) bang alpha ngau nhien (da kiem chung
truoc: AUC(F gia lap tho vs F that)=0.527 ve QRS-width -- hop ly thong ke
o muc do tho).

Buoc 2 -- REFINER hoc lam mau tron THO "giong that" hon: khong dung GAN doi
khang (rui ro mode-collapse voi du lieu it da xac nhan qua tim hieu y van)
-- dung MMD (Maximum Mean Discrepancy, kernel RBF) tren KHONG GIAN DAC TRUNG
da kiem chung (template-corr, QRS-width, bien do dinh, nang luong tin hieu)
de keo phan phoi cua {refined} ve gan phan phoi cua F THAT, cong them 1 loss
"giu gan dang goc" (identity, trong so nho) de KHONG mat da dang da co tu
nguon tron -- muc tieu: refined vua GIONG THAT (thong ke) vua DA DANG (ke
thua tu nhieu benh nhan cua N/V).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dsta_c_synth_fusion import synthesize_fusion_batch

SIGNAL_LEN = 187
N_IDX, V_IDX, F_IDX = 0, 2, 3


class ResidualRefiner(nn.Module):
    """refined = crude + scale * delta -- delta hoc 1 hieu chinh NHO, giu
    refined gan voi crude (bao toan da dang tu nguon tron)."""

    def __init__(self, signal_len: int = SIGNAL_LEN, hidden_dim: int = 64, delta_scale: float = 0.3):
        super().__init__()
        self.delta_scale = delta_scale
        self.net = nn.Sequential(
            nn.Linear(signal_len, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, signal_len),
        )

    def forward(self, x_crude):
        delta = torch.tanh(self.net(x_crude))  # tanh: gioi han bien do hieu chinh
        return x_crude + self.delta_scale * delta


def extract_features(x, template_norm):
    """x: (B, 187). Tra ve (B, 4): template-corr, QRS-width-proxy, bien do
    dinh, nang luong tin hieu -- dung KHONG GIAN DAC TRUNG da kiem chung
    truoc do, khong bia dac trung moi."""
    xn = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
    corr = (xn * template_norm).sum(dim=-1)  # (B,)

    peak_amp = x.abs().max(dim=-1).values
    thresh = 0.6 * peak_amp
    above = (x.abs() > thresh.unsqueeze(-1)).float()
    qrs_width = above.sum(dim=-1)  # xap xi tho (khong lan trai/phai tu dinh nhu ban numpy, du dung
                                     # de lam dac trung phan biet thong ke, khong can chinh xac tuyet doi)

    energy = (x ** 2).sum(dim=-1)

    feats = torch.stack([corr, qrs_width, peak_amp, energy], dim=-1)  # (B,4)
    return feats


def rbf_mmd(x, y, sigma=None):
    """MMD^2 voi kernel RBF giua 2 tap mau x,y (B1,D),(B2,D). sigma=None ->
    tu dong theo median heuristic."""
    xx = torch.cdist(x, x, p=2) ** 2
    yy = torch.cdist(y, y, p=2) ** 2
    xy = torch.cdist(x, y, p=2) ** 2
    if sigma is None:
        combined = torch.cat([xx.flatten(), yy.flatten(), xy.flatten()])
        sigma = combined.median().clamp(min=1e-6)
    kxx = torch.exp(-xx / (2 * sigma))
    kyy = torch.exp(-yy / (2 * sigma))
    kxy = torch.exp(-xy / (2 * sigma))
    return kxx.mean() + kyy.mean() - 2 * kxy.mean()


def normalize_features(feats, mean, std):
    return (feats - mean) / (std + 1e-8)


def train_refiner(X_crude, X_real_f, template_norm, epochs=300, batch_size=64,
                   lr=1e-3, lambda_identity=0.3, device="cpu", seed=42):
    torch.manual_seed(seed)
    model = ResidualRefiner(signal_len=X_crude.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    Xc_t = torch.from_numpy(X_crude).to(device)
    Xr_t = torch.from_numpy(X_real_f).to(device)
    template_t = torch.from_numpy(template_norm).to(device)

    # Chuan hoa dac trung -- fit tren F THAT (nguon can khop toi)
    with torch.no_grad():
        feat_real_raw = extract_features(Xr_t, template_t)
        feat_mean = feat_real_raw.mean(dim=0)
        feat_std = feat_real_raw.std(dim=0)
        feat_real_norm = normalize_features(feat_real_raw, feat_mean, feat_std)

    n = len(Xc_t)
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        ep_loss, ep_mmd, ep_id = 0.0, 0.0, 0.0
        n_batches = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = Xc_t[idx]
            optimizer.zero_grad()
            refined = model(xb)

            feat_refined_raw = extract_features(refined, template_t)
            feat_refined_norm = normalize_features(feat_refined_raw, feat_mean, feat_std)

            l_mmd = rbf_mmd(feat_refined_norm, feat_real_norm)
            l_identity = F.mse_loss(refined, xb)
            loss = l_mmd + lambda_identity * l_identity
            loss.backward()
            optimizer.step()

            ep_loss += loss.item()
            ep_mmd += l_mmd.item()
            ep_id += l_identity.item()
            n_batches += 1
        if epoch % 50 == 0 or epoch == 1:
            print(f"  epoch {epoch:4d} | loss={ep_loss/n_batches:.5f} mmd={ep_mmd/n_batches:.5f} "
                  f"identity={ep_id/n_batches:.5f}")

    return model, feat_mean, feat_std


def load_train_data(data_dir="data/processed/splits_rr"):
    d = np.load(f"{data_dir}/inter_train.npz")
    X, y = d["X"].astype(np.float32).squeeze(-1), d["y"].astype(np.int64)
    return X, y


# =============================================================================
#  SMOKE TEST -- CPU, du lieu that
# =============================================================================
def _smoke_test():
    rng = np.random.default_rng(42)
    X_train, y_train = load_train_data()
    X_train_3d = X_train[:, :, None]  # synthesize_fusion_batch mong doi (N,T,1)

    X_crude, y_crude, alpha_crude = synthesize_fusion_batch(
        X_train_3d, y_train, n_synth=1000, rng=rng, alpha_lo=0.3, alpha_hi=0.7)
    X_crude = X_crude.squeeze(-1)
    print(f"Da tron {len(X_crude)} mau F tho tu N/V da dang (nhieu benh nhan)")

    X_real_f = X_train[y_train == F_IDX]
    print(f"F that trong train: {X_real_f.shape}")

    template = X_train[y_train == N_IDX].mean(axis=0)
    template_norm = (template / (np.linalg.norm(template) + 1e-8)).astype(np.float32)

    model, feat_mean, feat_std = train_refiner(
        X_crude, X_real_f, template_norm, epochs=30, batch_size=64, device="cpu")  # rut gon smoke test
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    model.eval()
    with torch.no_grad():
        sample_in = torch.from_numpy(X_crude[:5])
        refined = model(sample_in)
    print(f"Refined shape: {tuple(refined.shape)}")
    print("Smoke test PASSED (khong loi runtime)")


if __name__ == "__main__":
    _smoke_test()
