"""
Conditional-class VAE de HOC phan phoi THAT cua nhip F (Fusion) tu du lieu
that (chi ~384 mau train), roi SINH mau moi tu latent space da hoc -- khac
han phep tron tuyen tinh N+V don gian da lam truoc do (alpha*N + (1-alpha)*V).

Ly do chon VAE thay vi GAN: y van (Collapse by Conditioning, ICLR 2022) xac
nhan GAN co dieu kien de sup mo hinh (mode collapse) khi du lieu it (vai
tram mau) -- VAE on dinh hon nhieu trong che do nay. Tim thay 1 pipeline da
cong bo dung cVAE + latent diffusion cai thien dung nhan dien "fused
contractions" 5-7% -- VAE la buoc dau tien, hop ly, truoc khi can nhac them
diffusion refinement.

Kien truc CO CHU Y nho (latent_dim nho, encoder/decoder it tham so) vi chi
co ~384 mau that -- VAE lon se hoc thuoc (memorize) thay vi hoc duoc cau
truc chung cua "F-ness" co the khai quat hoa duoc.

CHI train tren F trong TRAIN split (khong dung val/test) -- tranh ro ri,
dung quy uoc da thiet lap xuyen suot session.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SIGNAL_LEN = 187


class FusionVAE(nn.Module):
    def __init__(self, signal_len: int = SIGNAL_LEN, latent_dim: int = 12, hidden_dim: int = 48):
        super().__init__()
        self.signal_len = signal_len
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(signal_len, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, signal_len),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    @torch.no_grad()
    def sample(self, n, device):
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decode(z)


def vae_loss(x_recon, x, mu, logvar, beta=0.1):
    """beta < 1 (beta-VAE nguoc) -- uu tien tai tao dung hinh dang hon la ep
    latent gan chuan (chi ~384 mau, ep KL qua manh se lam mau sinh ra qua
    "trung binh hoa", mat chi tiet hinh dang dac trung cua F)."""
    recon_loss = F.mse_loss(x_recon, x, reduction="mean")
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss, recon_loss.detach(), kl_loss.detach()


def load_real_f_train(data_dir="data/processed/splits_rr"):
    d = np.load(f"{data_dir}/inter_train.npz")
    X, y = d["X"].astype(np.float32), d["y"].astype(np.int64)
    F_IDX = 3
    X_f = X[y == F_IDX].squeeze(-1)  # (n_F, 187)
    return X_f


def train_vae(X_f, epochs=300, batch_size=32, lr=1e-3, latent_dim=12, hidden_dim=48,
              beta=0.1, device="cpu", seed=42):
    torch.manual_seed(seed)
    model = FusionVAE(signal_len=X_f.shape[1], latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    X_t = torch.from_numpy(X_f).to(device)
    n = len(X_t)

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        ep_loss, ep_recon, ep_kl = 0.0, 0.0, 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = X_t[idx]
            optimizer.zero_grad()
            x_recon, mu, logvar = model(xb)
            loss, recon, kl = vae_loss(x_recon, xb, mu, logvar, beta=beta)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item() * len(idx)
            ep_recon += recon.item() * len(idx)
            ep_kl += kl.item() * len(idx)
        if epoch % 50 == 0 or epoch == 1:
            print(f"  epoch {epoch:4d} | loss={ep_loss/n:.5f} recon={ep_recon/n:.5f} kl={ep_kl/n:.5f}")

    return model


# =============================================================================
#  SMOKE TEST -- CPU, du lieu that (nhe, khong can GPU)
# =============================================================================
def _smoke_test():
    X_f = load_real_f_train()
    print(f"So mau F that trong train: {X_f.shape}")

    model = train_vae(X_f, epochs=30, batch_size=32, device="cpu")  # rut gon epoch cho smoke test
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    model.eval()
    samples = model.sample(5, device="cpu").numpy()
    print(f"Sample shape: {samples.shape}")
    print(f"Sample range: [{samples.min():.3f}, {samples.max():.3f}]  (that: [{X_f.min():.3f}, {X_f.max():.3f}])")
    print("Smoke test PASSED (khong loi runtime)")


if __name__ == "__main__":
    _smoke_test()
