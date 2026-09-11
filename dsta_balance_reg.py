"""
Regularizer "can bang" (balance) cho DSTALiteGate -- pha vo doi xung TY LE
(scaling symmetry) cua phan ra CP/trilinear (factor_a, factor_b, factor_c)
da xac nhan bang toan hoc (xem phan tich truoc: nhan cot rank r cua factor_a
len alpha, chia cot tuong ung cua factor_b cho alpha -- dau ra KHONG DOI).

Doi xung ty le lam mat 1 huong PHANG (Hessian = 0) trong loss landscape --
nhieu so hoc cuc nho (khac biet duy nhat giua 2 lan chay CUNG seed tren GPU)
co the day tham so troi dat doc theo huong nay qua hang nghin buoc, dan den
2 lan chay "giong het nhau" phan ky thanh ket qua rat khac nhau.

Ket qua kinh dien trong ly thuyet phan ra ma tran/tensor: neu ep CAC THANH
PHAN CAN BANG NHAU VE CHUAN (||factor_a[:,r]|| = ||factor_b[:,r]|| =
||factor_c[:,r]|| cho moi r), diem can bang duy nhat nay khong con doi xung
ty le lien tuc nua (chi con doi xung dau +-1, khong nguy hiem) -- bien
"thung lung phang" thanh "thung lung co day", giu nguyen kha nang bieu dien
cua model (vi day chi la 1 CACH THAM SO HOA khac cua CUNG 1 ham, khong doi
gia tri toi uu cua ham loss chinh, chi lam no de hoi tu on dinh hon).

L_balance = sum qua moi DSTALiteGate trong model, moi rank r:
    (||factor_a[:,r]||^2 - ||factor_b[:,r]||^2)^2 +
    (||factor_b[:,r]||^2 - ||factor_c[:,r]||^2)^2
"""

import torch
import torch.nn as nn


def find_dsta_gates(model: nn.Module):
    """Tim tat ca submodule co ca 3 tham so factor_a/factor_b/factor_c (tuc
    la DSTALiteGate), khong can import truc tiep class de tranh phu thuoc
    vong -- chi can duck-typing theo ten tham so."""
    gates = []
    for m in model.modules():
        if hasattr(m, "factor_a") and hasattr(m, "factor_b") and hasattr(m, "factor_c"):
            gates.append(m)
    return gates


def compute_balance_loss(model: nn.Module) -> torch.Tensor:
    """L_balance = tong qua moi DSTALiteGate, moi head, moi rank component:
    chenh lech binh phuong giua chuan^2 cua 3 factor. Tra ve tensor 0. neu
    khong tim thay gate nao (an toan, khong crash)."""
    gates = find_dsta_gates(model)
    if not gates:
        p = next(model.parameters())
        return p.new_zeros(())

    total = None
    for gate in gates:
        # factor_a/b/c: (num_heads, rank, head_dim) -- chuan^2 theo chieu head_dim (dim=-1)
        norm_a2 = (gate.factor_a ** 2).sum(dim=-1)  # (H, R)
        norm_b2 = (gate.factor_b ** 2).sum(dim=-1)
        norm_c2 = (gate.factor_c ** 2).sum(dim=-1)
        term = (norm_a2 - norm_b2) ** 2 + (norm_b2 - norm_c2) ** 2 + (norm_c2 - norm_a2) ** 2
        gate_loss = term.mean()
        total = gate_loss if total is None else total + gate_loss
    return total


# =============================================================================
#  SMOKE TEST -- xac nhan regularizer giam dan khi factor duoc can bang thu cong,
#  va = 0 khi factor da can bang hoan toan.
# =============================================================================
def _smoke_test():
    torch.manual_seed(0)
    H, R, Dh = 2, 4, 12

    class DummyGate(nn.Module):
        def __init__(self, scale_a=1.0, scale_b=1.0, scale_c=1.0):
            super().__init__()
            base = torch.randn(H, R, Dh)
            self.factor_a = nn.Parameter(base * scale_a)
            self.factor_b = nn.Parameter(base * scale_b)
            self.factor_c = nn.Parameter(base * scale_c)

    class DummyModel(nn.Module):
        def __init__(self, **kw):
            super().__init__()
            self.gate = DummyGate(**kw)

    m_balanced = DummyModel(scale_a=1.0, scale_b=1.0, scale_c=1.0)
    m_imbalanced = DummyModel(scale_a=5.0, scale_b=0.1, scale_c=1.0)

    l_bal = compute_balance_loss(m_balanced)
    l_imbal = compute_balance_loss(m_imbalanced)
    print(f"L_balance (da can bang, scale=1,1,1): {l_bal.item():.8f}  (ky vong = 0)")
    print(f"L_balance (mat can bang, scale=5,0.1,1): {l_imbal.item():.4f}  (ky vong > 0, lon)")

    ok = abs(l_bal.item()) < 1e-6 and l_imbal.item() > 1.0
    print("Smoke test PASSED" if ok else "Smoke test FAILED")


if __name__ == "__main__":
    _smoke_test()
