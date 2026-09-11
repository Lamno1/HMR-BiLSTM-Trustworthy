"""
CAMP-Net: Confusion-Aware Mixture-Prototype Network
================================================================================

Kiến trúc mới, độc lập với nhánh nghiên cứu DSTA (không đổi gì ở gate LSTM),
thiết kế để tấn công trực diện vấn đề đã chẩn đoán qua nhiều lần phân tích
trong dự án này: lớp Fusion (F) không phải là vấn đề THIẾU DỮ LIỆU, mà là vấn
đề HÌNH HỌC RANH GIỚI QUYẾT ĐỊNH -- về sinh lý học, một nhịp Fusion chính là
sự chồng lấn vật lý giữa một nhịp N và một nhịp V, nên embedding của nó tự
nhiên rơi vào vùng chồng lấn giữa 2 lớp đó. Focal Loss + class weight (đã có
sẵn trong pipeline) chỉ sửa được trục TẦN SUẤT, đã chạm trần (F1-F=0.24 dù đã
ưu tiên tối đa cho lớp hiếm) -- CAMP-Net thêm 3 cơ chế mới, cả 3 đều nằm trên
trục HÌNH HỌC, trực giao với trục tần suất nên KHÔNG chồng chéo với Focal
Loss/class-weight hiện có (không vi phạm nguyên tắc "không stack 2 correction
cùng trục" đã rút ra khi audit `ablation_dsta_variants.py`):

  1. Confusion-Margin Loss -- ép logit của lớp đúng (F) vượt logit của đúng
     lớp gây nhiễu (N hoặc V) tối thiểu 1 margin, thay vì chỉ cần > 0.
  2. Mixture-Decomposition Loss -- tận dụng thẳng sự thật sinh lý học "F =
     blend(N, V)" làm inductive bias: học hệ số trộn alpha_t và ép embedding
     của mẫu F thực sự giải thích được bằng tổ hợp lồi của centroid N, V.
  3. Prototype-Repulsion -- có 1 prototype học được (không phải centroid mô
     tả) cho mỗi lớp, dùng làm classifier phụ trợ theo khoảng cách; đẩy
     prototype F ra xa prototype N, V tối thiểu 1 khoảng eta.

Thiết kế 2 loại "điểm neo" khác nhau, KHÔNG dùng chung 1 thứ cho cả 2 việc
(đây là chỗ dễ có bug nếu làm ẩu -- xem lưu ý trong ClassPrototypeBank):
  - `ema_centroids`  : buffer cập nhật bằng EMA (không qua backprop), mô tả
                       ĐÚNG phân bố embedding thật của mỗi lớp -- chỉ dùng cho
                       mixture-decomposition (formula 2), vì ở đó cần so với
                       "N/V thật trông như thế nào", không phải một điểm ảo.
  - `class_prototypes`: nn.Parameter học qua backprop -- dùng cho classifier
                       phụ trợ + prototype-repulsion (formula 1, 3), vì
                       repulsion cần đẩy được (cần có gradient để dịch
                       chuyển), một buffer EMA thụ động không thể "bị đẩy".

Backbone CỐ Ý dùng BiLSTM chuẩn (nn.LSTM), KHÔNG dùng RLSTM/DSTA-Lite gate --
để đóng góp mới của CAMP-Net nằm hoàn toàn ở tầng biểu diễn/phân loại, độc
lập & trực giao với nhánh nghiên cứu gate (ablation_dsta_variants.py). Về
sau có thể ghép "đầu CAMP" lên trên bất kỳ backbone gate nào (RLSTM gốc,
DSTA-Lite, DSTA đầy đủ) thắng trong ablation, không xung đột.

Cảm hứng học thuật (không phải copy nguyên bản, chỉ mượn ý tưởng):
  - Prototypical Networks (Snell et al., 2017) cho classifier theo khoảng
    cách tới prototype lớp.
  - Large-Margin / ArcFace-style losses (Liu et al. 2016; Deng et al. 2019)
    cho ý tưởng ép margin giữa các lớp dễ nhầm thay vì chỉ tối đa hoá CE.
  - Mixture-of-2 decomposition là ý tưởng tự đề xuất, xuất phát trực tiếp từ
    bản chất sinh lý học của nhịp Fusion (không dựa trên 1 paper cụ thể).

CNN frontend, AttentionPooling, FocalLoss: TÁI SỬ DỤNG nguyên bản từ
`hmr_bilstm.py` (đã kiểm chứng đúng, không viết lại để tránh bug mới).
"""

import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

if sys.stdout.encoding is not None and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, f1_score

from hmr_bilstm import AttentionPooling, BiRLSTM, ECGFeatureExtractor, FocalLoss, temporal_smoothness_loss

CLASS_NAMES = ["N", "S", "V", "F", "Q"]
N_IDX, S_IDX, V_IDX, F_IDX, Q_IDX = 0, 1, 2, 3, 4


# =============================================================================
#  1) EMA CLASS CENTROID BANK -- mô tả phân bố embedding thật, KHÔNG học qua
#     backprop. Chỉ dùng làm mục tiêu cho mixture-decomposition loss.
# =============================================================================
class ClassPrototypeBank(nn.Module):
    """
    Centroid trung bình động (EMA) của embedding mỗi lớp, kiểu Prototypical
    Networks nhưng cập nhật online thay vì tính lại mỗi episode. KHÔNG phải
    nn.Parameter -- không có gradient chảy vào đây, cập nhật thủ công bằng
    `update()` dưới torch.no_grad() sau mỗi batch train.

    Vì sao tách riêng khỏi `class_prototypes` học được (xem PrototypeAugmentedClassifier
    bên dưới): nếu dùng chung 1 bộ prototype "học được" cho cả mixture-target
    lẫn repulsion, gradient của repulsion loss sẽ kéo prototype F ra xa
    prototype N/V theo hướng CÓ LỢI CHO CLASSIFIER, không nhất thiết còn phản
    ánh đúng "N và V thật trông như thế nào" nữa -- mixture-decomposition lúc
    đó sẽ so sánh embedding F với một mục tiêu đã bị bóp méo, mất hết ý nghĩa
    sinh lý học ban đầu. Tách bạch 2 vai trò giải quyết đúng vấn đề này.
    """

    def __init__(self, num_classes: int, embed_dim: int, momentum: float = 0.9):
        super().__init__()
        self.num_classes = num_classes
        self.momentum = momentum
        self.register_buffer("prototypes", torch.zeros(num_classes, embed_dim))
        self.register_buffer("initialized", torch.zeros(num_classes, dtype=torch.bool))

    @torch.no_grad()
    def update(self, embeddings: torch.Tensor, labels: torch.Tensor) -> None:
        embeddings = embeddings.detach()
        for c in range(self.num_classes):
            mask = labels == c
            if mask.sum() == 0:
                continue
            batch_mean = embeddings[mask].mean(dim=0)
            if not self.initialized[c]:
                self.prototypes[c] = batch_mean
                self.initialized[c] = True
            else:
                self.prototypes[c].mul_(self.momentum).add_(batch_mean, alpha=1.0 - self.momentum)

    def get(self, class_idx: int) -> torch.Tensor:
        return self.prototypes[class_idx]

    def is_ready(self, *class_indices: int) -> bool:
        return bool(all(self.initialized[i].item() for i in class_indices))


# =============================================================================
#  2) MIXTURE-DECOMPOSITION HEAD -- Công thức 2: F = alpha*mu_N + (1-alpha)*mu_V + residual
# =============================================================================
class MixtureDecompositionHead(nn.Module):
    """
    Với 1 lớp được khai báo là "blend" của 2 lớp thành phần (mặc định: F là
    blend của N, V), học hệ số trộn alpha_t theo từng mẫu và đo residual
    ||h - (alpha*mu_a + (1-alpha)*mu_b)||^2. Residual nhỏ = embedding thực sự
    giải thích được bằng công thức trộn vật lý của Fusion; residual lớn = mẫu
    này không phải "Fusion điển hình" (model không bị ép sai lệch, chỉ được
    khuyến khích khi đúng là hợp lý).

    alpha, residual_norm cũng được trả ra ngoài làm đặc trưng diagnostic hữu
    ích (có thể vẽ phân phối alpha của các mẫu F thật để interpretability --
    nếu phần lớn alpha lệch hẳn về 0 hoặc 1, mô hình đang "chọn phe" thay vì
    thực sự trộn, đáng ngờ).
    """

    def __init__(self, embed_dim: int, bottleneck_dim: Optional[int] = None):
        super().__init__()
        bottleneck_dim = bottleneck_dim or max(embed_dim // 4, 8)
        self.alpha_net = nn.Sequential(
            nn.Linear(embed_dim, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, 1),
        )

    def forward(
        self, h: torch.Tensor, mu_a: torch.Tensor, mu_b: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        alpha = torch.sigmoid(self.alpha_net(h))               # (B,1)
        reconstruction = alpha * mu_a + (1.0 - alpha) * mu_b    # (B,D), mu_a/mu_b là buffer const
        # mean() thay vì sum() trên chiều D: giữ magnitude không phụ thuộc embed_dim,
        # để so được với task/confusion_margin loss (phát hiện qua smoke test --
        # dùng sum() ban đầu cho residual ~90 trong khi task loss ~1.2, lấn át total loss).
        residual_norm = (h - reconstruction).pow(2).mean(dim=-1)  # (B,)
        return alpha.squeeze(-1), residual_norm


# =============================================================================
#  3) PROTOTYPE-AUGMENTED CLASSIFIER -- Công thức 3 (nửa đầu): logits kết hợp
#     linear softmax (an toàn, đã biết hoạt động) + khoảng cách tới prototype
#     HỌC ĐƯỢC (khác EMA bank ở trên).
# =============================================================================
class PrototypeAugmentedClassifier(nn.Module):
    """
    logits = linear_logits + sigmoid(gate) * (-||h - prototype_k||^2 / tau)

    `gate_logit` khởi tạo -5.0 (sigmoid(-5) ~= 0.007) -- ĐÚNG convention đã
    dùng cho beta_gate trong RLSTMCell/DSTALiteCell của project này (bias
    W_beta.bias = -5.0): bắt đầu gần như thuần linear-softmax (hành vi đã
    biết an toàn), để tín hiệu khoảng cách-prototype chỉ dần được tin dùng
    khi `class_prototypes` đã đủ ổn định qua vài epoch đầu, tránh phá vỡ
    quá trình học ngay từ epoch 1 khi prototype còn khởi tạo ngẫu nhiên.
    """

    def __init__(self, embed_dim: int, num_classes: int, tau: float = 1.0,
                 hidden_dim: Optional[int] = None, dropout: float = 0.25):
        super().__init__()
        # FIX (2026-08-21): ban đầu chỉ có 1 Linear thẳng (embed_dim -> num_classes),
        # thiếu hẳn lớp ẩn + ReLU mà classifier của RLSTMClassifier chính thức có
        # (192->96->5). Ablation cho thấy F1-F sập về ~0 dù TẮT hết cả 3 loss phụ
        # trợ (confusion-margin/mixture/repulsion) -- tức lỗi nằm ở kiến trúc
        # classifier thiếu capacity phi tuyến, không phải xung đột loss. Chỉ
        # nhánh linear_logits được thêm lớp ẩn -- nhánh proto_logits vẫn tính
        # khoảng cách trực tiếp trên embedding h (đúng ý nghĩa "khoảng cách tới
        # prototype trong không gian embedding", không nên đi qua biến đổi phi
        # tuyến riêng).
        hidden_dim = hidden_dim or embed_dim // 2
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.class_prototypes = nn.Parameter(torch.randn(num_classes, embed_dim) * 0.02)
        self.tau = tau
        self.gate_logit = nn.Parameter(torch.tensor(-5.0))

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        linear_logits = self.linear(h)
        diff = h.unsqueeze(1) - self.class_prototypes.unsqueeze(0)  # (B,K,D)
        dist_sq = diff.pow(2).sum(dim=-1)                            # (B,K)
        proto_logits = -dist_sq / self.tau
        gate = torch.sigmoid(self.gate_logit)
        logits = linear_logits + gate * proto_logits
        return logits, {
            "linear_logits": linear_logits,
            "proto_logits": proto_logits,
            "proto_gate": gate.detach(),
        }


# =============================================================================
#  4) CAMP CLASSIFIER -- CNN + BiLSTM chuẩn + Attention Pooling (tái sử dụng)
#     + 3 module mới ở trên
# =============================================================================
class CAMPClassifier(nn.Module):
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 96,
        dropout: float = 0.25,
        num_classes: int = 5,
        cnn_out_channels: int = 64,
        num_layers: int = 2,
        proto_momentum: float = 0.9,
        proto_tau: float = 1.0,
        mixture_class: int = F_IDX,
        mixture_components: Tuple[int, int] = (N_IDX, V_IDX),
        backbone: str = "rmc",
    ):
        """
        backbone: "rmc" (mac dinh, MOI 2026-08-21) dung BiRLSTM (keep/add
            softmax blend + beta-gated hybrid path -- giong het RLSTMClassifier
            chinh thuc) thay cho nn.LSTM chuan. Ly do doi: tong hop lai TAT CA
            thi nghiem trong du an (baseline don gian, DSTA A-D, CAMP-Net ban
            nn.LSTM) cho thay F1-F chi bao gio vuot 0.15 khi model CO co che
            RMC hybrid-blend + beta-gate trong cell hoi quy -- moi model
            KHONG co co che nay (kem ca CAMP-Net voi classifier da sua) deu
            bi chan duoi 0.11. "plain_lstm" giu lai code cu (nn.LSTM) de doi
            chieu/rollback neu can.
        """
        super().__init__()
        self.num_classes = num_classes
        self.mixture_class = mixture_class
        self.mixture_components = mixture_components
        self.backbone = backbone
        if backbone not in ("rmc", "plain_lstm"):
            raise ValueError(f"backbone phai la 'rmc' hoac 'plain_lstm', nhan: {backbone}")

        self.cnn = ECGFeatureExtractor(
            input_channels=input_size, output_channels=cnn_out_channels, dropout=dropout * 0.5,
        )
        if backbone == "rmc":
            self.birlstm = BiRLSTM(cnn_out_channels, hidden_size, num_layers=num_layers, dropout=dropout)
        else:
            self.bilstm = nn.LSTM(
                input_size=cnn_out_channels, hidden_size=hidden_size, num_layers=num_layers,
                batch_first=True, bidirectional=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        embed_dim = 2 * hidden_size
        self.attention_pool = AttentionPooling(embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.ema_centroids = ClassPrototypeBank(num_classes, embed_dim, momentum=proto_momentum)
        self.mixture_head = MixtureDecompositionHead(embed_dim)
        self.classifier = PrototypeAugmentedClassifier(embed_dim, num_classes, tau=proto_tau, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        update_prototypes: bool = False,
        return_internals: bool = False,
    ):
        features = self.cnn(x)
        r_fwd = r_bwd = None
        if self.backbone == "rmc":
            h_seq, _, r_fwd, r_bwd, _, _, _, _ = self.birlstm(features)
        else:
            h_seq, _ = self.bilstm(features)
        h_pooled, attn_weights = self.attention_pool(h_seq)
        h_embed = self.dropout(self.layer_norm(h_pooled))

        if update_prototypes and labels is not None and self.training:
            self.ema_centroids.update(h_embed, labels)

        logits, cls_internals = self.classifier(h_embed)

        a, b = self.mixture_components
        mu_a, mu_b = self.ema_centroids.get(a), self.ema_centroids.get(b)
        alpha, residual_norm = self.mixture_head(h_embed, mu_a, mu_b)

        if not return_internals:
            return logits

        internals = {
            "embedding": h_embed,
            "attention_weights": attn_weights,
            "mixture_alpha": alpha,
            "mixture_residual_norm": residual_norm,
            "mixture_ready": self.ema_centroids.is_ready(a, b),
            "class_prototypes": self.classifier.class_prototypes,  # Parameter thật, CÓ gradient
            "r_fwd": r_fwd, "r_bwd": r_bwd,  # None neu backbone="plain_lstm"
            **cls_internals,
        }
        return logits, internals


# =============================================================================
#  5) CAMP LOSS -- task loss (tái dùng FocalLoss có sẵn) + 3 thành phần mới,
#     tất cả trên trục HÌNH HỌC, trực giao với trục tần suất của Focal/weight.
# =============================================================================
class CAMPLoss(nn.Module):
    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        focal_gamma: float = 1.5,
        lambda_cm: float = 0.3,
        margin_cm: float = 1.0,
        lambda_mix: float = 0.1,
        lambda_rep: float = 0.1,
        margin_rep: float = 2.0,
        confusable_pairs: Sequence[Tuple[int, int]] = ((F_IDX, N_IDX), (F_IDX, V_IDX)),
        mixture_class: int = F_IDX,
        mixture_components: Tuple[int, int] = (N_IDX, V_IDX),
        warmup_epochs: int = 10,
        lambda_smooth: float = 0.003,
    ):
        """
        warmup_epochs: 3 loss term MOI (confusion-margin, mixture, repulsion)
            duoc nhan voi 1 he so ramp tang tuyen tinh tu 0 -> 1 qua
            warmup_epochs epoch dau, thay vi ap full-strength ngay epoch 1.
            Bai hoc rut ra tu that bai cua DSTA-Lite + Confusion-Margin/RR:
            them 1 loss/co che moi voi cuong do toi da ngay tu dau, trong luc
            backbone (va o DSTA la cong beta) con chua on dinh, gay xung dot
            hieu chinh thay vi giup ich. task_loss (FocalLoss) KHONG bi ramp --
            no la muc tieu chinh, luon full-strength.
        """
        super().__init__()
        self.task_loss = FocalLoss(alpha=class_weights, gamma=focal_gamma)
        self.lambda_cm = lambda_cm
        self.margin_cm = margin_cm
        self.lambda_mix = lambda_mix
        self.lambda_rep = lambda_rep
        self.margin_rep = margin_rep
        self.confusable_pairs = confusable_pairs
        self.mixture_class = mixture_class
        self.mixture_components = mixture_components
        self.warmup_epochs = max(warmup_epochs, 1)
        self.lambda_smooth = lambda_smooth
        self._ramp = 0.0  # cap nhat qua set_epoch() moi epoch

    def set_epoch(self, epoch: int) -> None:
        """Goi 1 lan moi epoch (epoch bat dau tu 1) truoc khi train batch dau tien."""
        self._ramp = min(epoch / self.warmup_epochs, 1.0)

    def _confusion_margin_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        total = logits.new_zeros(())
        n_pairs = 0
        for true_k, confuser_k in self.confusable_pairs:
            mask = targets == true_k
            if mask.sum() == 0:
                continue
            violation = self.margin_cm - (logits[mask, true_k] - logits[mask, confuser_k])
            total = total + F.relu(violation).mean()
            n_pairs += 1
        return total / max(n_pairs, 1)

    def _mixture_loss(self, targets: torch.Tensor, internals: Dict[str, torch.Tensor]) -> torch.Tensor:
        if not internals.get("mixture_ready", False):
            return targets.new_zeros((), dtype=torch.float32)
        mask = targets == self.mixture_class
        if mask.sum() == 0:
            return targets.new_zeros((), dtype=torch.float32)
        return internals["mixture_residual_norm"][mask].mean()

    def _repulsion_loss(self, internals: Dict[str, torch.Tensor]) -> torch.Tensor:
        prototypes = internals["class_prototypes"]  # (K, D), CÓ gradient
        k = self.mixture_class
        a, b = self.mixture_components
        d_ka = (prototypes[k] - prototypes[a]).pow(2).sum().clamp_min(1e-12).sqrt()
        d_kb = (prototypes[k] - prototypes[b]).pow(2).sum().clamp_min(1e-12).sqrt()
        return F.relu(self.margin_rep - d_ka) + F.relu(self.margin_rep - d_kb)

    def _smoothness_loss(self, logits: torch.Tensor, internals: Dict[str, torch.Tensor]) -> torch.Tensor:
        r_fwd, r_bwd = internals.get("r_fwd"), internals.get("r_bwd")
        if r_fwd is None or r_bwd is None:
            return logits.new_zeros(())  # backbone="plain_lstm" -- khong co r_t
        return (temporal_smoothness_loss(r_fwd) + temporal_smoothness_loss(r_bwd)) / 2.0

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, internals: Dict[str, torch.Tensor]):
        l_task = self.task_loss(logits, targets)
        l_cm = self._confusion_margin_loss(logits, targets)
        l_mix = self._mixture_loss(targets, internals)
        l_rep = self._repulsion_loss(internals)
        l_smooth = self._smoothness_loss(logits, internals)

        ramp = self._ramp
        total = (l_task + self.lambda_smooth * l_smooth
                 + ramp * (self.lambda_cm * l_cm + self.lambda_mix * l_mix + self.lambda_rep * l_rep))
        return total, {
            "task": l_task.detach(),
            "confusion_margin": l_cm.detach(),
            "mixture": l_mix.detach(),
            "repulsion": l_rep.detach(),
            "smooth": l_smooth.detach(),
            "ramp": torch.tensor(ramp),
            "total": total.detach(),
        }


# =============================================================================
#  6) FGSM ADVERSARIAL STEP -- để so sánh công bằng với ResMemECG chính thức
#     (checkpoint config: adversarial_training=True, adv_epsilon=0.02, adv_ratio=0.3)
#     KHÔNG có trong bản ablation_dsta_variants.py hiện tại -- đây chính là 1
#     trong những lệch chuẩn tôi vừa phát hiện khi audit; CAMP-Net cố tình làm
#     đúng ngay từ đầu để so sánh với 0.6279 không bị vênh vì thiếu adv-training.
# =============================================================================
def fgsm_adversarial_batch(
    model: CAMPClassifier, criterion: CAMPLoss,
    x: torch.Tensor, y: torch.Tensor, epsilon: float,
) -> torch.Tensor:
    x_req = x.clone().detach().requires_grad_(True)
    logits, internals = model(x_req, labels=y, update_prototypes=False, return_internals=True)
    loss, _ = criterion(logits, y, internals)
    grad = torch.autograd.grad(loss, x_req)[0]
    x_adv = (x_req + epsilon * grad.sign()).detach()
    return x_adv


# =============================================================================
#  7) EVAL THEO BATCH -- bài học rút ra từ crash CUDA OOM thật của
#     ablation_dsta_variants.py (đẩy nguyên 49,669 mẫu test qua 1 lần forward
#     -> OOM trên GPU 4GB). CAMP-Net áp dụng batched-eval NGAY TỪ ĐẦU.
# =============================================================================
EVAL_BATCH_SIZE = 256


@torch.no_grad()
def predict_in_batches(model: CAMPClassifier, X: torch.Tensor, device: torch.device) -> np.ndarray:
    model.eval()
    preds = []
    for i in range(0, len(X), EVAL_BATCH_SIZE):
        xb = X[i:i + EVAL_BATCH_SIZE].to(device)
        logits = model(xb, update_prototypes=False, return_internals=False)
        preds.append(logits.argmax(-1).cpu())
    return torch.cat(preds).numpy()


# =============================================================================
#  8) TRAIN + EVAL TRÊN DỮ LIỆU THẬT -- khớp ĐÚNG hyperparameter của checkpoint
#     ResMemECG chính thức (results/checkpoints/inter_best_rlstm.pt), để so
#     sánh với 0.6279 là công bằng, không lặp lại lỗi lệch chuẩn vừa tìm thấy.
#
#  epochs=45, patience=15, batch_size=128, lr=1e-3, weight_decay=1e-4,
#  focal_gamma=1.5, adversarial_training=True (epsilon=0.02, ratio=0.3).
#  LR scheduler: checkpoint config chỉ lưu min_lr=1e-5, không lưu
#  factor/patience của scheduler -- dùng ReduceLROnPlateau(factor=0.5,
#  patience=5, min_lr=1e-5) làm giả định hợp lý (phổ biến nhất), có ghi rõ
#  đây là giả định, không phải số chính xác đã dùng cho model gốc.
# =============================================================================
EPOCHS = 45
PATIENCE = 15
BATCH_SIZE = 128
SEED = 42
ADV_EPSILON = 0.02
ADV_RATIO = 0.3
DATA_DIR = Path("data/processed/splits")


def load_split(name: str):
    d = np.load(DATA_DIR / f"inter_{name}.npz")
    return d["X"].astype(np.float32), d["y"].astype(np.int64)


def class_weights_from(y_train: np.ndarray, num_classes: int = 5) -> torch.Tensor:
    counts = np.bincount(y_train, minlength=num_classes).astype(np.float64)
    counts = np.where(counts == 0, 1e-9, counts)
    w = counts.sum() / (num_classes * counts)
    w = np.clip(w, 0.5, 50.0).astype(np.float32)
    return torch.from_numpy(w)


def train_and_eval_camp_net(device: torch.device) -> Dict:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    X_test, y_test = load_split("test")
    print(f"Train={X_train.shape} Val={X_val.shape} Test={X_test.shape}")

    cw = class_weights_from(y_train).to(device)

    model = CAMPClassifier(
        input_size=1, hidden_size=96, dropout=0.25, num_classes=5,
        cnn_out_channels=64, num_layers=2,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"CAMP-Net parameters: {n_params:,} (ResMemECG chính thức: 505,038)")

    criterion = CAMPLoss(class_weights=cw, focal_gamma=1.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-5,
    )

    Xtr_t, ytr_t = torch.from_numpy(X_train), torch.from_numpy(y_train)
    Xva_t, yva_t = torch.from_numpy(X_val), torch.from_numpy(y_val)
    Xte_t, yte_t = torch.from_numpy(X_test), torch.from_numpy(y_test)

    n_train = len(Xtr_t)
    best_f1, best_state, patience_cnt = 0.0, None, 0
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        criterion.set_epoch(epoch)
        perm = torch.randperm(n_train)
        epoch_loss_comps: Dict[str, float] = {}
        n_batches = 0

        for i in range(0, n_train, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            xb = Xtr_t[idx].to(device)
            yb = ytr_t[idx].to(device)

            optimizer.zero_grad()
            logits, internals = model(xb, labels=yb, update_prototypes=True, return_internals=True)
            loss, comps = criterion(logits, yb, internals)

            if ADV_RATIO > 0:
                n_adv = max(int(len(xb) * ADV_RATIO), 1)
                adv_idx = torch.randperm(len(xb), device=device)[:n_adv]
                x_adv = fgsm_adversarial_batch(model, criterion, xb[adv_idx], yb[adv_idx], ADV_EPSILON)
                logits_adv, internals_adv = model(
                    x_adv, labels=yb[adv_idx], update_prototypes=False, return_internals=True,
                )
                loss_adv, _ = criterion(logits_adv, yb[adv_idx], internals_adv)
                loss = loss + loss_adv

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            for k, v in comps.items():
                epoch_loss_comps[k] = epoch_loss_comps.get(k, 0.0) + v.item()
            n_batches += 1
            del xb, yb, logits, internals, loss

        model.eval()
        val_preds = predict_in_batches(model, Xva_t, device)
        val_f1 = f1_score(yva_t.numpy(), val_preds, labels=[0, 1, 2, 3], average="macro", zero_division=0)
        scheduler.step(val_f1)

        comps_str = "  ".join(f"{k}={v / n_batches:.4f}" for k, v in epoch_loss_comps.items())
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"  epoch {epoch:2d} | val F1_macro = {val_f1:.4f} | lr={lr_now:.2e} | {comps_str}")

        if device.type == "cuda":
            torch.cuda.empty_cache()

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"  [early stop] hết patience={PATIENCE} ở epoch {epoch}")
                break

    train_time = time.time() - t0
    model.load_state_dict(best_state)
    model.eval()
    test_preds = predict_in_batches(model, Xte_t, device)

    y_true = yte_t.numpy()
    acc = accuracy_score(y_true, test_preds)
    f1_macro = f1_score(y_true, test_preds, labels=[0, 1, 2, 3], average="macro", zero_division=0)
    f1_per_class = f1_score(y_true, test_preds, labels=[0, 1, 2, 3], average=None, zero_division=0)

    print(f"\n[CAMP-Net] best val F1_macro={best_f1:.4f}  train_time={train_time:.1f}s")
    print(f"[CAMP-Net] TEST accuracy={acc:.4f}  F1_macro={f1_macro:.4f}")
    print(f"[CAMP-Net] TEST per-class F1: " +
          "  ".join(f"{CLASS_NAMES[i]}={f1_per_class[i]:.4f}" for i in range(4)))
    print(classification_report(y_true, test_preds, labels=[0, 1, 2, 3],
                                 target_names=["N", "S", "V", "F"], zero_division=0, digits=4))

    result = {
        "n_params": n_params, "accuracy": float(acc), "f1_macro": float(f1_macro),
        "f1_per_class": {CLASS_NAMES[i]: float(f1_per_class[i]) for i in range(4)},
        "train_time_sec": train_time, "best_val_f1_macro": float(best_f1),
    }

    official = {"f1_macro": 0.6279, "f1_per_class": {"N": 0.9643, "S": 0.4335, "V": 0.8728, "F": 0.2411}}
    print("\n" + "=" * 90)
    print(" SUMMARY: CAMP-Net vs official ResMemECG ")
    print("=" * 90)
    print(f"{'Model':<28} {'Params':>10} {'F1_macro':>10} {'F1-S':>8} {'F1-F':>8}")
    print(f"{'ResMemECG (official)':<28} {505038:>10,} {official['f1_macro']:>10.4f} "
          f"{official['f1_per_class']['S']:>8.4f} {official['f1_per_class']['F']:>8.4f}")
    print(f"{'CAMP-Net':<28} {n_params:>10,} {f1_macro:>10.4f} "
          f"{result['f1_per_class']['S']:>8.4f} {result['f1_per_class']['F']:>8.4f}")

    Path("results/logs").mkdir(parents=True, exist_ok=True)
    out_path = Path("results/logs/camp_net_results.json")
    with open(out_path, "w") as f:
        json.dump({"official_hmr_bilstm": official, "camp_net": result}, f, indent=2)
    print(f"\n[OK] Saved -> {out_path}")
    return result


# =============================================================================
#  9) SMOKE TEST -- dummy data, kiểm tra shape + gradient chảy đúng qua mọi
#     nhánh mới (mixture head, prototype classifier, confusion-margin,
#     repulsion) trước khi chạm vào dữ liệu thật. Convention giống hệt
#     __main__ của hmr_bilstm.py / hmr_dsta_lite.py.
# =============================================================================
def _smoke_test() -> None:
    torch.manual_seed(42)
    B, T, D, NUM_CLASSES = 16, 187, 1, 5
    HIDDEN = 96

    x = torch.randn(B, T, D)
    y = torch.randint(0, NUM_CLASSES, (B,))

    model = CAMPClassifier(input_size=D, hidden_size=HIDDEN, num_classes=NUM_CLASSES, num_layers=2)
    n_params = sum(p.numel() for p in model.parameters())
    print("=" * 70)
    print("1) Đếm tham số -- so với ResMemECG chính thức (505,038)")
    print("=" * 70)
    print(f"  CAMP-Net: {n_params:>10,d}")

    print("\n" + "=" * 70)
    print("2) Forward pass + shape check")
    print("=" * 70)
    model.train()
    logits, internals = model(x, labels=y, update_prototypes=True, return_internals=True)
    assert logits.shape == (B, NUM_CLASSES), logits.shape
    assert internals["mixture_alpha"].shape == (B,)
    assert internals["mixture_residual_norm"].shape == (B,)
    assert internals["class_prototypes"].shape == (NUM_CLASSES, 2 * HIDDEN)
    print(f"  logits: {tuple(logits.shape)}  OK")
    print(f"  mixture_alpha range: [{internals['mixture_alpha'].min():.3f}, "
          f"{internals['mixture_alpha'].max():.3f}]")
    print(f"  mixture_ready sau 1 batch: {internals['mixture_ready']} "
          f"(cần đủ mẫu N và V trong batch đầu)")

    print("\n" + "=" * 70)
    print("3) Loss + gradient check -- mọi nhánh mới phải có gradient chảy về")
    print("=" * 70)
    cw = torch.ones(NUM_CLASSES)
    criterion = CAMPLoss(class_weights=cw)
    criterion.set_epoch(criterion.warmup_epochs)  # ramp=1.0 de smoke test kiem tra dung ca 3 loss term
    loss, comps = criterion(logits, y, internals)
    print(f"  loss components: " + "  ".join(f"{k}={v.item():.4f}" for k, v in comps.items()))
    loss.backward()

    grad_ok = {
        "cnn.conv1.weight": model.cnn.conv1.weight.grad,
        "birlstm.fwd_layers[0].cell.W_x": model.birlstm.fwd_layers[0].cell.W_x.weight.grad,
        "classifier.linear[0].weight": model.classifier.linear[0].weight.grad,
        "classifier.linear[3].weight": model.classifier.linear[3].weight.grad,
        "classifier.class_prototypes": model.classifier.class_prototypes.grad,
        "classifier.gate_logit": model.classifier.gate_logit.grad,
        "mixture_head.alpha_net.0.weight": model.mixture_head.alpha_net[0].weight.grad,
    }
    for name, g in grad_ok.items():
        status = "OK (có gradient)" if (g is not None and g.abs().sum() > 0) else "LỖI (không có gradient!)"
        print(f"  {name:<32} {status}")

    print("\n" + "=" * 70)
    print("4) FGSM adversarial step -- kiểm tra không lỗi runtime")
    print("=" * 70)
    x_adv = fgsm_adversarial_batch(model, criterion, x, y, epsilon=0.02)
    assert x_adv.shape == x.shape
    print(f"  x_adv shape: {tuple(x_adv.shape)}  max |x_adv - x| = {(x_adv - x).abs().max().item():.4f}  OK")

    print("\nSmoke test PASSED.\n")


if __name__ == "__main__":
    _smoke_test()

    if DATA_DIR.exists() and (DATA_DIR / "inter_train.npz").exists():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nTìm thấy data thật tại {DATA_DIR} -- bắt đầu train CAMP-Net trên đó.")
        print(f"Device: {device}\n")
        train_and_eval_camp_net(device)
    else:
        print(f"\nKhông tìm thấy {DATA_DIR}/inter_train.npz -- chỉ chạy smoke test, "
              f"bỏ qua train trên dữ liệu thật.")
