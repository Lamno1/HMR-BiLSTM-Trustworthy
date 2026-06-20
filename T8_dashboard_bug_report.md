# BÁO LỖI DASHBOARD T8 + VIỆC CÒN TREO — gửi coder

**Checkpoint chuẩn:** `inter_best_rlstm.pt`, SHA-1 `3d5434314e7132f07913462a7bd7ab6095b2e3a3`.
**Nguyên tắc:** số thô trước, verdict sau. Không gọi "OOD". Đối chiếu dashboard với số gốc từng module trước khi tin.

---

## 🔴 LỖI DASHBOARD (T8 chạy nhưng GOM SAI — module không hỏng, khâu đọc file hỏng)

### LỖI 1 (nghiêm trọng nhất): AutoAttack = 0.0000 trong dashboard — SAI
- Table 3 + Table 5 ghi `AutoAttack Macro F1 = 0.0000`.
- **Số thật đã chạy:** Full HMR robust accuracy 0.707 / 0.683 / 0.637 ở ε = 0.02 / 0.03 / 0.05. KHÔNG phải 0.
- Dashboard không tìm thấy file kết quả AutoAttack → điền 0 thay vì đọc đúng.
- **Hậu quả:** "AutoAttack F1 = 0" đọc ra là "model sụp hoàn toàn" — NGƯỢC 180° với kết quả thật (không masking, robust tốt). Tuyệt đối không để vào paper.
- **Sửa:** dashboard phải đọc đúng file AutoAttack (`autoattack_results.csv`), nếu không tìm thấy thì để `N/A` + cảnh báo, KHÔNG điền 0.

### LỖI 2 (nghiêm trọng): Calibration toàn N/A — dashboard không đọc được T1b
- Table 2 ECE/MCE/NLL/Brier đều `N/A`.
- **Số thật đã chạy:** ECE 0.0499→0.0391, MCE 0.238→0.369, Brier 0.1711→0.1687, conditional_ece (N 0.084, S 0.326, V 0.068, F 0.589).
- Nguyên nhân: calibration lưu ở run `180710`, dashboard đọc run `061207` → không thấy.
- Calibration là TRỤ CỘT của bài "trustworthy" — để N/A là bỏ trống phần quan trọng nhất.
- **Sửa:** xem LỖI 4 (gốc rễ).

### LỖI 3: "AUROC-OOD" — tên đã cấm 2 lần, vẫn xuất hiện
- Table 4 + Scorecard ghi `AUROC-OOD (Mean) 0.6341 / 0.6228`.
- Đã thống nhất: KHÔNG gọi "OOD". Đây là corruption set (cùng phân phối + nhiễu), không phải out-of-distribution thật.
- 0.63 ≈ ngẫu nhiên (0.5) → model phát hiện corruption YẾU, không phải "OOD detection tốt".
- **Sửa:** đổi mọi `OOD` / `AUROC-OOD` → `corruption detection AUROC` trong dashboard + `mc_results.json` + `ensemble_results.json`. Đọc 0.63 = yếu, ghi đúng.

### LỖI 4 (GỐC RỄ của lỗi 1+2): dashboard trộn run-id
- Các module lưu rải rác nhiều run: SHAP `160039`, calibration `180710`, uncertainty/robustness `061207`...
- Dashboard quét MỘT run (`061207`) → module ở run khác = N/A hoặc 0.
- **Sửa (chọn 1):**
  - (a) Dashboard đọc đúng run mới nhất của TỪNG module, hoặc
  - (b) Gom mọi `results.json` về một nơi, dashboard đọc theo `checkpoint_hash = 3d5434...` thay vì theo run-id.
- Sau khi sửa: AutoAttack phải ra 0.707 (không 0), calibration ra ECE 0.0391 (không N/A).

---

## ✅ MODULE CHẠY ĐÚNG (số gốc để đối chiếu dashboard sau khi sửa)

| Module | Số gốc (dashboard phải khớp) |
|---|---|
| Classification | acc 0.8784, 4-class macro-F1 0.5644, recall-F 0.724 |
| Calibration | ECE 0.0499→0.0391, MCE 0.238→0.369; cond-ECE: N 0.084 / S 0.326 / V 0.068 / F 0.589 |
| AUC per-class | V 0.995, F 0.933, S 0.899 |
| FGSM (ε0.02) | acc 0.9637, F1 0.8425 |
| PGD (ε0.02) | acc 0.8439, F1 0.4103, ASR 0.039 |
| CW (L2) | total ASR 0.105, adv F1 0.894 (S ASR 0.28, V ASR 0.0) |
| AutoAttack | robust acc 0.707/0.683/0.637 @ ε 0.02/0.03/0.05; **KHÔNG masking** (APGD phá chính, Square thêm ~0) |
| MC-dropout | per-class entropy: N 0.494 / S 0.646 / V 0.326 / F 0.896 (F cao nhất = đúng) |

---

## ⚠️ VIỆC CÒN TREO — chưa đóng, làm trước khi viết phần liên quan

### A. MC-dropout: tách entropy ĐÚNG vs SAI
- Bảng hiện tại là entropy trung bình toàn lớp (cả đúng+sai).
- Cần: entropy trên beat F model **đoán SAI** cụ thể. Nếu cao → model "biết khi nó sai" (tốt). Nếu thấp → tự tin sai.
- Xu hướng tổng đã tốt (F entropy 0.896 cao nhất), nhưng phải xác nhận cao ĐÚNG LÚC sai.

### B. Ensemble: nghi DEGENERATE — phải kiểm trước khi gọi "deep ensemble" là đóng góp
- Mutual Info ID = 0.0428 → THẤP → 3 model đồng ý gần hết → nghi degenerate (adversarial training làm members giống nhau).
- Ensemble macro-F1 0.4937 < single full model 0.5644 (?!) → nghi 2 member mới (seed 123/456) yếu hơn primary, kéo trung bình xuống.
- **Cần:** (1) disagreement rate per-class (thấp N, cao S/F = lành mạnh; <2% mọi lớp = degenerate); (2) F1 từng member riêng.
- Nếu degenerate → ghi caveat trung thực, KHÔNG thổi "deep ensemble" thành đóng góp lớn.

### C. Corruption robustness: chưa thấy đường xuống cấp
- Đã chạy (`corruption_degradation.png` có), nhưng chưa gửi số đọc.
- Cần: đường xuống cấp 4 loại nhiễu + **recall-S/F** (không chỉ acc tổng) + so shift vs gaussian (shift hại S/F mạnh hơn = nhất quán coupling, không phải bug).
- Tên: "corruption robustness / sensitivity to signal degradation", KHÔNG "OOD".

---

## THỨ TỰ SỬA
1. LỖI 4 (gốc): dashboard đọc đúng run/hash → tự khắc sửa LỖI 1 + 2.
2. LỖI 3: đổi tên OOD → corruption khắp nơi.
3. Chạy lại dashboard, **đối chiếu từng ô với bảng số gốc ở trên** (AutoAttack 0.707, ECE 0.0391, CW 0.894...). Không N/A, không 0 giả.
4. Đóng việc treo: MC entropy đúng/sai (A), ensemble disagreement (B), corruption đường xuống cấp (C).
5. Verify mọi results.json mang hash `3d5434...`.

## NGUYÊN TẮC (nhắc lại)
- Dashboard điền 0 khi thiếu file = NGUY HIỂM (kết luận ngược). Thiếu → N/A + cảnh báo, không bao giờ 0.
- "Module chạy đúng" ≠ "dashboard gom đúng" — luôn đối chiếu dashboard với số gốc.
- Không gọi corruption là OOD. 0.63 AUROC = yếu, đọc đúng.
- Ensemble chưa được gọi đóng góp tới khi chứng minh không degenerate.

---

# PHỤ LỤC: ROOT CAUSE Ở MỨC CODE (sau khi đọc source trong zip)

## LỖI 1 thực chất: AutoAttack — CHẠY SAI FILE, không phải chỉ run-id
- Bạn chạy `evaluate_autoattack.py` (root) → ghi ra **`results/robustness/autoattack_results.csv`** (CSV, cột `Overall_ASR`/`ASR_V`/`Robust_Acc`).
- Dashboard (`evaluate_trustworthiness.py` dòng ~129) đọc **`outputs/<run_id>/robustness/autoattack_results.json`** (JSON, field `autoattack_asr`/`autoattack_f1_macro`).
- → File CSV bạn tạo ≠ file JSON dashboard tìm. Hai script khác nhau: `evaluate_autoattack.py` (đã chạy) vs `robustness/auto_attack.py` (CHƯA chạy, đây mới là cái ghi JSON đúng format).
- **Sửa (chọn 1):**
  - (a) Chạy `robustness/auto_attack.py` để sinh `autoattack_results.json` đúng format/đường dẫn dashboard cần, HOẶC
  - (b) Sửa dashboard đọc từ CSV `results/robustness/autoattack_results.csv` và map cột `Overall_ASR`→asr, tính f1 từ robust_acc.
- ⚠️ Khuyến nghị (a) — vì `robustness/auto_attack.py` ghi đầy đủ `autoattack_f1_macro`, `autoattack_asr_per_class`, masking_gap. Nhưng phải verify nó cho CÙNG số với `evaluate_autoattack.py` đã chạy (robust acc 0.707 @ ε0.02).

## LỖI 2 thực chất: calibration N/A = RUN-ID lệch (xác nhận bằng code)
- `temperature_scaling.py` ghi ĐÚNG field (`ece_before/after`, `mce_*`, `nll_*`, `brier_*`) vào ĐÚNG path (`paths["out_calib"]/results.json`). Field + path đều khớp dashboard.
- Nguyên nhân N/A: `get_run_id()` trong `configs/paths.py` đọc env `TRUSTWORTHY_RUN_ID`; nếu KHÔNG set → config `run_id: auto` → sinh **timestamp mới mỗi lần chạy**.
- → Calibration chạy ở run X, dashboard chạy ở run Y (khác timestamp) → đọc thư mục rỗng → N/A.
- **Sửa:** set `TRUSTWORTHY_RUN_ID` GIỐNG NHAU cho MỌI lệnh (calibration, SHAP, uncertainty, robustness, dashboard). Hoặc set `run_id` cố định trong `experiment_config.yaml` (không để `auto`).
- Kiểm: trước khi chạy dashboard, `echo $env:TRUSTWORTHY_RUN_ID` phải = run chứa các results.json thật (vd `v1.0_20260616_061207`), và run đó phải có ĐỦ calibration/results.json (nếu calibration chạy ở run khác → copy sang, hoặc chạy lại calibration với env này).

## LỖI 3: "AUROC-OOD" hardcode trong dashboard
- Dòng Table 4 header + scorecard: chữ `"AUROC-OOD (Mean)"`, `"MC Dropout AUROC-OOD"` hardcode.
- Đọc field `ood_detection_auroc` từ mc_results.json/ensemble_results.json.
- **Sửa:** đổi chữ → `"Corruption-Detection AUROC"`; đổi field trong mc_dropout.py/deep_ensemble.py từ `ood_detection_auroc` → `corruption_detection_auroc`. Đọc 0.63 = YẾU (gần ngẫu nhiên), không phải tốt.

## LỖI 5 (MỚI phát hiện): nhãn "Mean Variance" SAI
- Dashboard Table 4 cột "Mean Variance" đọc field `id_mean_mi` (Mutual Information).
- MI ≠ Variance. Sai nhãn.
- **Sửa:** đổi tên cột → "Mean MI" / "Mutual Information".

## LỖI 6 (gốc thiết kế): dashboard điền 0.0 khi thiếu file
- Mọi `m.get(key, 0.0)` → khi module thiếu, ra 0.0 thay vì N/A.
- 0.0 đọc thành "model sụp" (đặc biệt AutoAttack F1=0 = thảm họa diễn giải).
- **Sửa:** khi file/field thiếu → in `N/A` + warning rõ "file X không tìm thấy ở run Y", KHÔNG bao giờ 0.0 cho metric.

## QUY TRÌNH ĐÚNG ĐỂ CHẠY DASHBOARD
```
# 1. Set run-id cố định cho TẤT CẢ
$env:TRUSTWORTHY_RUN_ID="v1.0_FINAL"
# 2. Chạy lại MỌI module dưới cùng run-id này (để mọi results.json về một chỗ):
venv\Scripts\python -m calibration.temperature_scaling
venv\Scripts\python -m explainability.shap_analysis
venv\Scripts\python -m explainability.integrated_gradients
venv\Scripts\python -m uncertainty.mc_dropout
venv\Scripts\python -m uncertainty.deep_ensemble
venv\Scripts\python -m uncertainty.evaluate_corruptions
venv\Scripts\python -m robustness.cw_attack
venv\Scripts\python -m robustness.auto_attack      # <-- cái này, KHÔNG phải evaluate_autoattack.py
# 3. Dashboard cuối:
venv\Scripts\python -m evaluate_trustworthiness
# 4. Đối chiếu từng ô dashboard với số gốc (AutoAttack 0.707, ECE 0.0391, CW 0.894)
```
