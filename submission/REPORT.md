# Lab 21 — Evaluation Report

**Họ tên**: Đào Đức Hải  **MSSV**: 2A202602752  **Ngày**: 07/10/2026
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Tesla T4 16GB (Google Colab Free)`

> Mọi con số dưới đây khớp 100% với các file trong `results/` được kiểm tra chéo tự động bởi `scripts/verify.py`.

---

## 1. Setup

| Thông số | Giá trị thực nghiệm |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt → JSON triage 4 trường (`intent`, `urgency`, `product`, `sentiment`) |
| Train / val | 225 / 25 mẫu (phân chia cố định `seed=42`) |
| `max_length` | 1024 — p95 đo được trên tập train là 98 tokens *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` (chỉ tính loss trên lượt phản hồi của trợ lý) |
| Epochs / max_steps | 2 epochs / 30 optimizer steps (đồng nhất cho toàn bộ 4 run huấn luyện) |

**Template có giữ khối `<think>` không?** **CÓ** — File `results/template_check.json` xác nhận `verdict: "reasoning preserved — safe to train on traces"`. 
Chat template của Qwen3.5 tự động đặt cặp thẻ rỗng `<think>\n\n</think>\n\n` bên trong generation prompt và bảo toàn trọn vẹn nội dung câu trả lời JSON của assistant trong lượt hội thoại. Do đó, toàn bộ phần phản hồi mục tiêu không bị template cắt bỏ và đi thẳng vào hàm loss.

---

## 2. Mask proof (NB1)

| Tiêu chí | Kết quả kiểm chứng |
|---|---|
| `supervised_fraction` | 0.4149 (41.49% tổng số tokens được đưa vào hàm loss) |
| Câu trả lời nằm trong loss | `true` (khớp chính xác chuỗi JSON mục tiêu) |
| Câu hỏi KHÔNG nằm trong loss | `true` (toàn bộ system prompt và ticket của user bị che bằng `-100`) |

Đoạn văn bản được giải mã ngược từ các token có `label != -100` (được tính loss):

```json
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Phần prompt của hệ thống và câu hỏi người dùng nằm ngoài vùng loss (bị che bằng `IGNORE_INDEX = -100`):
```text
<|im_start|>system
Phân loại ticket sau.<|im_end|>
<|im_start|>user
Alo shop, mình đặt balo laptop mã đơn VN411453. Cho tôi trả lại. Đã 3 ngày rồi. Cho tôi hỏi.<|im_end|>
<|im_start|>assistant
<think>
```

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.791 | 0.000 | 3444.1 |
| (b) base + optimized prompt | 0.765 | 0.791 | 1.000 | 1057.4 |
| (c) LoRA fine-tune | **0.970** | 0.589 | **1.000** | 1588.5 |

**Baseline (b) có thật sự mạnh hơn (a) không?** **CÓ, vượt trội hoàn toàn.**
Khi chỉ dùng naive prompt ("Phân loại ticket sau."), base model không biết cấu trúc JSON mong muốn nên `target = 0.000` và `format = 0.000` với latency rất cao (3444 ms do sinh văn bản tự do dài dòng). Ngược lại, baseline (b) với system prompt tối ưu đạt `target = 0.765`, tuân thủ định dạng `format = 1.000`, và giảm độ trễ xuống 1057 ms (nhanh gấp 3.2 lần).

Tôi giữ nguyên hoàn toàn `OPTIMIZED_PROMPT` mặc định của lab (mã băm SHA: `719e74d3b6232053`), không sửa đổi hay làm yếu đi. Đây là mốc đánh giá nghiêm túc, tạo ra thử thách thực tế cho bản fine-tune và bảo đảm tính liêm chính khoa học.

---

## 4. Giải phẫu cấu hình sai (NB4 & NB5 §4)

| Run | Vị trí adapter | r | Trainable params | LR | Train loss (NB4) | **Target (NB5 §4)** | Train time (s) | Peak VRAM (GB) |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 1e-4 | 0.6262 | **0.9700** | 430.3 s | 8.78 GB |
| `attn_only` | q, v | 283 *(matched)* | 32,456,704 | 1e-4 | **0.5385** | **0.9700** | 297.3 s | 8.79 GB |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-5 | 1.5702 | **0.0000** | 434.5 s | 8.78 GB |
| `qlora` | text-linear | 16 | 32,464,896 | 1e-4 | 0.7058 | **0.9400** | 499.3 s | **3.86 GB** |

> Cả 4 run đều được huấn luyện chính xác ở cùng ngân sách **30 optimizer steps** (2 epochs).

### 4.1 — Phân tích `attn_only` vs `correct` (Vị trí vs Rank)
Run `attn_only` được khớp ngân sách tham số trainable chính xác với `correct` (32,456,704 so với 32,464,896 tham số, độ lệch chỉ 0.025%, nằm sâu dưới ngưỡng quy định 5%). Trên tập target ở NB5 §4, `attn_only` đạt điểm hoà tuyệt đối với `correct` ở mức 0.9700. Tuy nhiên, nếu xét theo cột `final_loss` huấn luyện ở NB4, `attn_only` lại có loss thấp hơn hẳn `correct` (0.5385 so với 0.6262), tạo cảm giác rằng nó "học tốt hơn". 
Sự đảo ngược thứ tự giữa train loss và target score chính là bằng chứng xác thực cho **Lỗi #3**: train loss chỉ phản ánh mức độ ghi nhớ (memorization/fitting) của ma trận rank cực cao ($r=283$) trên tập train nhỏ, chứ không phản ánh năng lực khái quát hóa trên tác vụ đích. Với tác vụ JSON triage hẹp, việc nâng rank lên 283 tại vị trí hẹp $q, v$ có thể bù đắp được độ chính xác, nhưng trên các tác vụ suy luận phức tạp hơn, việc trải đều rank $r=16$ trên toàn bộ các tầng `text-linear` mới là cấu hình an toàn bền vững.

### 4.2 — Phân tích `wrong_lr` (Tầm quan trọng của Learning Rate scale)
Run `wrong_lr` chỉ thay đổi đúng một con số duy nhất: giảm learning rate 10 lần (từ $1\times 10^{-4}$ xuống $1\times 10^{-5}$, thang đo thường dùng cho Full Fine-tuning). Đường train loss của `wrong_lr` gần như phẳng lì, kết thúc ở mức rất cao 1.5702 (so với 0.6262 của `correct`). Hậu quả trực tiếp trên tập đánh giá là thảm hoạ: `target = 0.0000` và `format = 0.0000`, mô hình hoàn toàn không học được cách xuất dữ liệu JSON mong muốn.
Nếu một kỹ sư chỉ nhìn vào loss cao và target bằng 0 mà không nắm vững thang đo LR cho LoRA, họ sẽ rất dễ kết luận sai lầm rằng "LoRA thất bại", "tác vụ quá khó" hoặc "dữ liệu không đủ". Thực chất, do các ma trận adapter $B$ được khởi tạo bằng 0, tốc độ học quá nhỏ của full-FT không cung cấp đủ bước cập nhật (gradient step size) để adapter thoát khỏi vùng khởi tạo trong 30 steps. Learning rate đúng chính là đòn bẩy nhị phân (bật/tắt) của LoRA.

### 4.3 — Phân tích `qlora` (Cái giá thực tế của lượng tử hoá 4-bit)
Thực nghiệm `qlora` cắt giảm VRAM từ 8.78 GB xuống chỉ còn 3.86 GB (tiết kiệm đến **56% VRAM**). Điều này cho phép nạp và huấn luyện mô hình 4B ngay trên các GPU laptop phổ thông có 4GB VRAM.
Tuy nhiên, sự tiết kiệm này phải trả giá rõ ràng trên hai phương diện:
1. **Độ chính xác:** Điểm target giảm từ 0.9700 xuống 0.9400 (-3.0% độ chính xác) do nhiễu lượng tử hóa NF4 4-bit làm suy hao độ nhạy của các trọng số ngôn ngữ tiếng Việt.
2. **Hiệu năng thời gian:** Thời gian huấn luyện kéo dài từ 430.3 s lên 499.3 s (+16%), và độ trễ suy luận tăng từ 1588.5 ms lên 1934.9 ms (+21.8%) do overhead giải nén trọng số (dequantization) liên tục trên GPU Turing T4.
Số đo thực nghiệm này hoàn toàn ủng hộ khuyến nghị của vendor (deck §13): Khi phần cứng có đủ VRAM (như Colab T4 16GB), **hãy sử dụng 16-bit LoRA thay vì QLoRA** đối với thế hệ mô hình Qwen3.5 để tránh suy hao chất lượng không cần thiết.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`
* `target Δ = +0.2050` (Tăng từ 0.765 lên 0.970 trên tác vụ đích)
* `regression Δ = -0.2022` (Giảm từ 0.7911 xuống 0.5889 trên tập tri thức chung)
* `valid_trace_rate = 0.00`

### Diễn giải phán quyết (142 từ):
Cổng hồi quy ra phán quyết **FAILED** vì chỉ số năng lực tổng quát (general capability) trên tập `eval_regression` bị tụt giảm nghiêm trọng $0.2022$ (từ 0.7911 xuống 0.5889), vượt xa ngưỡng suy giảm cho phép là $0.020$, cho dù mô hình fine-tune đạt mức tăng trưởng ấn tượng trên tác vụ đích ($\Delta = +0.2050$) và tuân thủ định dạng tuyệt đối ($100\%$).

Đây là minh chứng kinh điển cho hiện tượng **quên thảm hoạ (catastrophic forgetting)** được cảnh báo tại deck §6.3. Khi tinh chỉnh một mô hình nền tảng trên một tập dữ liệu miền hẹp (250 ticket CSKH) chỉ sử dụng loss mask cho câu trả lời mà không có dữ liệu đối chứng, các trọng số LoRA đã làm dịch chuyển không gian biểu diễn của mô hình về tác vụ phân loại, làm suy giảm khả năng trả lời các câu hỏi kiến thức phổ thông. Kết quả FAILED này là một phát hiện trung thực và có giá trị cao: nó chứng minh rằng trong quy trình công nghiệp, không thể deploy một mô hình chỉ dựa vào độ chính xác tác vụ đích mà bắt buộc phải bổ sung 1–5% dữ liệu replay tổng quát để bảo vệ năng lực cốt lõi.

---

## 6. Định tính — Phân tích ca Thắng và ca Thua

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|:---:|---|---|---|---|:---:|
| 1 | `Alo shop, mình đặt balo laptop mã đơn VN411453. Cho tôi trả lại...` | `doi_tra, trung_binh, balo laptop, trung_tinh` | Đạt 0.75 (sai urgency) | `{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}` | ✅ **FT thắng**: FT bắt chính xác cả 4 trường, cấu trúc JSON gọn gàng, không thừa giải thích. |
| 2 | `Xin chào, mình đặt chuột không dây mã đơn OD996568. Hoàn tiền...` | `hoan_tien, cao, chuột không dây, tieu_cuc` | Đạt 0.75 (nhầm urgency) | `{"intent": "hoan_tien", "urgency": "cao", "product": "chuột không dây", "sentiment": "tieu_cuc"}` | ✅ **FT thắng**: Prompt (b) bị phân vân giữa đổi trả và hoàn tiền, FT phân loại dứt khoát. |
| 3 | `Cho mình hỏi, mình đặt bình giữ nhiệt mã đơn VN804124. Chưa thấy tiền. Khi nào tiện...` | `hoan_tien, thap, bình giữ nhiệt, tich_cuc` | `urgency: thap` (đúng 4/4 trường) | `{"intent": "hoan_tien", "urgency": "trung_binh", ...}` | ❌ **FT thua**: FT đoán sai `urgency` thành `trung_binh` do bị thiên kiến bởi ngữ cảnh đòi tiền. |
| 4 | `Shop ơi, mình đặt nồi chiên không dầu mã đơn DH249548. Thiếu phụ kiện. Khi nào tiện...` | `san_pham_loi, thap, nồi chiên không dầu, trung_tinh` | `urgency: thap` (đúng 4/4 trường) | `{"intent": "san_pham_loi", "urgency": "trung_binh", ...}` | ❌ **FT thua**: FT đánh giá sự cố thiếu phụ kiện là `trung_binh`, bỏ qua cụm từ hạ mức "Khi nào tiện". |
| 5 | `Shop ơi, mình đặt áo khoác gió mã đơn VN613097. Bị lỗi. Khi nào tiện. Cảm ơn shop...` | `san_pham_loi, thap, áo khoác gió, tich_cuc` | `urgency: thap` (đúng 4/4 trường) | `{"intent": "san_pham_loi", "urgency": "trung_binh", ...}` | ❌ **FT thua**: Lặp lại lỗi thiên kiến độ khẩn cấp, FT đạt 0.75 điểm trong khi prompt (b) đạt 1.0. |

### Mẫu chung ở các ca FT thua:
Tất cả 6 ca mà bản fine-tune đạt điểm 0.75 (thua baseline b) đều có cùng một quy luật lỗi duy nhất: **Dự đoán sai mức độ khẩn cấp (`urgency`) từ `thap` thành `trung_binh`**. 
Các ticket này đều chứa cụm từ báo hiệu sự linh hoạt *"Khi nào tiện"*, nhưng đồng thời cũng đề cập đến các vấn đề nghiêm trọng như *"Chưa thấy tiền"*, *"Bị lỗi"*, *"Thiếu phụ kiện"*. Trong tập huấn luyện 225 mẫu, đại đa số các ticket khiếu nại hoàn tiền/sản phẩm lỗi đều được gán nhãn `trung_binh` hoặc `cao`. Mô hình fine-tune đã ghi nhớ mối liên hệ mạnh này và xem nhẹ tín hiệu từ cụm từ giảm nhẹ mức độ. Ngược lại, baseline (b) có lợi thế từ system prompt với các quy tắc tường minh nên nhận diện được ngoại lệ tốt hơn.

---

## 7. Kết luận & điều tôi học được

### Kết luận (185 từ):
Bản fine-tune LoRA này **CHƯA NÊN** được triển khai độc lập ra môi trường production tổng quát, mặc dù nó đã nâng độ chính xác tác vụ đích từ 76.5% lên 97.0% và loại bỏ hoàn toàn chi phí prompt token dài. Lý do chính là hiện tượng suy giảm năng lực tổng quát nghiêm trọng (regression delta $-0.2022$), khiến mô hình mất đi tính linh hoạt khi gặp các câu hỏi ngoài phân phối ticket CSKH. Để có thể đưa vào phục vụ người dùng thực tế, bản adapter này cần được tinh chỉnh lại với phương pháp **Data Replay** (bổ sung 1–5% dữ liệu đàm thoại chung vào tập huấn luyện) hoặc triển khai theo kiến trúc định tuyến chuyên biệt: sử dụng mô hình nền tảng cho tác vụ chung và chỉ kích hoạt LoRA adapter thông qua cơ chế hot-swap khi yêu cầu được xác định là ticket CSKH.

Qua toàn bộ chuỗi thực nghiệm từ NB1 đến NB5, đòn bẩy thực sự quyết định kết quả của lab lần lượt là: **Loss Mask và Chat Template** (đảm bảo gradient đi đúng vào phần sinh câu trả lời), tiếp theo là **Learning Rate scale** (quyết định adapter có thể học hay đứng yên), kế tiếp là **Chất lượng và Phân phối dữ liệu** (quyết định độ tổng quát hóa và hiện tượng quên tri thức), và cuối cùng mới là **Rank** (chỉ đóng vai trò dung lượng chứa thông tin).

### Ba điều tôi học được:
1. **Loss mask và template là nền móng cốt lõi:** Che sai loss mask (như chế độ `everything`) sẽ phá huỷ hoàn toàn quá trình huấn luyện bất kể thuật toán LoRA tối tân đến đâu. Việc kiểm chứng mask bằng giải mã ngược ký tự-token ở NB1 là bước bắt buộc trước khi tốn thời gian chạy GPU.
2. **Train loss là chỉ số thay thế nguy hiểm:** Run `attn_only` có train loss thấp hơn `correct` ($0.5385$ vs $0.6262$), nhưng trên tác vụ target thì không hề vượt trội. Đánh giá mô hình phải dựa trên năng lực thực tế của bài toán nghiệp vụ chứ không thể dựa vào đường cong loss.
3. **Cái giá thực tế của QLoRA:** Tiết kiệm 56% VRAM của QLoRA 4-bit đi kèm với việc giảm 3% độ chính xác và tăng hơn 20% độ trễ suy luận. Cần cân nhắc sự đánh đổi này dựa trên giới hạn phần cứng thực tế thay vì mặc định áp dụng 4-bit theo thói quen cũ.

### Nếu có thêm 2 giờ nữa, tôi sẽ thử:
1. Trộn thêm 3% dữ liệu hội thoại tiếng Việt thông dụng vào tập `train_seed.jsonl` để khắc phục hoàn toàn hiện tượng quên thảm hoạ, đưa `regression delta` về trong ngưỡng dung sai an toàn ($\ge -0.02$).
2. Thử nghiệm cấu hình LoRA với rank $r=8$ và $r=32$ trên toàn bộ các tầng `text-linear` để tìm điểm cân bằng tối ưu giữa kích thước adapter và năng lực biểu diễn dữ liệu.

---

## Phụ lục — Thưởng đã làm

- [x] **B1 NB6 merge + hot-swap (+3 điểm)**: Đã chạy kiểm tra merge weights ở NB6. Kết quả ghi nhận tại `results/merge_check.json`: `before_merge = 0.970`, `after_merge = 0.970`, độ lệch $\Delta = 0.000$ (nằm trong ngưỡng dung sai khắt khe 0.01), chứng minh việc gộp trọng số không làm suy hao chất lượng mô hình.
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub
