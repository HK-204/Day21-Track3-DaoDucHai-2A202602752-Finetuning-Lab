# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**
Sự chênh lệch giữa train loss và task target ở NB4: run `attn_only` ép loss huấn luyện xuống thấp hơn hẳn `correct` (0.5385 so với 0.6262), nhưng trên tập target thì không hề vượt trội (hoà ở 0.9700). Tôi cũng rất bất ngờ khi thấy hiện tượng quên thảm hoạ (catastrophic forgetting) diễn ra mạnh mẽ đến vậy chỉ sau 30 optimizer steps: điểm kiểm tra tri thức chung tụt từ 0.7911 xuống 0.5889 (-0.2022).

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**
Tôi mất nhiều thời gian nhất ở công đoạn sinh văn bản (generation) đánh giá và chạy 3 run đối chứng ở NB4 (~60 phút trên GPU Colab T4). Ban đầu tôi dự đoán bước nạp mô hình và tải weights (9.32 GB) sẽ là nút thắt cổ chai, nhưng thực tế việc greedy decode lặp lại nhiều lần trên tập eval mới là thứ tiêu tốn nhiều thời gian GPU nhất.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**
Trước đây tôi tin rằng QLoRA 4-bit luôn là lựa chọn hiển nhiên cho mọi tác vụ vì "tiết kiệm VRAM mà không mất mát gì", và cho rằng rank càng cao thì mô hình càng thông minh. Giờ tôi nhận ra QLoRA trên Qwen3.5 phải trả giá bằng 3% độ chính xác và tăng hơn 20% độ trễ suy luận; còn rank cao ở vị trí hẹp chỉ là công cụ ép ghi nhớ (memorization) cục bộ chứ không đại diện cho năng lực tổng quát.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**
Tôi dùng AI assistant để điều tra cấu trúc toàn bộ repo, sửa lỗi dòng kết thúc Windows CRLF làm sai lệch checksum tập eval, chuẩn hoá mã hoá UTF-8, và tái cấu trúc notebook runner trên Colab để tự động đóng gói kết quả. Điểm AI ban đầu suýt sai là cân nhắc train thử nghiệm trên GPU máy laptop (RTX 3050 4GB), nhưng sau đó đã kịp thời tính toán chính xác mức chiếm dụng VRAM của Qwen3.5-4B (~10GB) để kiên quyết chuyển sang Colab T4 nhằm tránh lỗi CUDA OOM.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**
Tôi sẽ **đóng băng một tập đánh giá chất lượng cao và thiết kế prompt tối ưu để đo baseline (b) trước tiên**. Nếu baseline prompt đã giải quyết tốt bài toán trong giới hạn chi phí và độ trễ, tôi sẽ khuyến nghị khách hàng dùng prompt thay vì fine-tune. Nếu bắt buộc fine-tune, bước đầu tiên sau đó là kiểm chứng giải mã ngược loss mask (char-to-token) để chắc chắn gradient chỉ đi vào phần câu trả lời mong muốn.
