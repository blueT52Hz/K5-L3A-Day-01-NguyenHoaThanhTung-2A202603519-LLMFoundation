# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603519
- **Họ và tên:** Nguyen Hoa Thanh Tung
- **SEED** (in ở ô đầu notebook): 3964 · Thử thách #5

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

Bộ kiểm tra: **15/15 đạt**, không thiếu dự đoán nào.

## 2. Kết quả chính

| Cách | F1 test | F1 SUB (40) | p95 ms/tin | USD/1k tin |
| --- | --- | --- | --- | --- |
| 1. BoW + LogReg (tự code) | 0.896 | 0.889 | 0.06 | 0 |
| 2. BoW + MLP (deep learning) | 0.938 | 0.947 | 4.57 | 0 |
| 3a. Encoder (MiniLM) + kNN | 0.922 | 0.895 | 41.91 | 0 |
| 3b. Encoder (MiniLM) + LogReg | **0.953** | 0.889 | 30.72 | 0 |
| 3c. Encoder (MiniLM) + MLP head | 0.945 | 0.947 | 32.42 | 0 |
| 4a. LLM nhỏ zero-shot (gpt-6-luna) | — | 0.000* | 1970.19 | 0.0064 |
| 4b. LLM lớn zero-shot (gpt-6.1-terra) | — | 0.000* | 1352.63 | 0.0639 |
| 4c. LLM nhỏ few-shot | — | 0.000* | 1293.61 | 0.0193 |

\* Cả 3 dòng LLM đều có `invalid = 40/40`: model trả về chuỗi rỗng ở mọi tin, nên `parse_label` ra `None` và mọi tin bị coi là ham. Chi phí đo được gần như chỉ là chi phí token input, tức output khoảng 0 token. Nguyên nhân khả năng cao: `max_completion_tokens = 3` không đủ, model kết thúc với `finish_reason = "length"` trước khi kịp sinh nội dung. Vì vậy F1 của LLM ở đây **không phản ánh năng lực thật** của model.

## 3. Dự đoán và thực tế

**1. `llm_models` / `llm_fewshot`: lệch nhiều nhất**
- **Dự đoán:** model lớn đạt F1 khoảng 0.95-0.98, model nhỏ khoảng 0.90-0.95; model lớn đắt gấp khoảng 10 lần; few-shot đắt gấp khoảng 2.5-3.5 lần.
- **Thực tế:** cả 3 cách đều có F1 = 0 với invalid = 40. Riêng phần chi phí thì khớp: 0.0639 / 0.0064 = **10 lần** và 0.0193 / 0.0064 = **3 lần**.
- **Vì sao lệch:** mình không lường trước được giới hạn 3 token lại làm câu trả lời rỗng. API vẫn trả về kết quả và tính tiền bình thường, không báo lỗi gì. Nếu `parse_label` không trả `None` và không có bộ đếm `n_invalid`, lỗi này đã bị che mất.

**2. `ml_lr`: lr = 50**
- **Dự đoán:** loss răng cưa, dao động mạnh, loss cuối không ổn định và có thể cao hơn lr = 0.5.
- **Thực tế:** loss vọt lên khoảng 2.5 ở epoch đầu, sau đó giảm thẳng xuống **0.007**, thấp nhất trong 3 giá trị lr. (lr = 0.001 cho loss 0.625; lr = 0.5 cho 0.085.)
- **Vì sao lệch:**
  - Bước cập nhật đầu tiên đẩy `b` và các trọng số về rất âm, nên mọi p ≈ 0. Khi đó tin spam bị phạt ở mức tối đa: 0.123 × 20.7 ≈ 2.5.
  - Sigmoid bão hoà khiến gradient chỉ còn đến từ các tin spam, và dữ liệu BoW gần như tách được tuyến tính, nên BCE không có đáy hữu hạn. Bước lớn chỉ giúp đi xuống nhanh hơn, không bị "nảy qua nảy lại".

**3. `tradeoff`: đường Pareto**
- **Dự đoán:** 1, 2 và 3b nằm trên đường Pareto.
- **Thực tế:** chỉ có **1 và 2**. Trên tập SUB, 3b chỉ đạt F1 = 0.889, thấp hơn 2 (0.947) mà còn chậm hơn khoảng 7 lần.
- **Vì sao lệch:** mình đã dựa vào F1 trên toàn tập test (ở đó 3b cao nhất, 0.953), trong khi biểu đồ dùng F1 trên **SUB 40 tin**. Trên tập này, mỗi tin sai làm F1 thay đổi khoảng 2.5 điểm, nên thứ tự giữa các cách dễ bị đảo.

Ngoài ra, ở `dl_relu` mình đoán mạng bỏ ReLU sẽ kém MLP một chút. Thực tế cả hai đều đạt **0.938**. Điều này cho thấy phần cải thiện của MLP so với LogReg (từ 0.896 lên 0.938) đến từ cách train (Adam, mini-batch), không phải từ phi tuyến.

## 4. Phân tích 50 case khó

| Model | F1 test thường (SUB) | F1 50 case khó | Acc 50 case khó |
| --- | --- | --- | --- |
| 1. BoW + LogReg | 0.889 | 0.378 | 0.54 |
| 2. BoW + MLP | 0.947 | 0.432 | 0.58 |
| 3a. Emb + kNN | 0.895 | **0.727** | **0.70** |
| 3b. Emb + LogReg | 0.889 | 0.583 | 0.60 |
| 3c. Emb + MLP head | 0.947 | 0.571 | 0.58 |
| 4a / 4d. LLM nhỏ | 0.000* | 0.000* | 0.42* |

\* LLM đoán mọi tin là ham do lỗi trả về rỗng, nên accuracy = 21/50 tin ham = 0.42.

- **Tụt nhiều nhất là BoW:** BoW + MLP giảm từ 0.947 xuống 0.432. Nhóm khó nhất với BoW là **thao túng / lừa đảo** (12-25% đúng), **spam giọng tự nhiên** (10-20%) và **né từ khoá** (17%). Ngược lại, BoW đúng 100% ở nhóm "tưởng spam" và "nói về spam", vì mô hình vốn dè dặt (bias âm).
- **Embedding giữ tốt hơn:** nhóm né từ khoá đạt 83%, nhưng nhóm "nói về spam" chỉ 25-50%. Lý do là tin cảnh báo lừa đảo có nghĩa gần với spam.
- **Điểm trên test thường quá lạc quan** vì tập test có cùng phân phối với train và chứa nhiều spam gần trùng nhau.

Có **6 case mà mọi model đều sai**: 5 tin thuộc nhóm "spam giọng tự nhiên" và 1 tin thuộc nhóm "thao túng". Hai ví dụ:
- **#14:** *"Hi, I'm working from home and making about £300 a day, happy to show you how, no experience needed…"* (lừa đảo tuyển việc).
  - Với BoW, các từ thân mật như `I'm`, `happy`, `you` cộng với bias âm lấn át token `£`.
  - Embedding xếp câu này vào vùng "kể chuyện cá nhân / công việc", gần với tin ham.
  - Kiểu lừa đảo tuyển việc gần như không có trong dataset (spam UK khoảng năm 2012), nên LogReg chưa từng học vùng nghĩa này là spam.
- **#22:** *"Your bank account has been temporarily suspended due to unusual activity. Verify your identity…"* (mạo danh ngân hàng).
  - Tin không có từ khoá trúng thưởng nào (free, win, prize, call).
  - Dataset hầu như không có tin phishing ngân hàng, nên cả BoW lẫn Embedding + LogReg đều đặt nó vào vùng ham.
  - Nếu LLM hoạt động đúng, đây là loại tin mà LLM có lợi thế nhất, nhờ kiến thức về lừa đảo có từ quá trình pre-train.

## 5. Thử thách cá nhân

**Đề #5:** viết 4 tin nhắn (2 spam né từ khoá, 2 ham chứa "free" hoặc "win"), rồi so sánh dự đoán của BoW với Embedding + LogReg.

**Cách làm:** dùng lại `predict_proba` của BoW (`w`, `b`) và của Encoder + LogReg (`w_e`, `b_e`, đã chuẩn hoá theo `mu` và `sd`), in thêm xác suất spam cho từng tin.

| Tin | Thật | BoW | Enc + LR |
| --- | --- | --- | --- |
| "Hi, your parcel is on hold because of an unpaid delivery fee. Please update your payment details at this link…" | spam | ham ✗ (p=0.15) | ham ✗ (p=0.10) |
| "Earn 3 million a week working from home, only 2 hours a day, no experience needed…" | spam | ham ✗ (p=0.39) | ham ✗ (p=0.11) |
| "Are you free this Saturday? We could watch the final together at my place" | ham | ham ✓ (p=0.05) | ham ✓ (p=0.01) |
| "Haha I win the bet, so dinner is free for me tonight. See you at 7" | ham | ham ✓ (p=0.02) | ham ✓ (p=0.01) |

**Kết quả:** cả hai đều đúng 2/4. Mình đã đoán Embedding sẽ đúng 3-4/4, nên kết quả này sai so với dự đoán.

**Giải thích:**
- **Ham có từ khoá spam không lừa được model.** Ngữ cảnh thân mật (`are you`, `Saturday`, `I`, `me`, `tonight`, `haha`) đủ mạnh để lấn át `free` và `win`. Embedding còn hiểu "free" ở đây nghĩa là *rảnh rỗi*.
- **Spam né từ khoá lừa được cả hai model.** BoW không thấy từ khoá quen thuộc nào. Embedding hiểu được *chủ đề* của câu (giao hàng, việc làm), nhưng LogReg chỉ học những **kiểu spam có trong dữ liệu train** (trúng thưởng, nhạc chuông, đầu số trả phí). Phishing giao hàng và lừa đảo tuyển việc là kiểu spam mới, nên mô hình không nhận ra.
- **Bài học:** embedding tốt chưa đủ. Muốn bắt được kiểu spam mới thì cần thêm dữ liệu có nhãn của kiểu đó, hoặc dùng LLM zero-shot.

## 6. Phương án cho sản phẩm thật

**Yêu cầu:** 10 triệu SMS/ngày (trung bình khoảng 116 tin/giây), p95 < 50 ms/tin.

- **Đạt SLA:** 1, 2, 3a, 3b, 3c, với p95 lần lượt là 0.06 / 4.57 / 41.9 / 30.7 / 32.4 ms. Cả ba cách LLM đều có p95 khoảng **1.3-2.0 giây**, gấp khoảng 30-40 lần SLA.
- **Chi phí nếu dùng LLM:** gpt-6-luna tốn $0.0064 cho 1k tin. Với 10 triệu tin/ngày, chi phí khoảng **$64/ngày, tức khoảng $1,900/tháng**. Với gpt-6.1-terra là khoảng **$19,000/tháng**. Và cả hai đều không đạt SLA.
- **Phương án chọn: router 2 tầng.**
  - **Tầng 1, BoW + MLP (2):** xử lý mọi tin. F1 đạt 0.938, F1 SUB đạt 0.947 (cao nhất), p95 là 4.6 ms, khoảng 21,000 tin/giây trên một máy, chi phí 0. Tin nào có p < 0.1 hoặc p > 0.9 thì kết luận ngay.
  - **Tầng 2, Encoder + LogReg (3b):** chỉ xử lý những tin lưỡng lự, p ∈ [0.1, 0.9]. F1 test đạt 0.953, p95 là 30.7 ms. Một tiến trình chỉ xử lý khoảng 70 tin/giây, nhưng chỉ phải xử lý vài phần trăm lưu lượng, nên vẫn đủ.
  - **LLM** chỉ dùng **ngoài luồng chính**: duyệt lại những tin bị gắn cờ, hoặc gán nhãn dữ liệu mới (phishing ngân hàng, lừa đảo tuyển việc) để train lại tầng 1 và tầng 2.
- **Đánh đổi chấp nhận:**
  - Hệ thống bỏ lọt những kiểu spam mới không có từ khoá. Bằng chứng là 6 case mọi model đều sai, và stress test chỉ đạt accuracy khoảng 0.58-0.70.
  - Đổi lại, hệ thống nhanh, rẻ và ổn định.
  - Cần cập nhật dữ liệu định kỳ để bù cho điểm yếu này.

## 7. AI log và điều đã học

| Phần | Bạn hỏi AI gì (prompt) | AI trả lời/viết gì | Bạn đã sửa / kiểm chứng gì |
| --- | --- | --- | --- |
| Phần 1-2 (ML/DL) | "Giải thích gradient descent trong `train_logreg` và vai trò của ReLU trong MLP." | Giải thích 4 bước cập nhật `w`, `b`. ReLU giúp mạng học quan hệ phi tuyến; bỏ ReLU thì 2 lớp Linear gộp thành 1 lớp tuyến tính. AI cũng viết code mẫu cho TODO 2.1. | Tự viết code TODO 1.1-1.4 (AI chỉ review, chỉ ra lỗi) và 5 dòng TODO 2.2. Chạy unit test và thí nghiệm 2.B: F1 khi bỏ ReLU vẫn là **0.938, bằng MLP** chứ không giảm. Như vậy phần tăng so với LogReg (0.896) đến từ Adam + mini-batch, không phải từ ReLU. |
| Phần 3 (Embedding) | "Công thức cosine similarity và vì sao phải chuẩn hoá vector trước khi tính `top_k_similar`." | `dot(a,b)/(‖a‖·‖b‖)`. Không chuẩn hoá thì tích vô hướng bị lệch theo độ dài vector chứ không theo hướng. AI viết code mẫu cho TODO 3.1-3.3. | Gõ lại code vào notebook, đọc hiểu từng dòng (xử lý vector 0, chuẩn hoá bằng `np.linalg.norm`). Chạy unit test 3.1-3.3 và kiểm tra thứ tự top-k đúng. |
| Phần 4 (LLM) | "Viết `call_openai` gọi Chat Completions trả về text + latency, và cách retry khi gặp lỗi 429." | Code mẫu dùng `client.chat.completions.create` + `time.perf_counter()`; retry exponential backoff chỉ cho lỗi tạm thời. | Tự review, thêm xử lý API key từ Colab Secrets/`.env`, truyền `**provider_kwargs(max_tokens)`. Chạy unit test 4.1 + 4.4 với mock API và pass. Chạy thật thì `invalid = 40/40` (output rỗng); phát hiện nhờ chi phí đo được chỉ bằng chi phí token input. |

**Điều hiểu ra hôm nay:**
- Mô hình đạt **F1 cao trên tập test chưa có nghĩa là hiểu spam**. Trên 50 case khó, mọi model đều tụt mạnh, và mô hình 128k tham số không thắng được khi dữ liệu không có kiểu spam mới.
- **Phần cải thiện thường đến từ dữ liệu và cách train** (Adam, feature pre-train), không phải từ việc thêm lớp ẩn hay phi tuyến. Bằng chứng: mạng bỏ ReLU vẫn đạt 0.938, và 3c không hơn 3b.
- **Phải đo, đừng tin trực giác:** lr = 50 hội tụ được, còn LLM "đắt và thông minh" lại cho F1 = 0 chỉ vì giới hạn 3 token. Các bộ đếm như `n_invalid` và việc đọc chi phí giúp phát hiện những lỗi im lặng này.
