# Fine-tuning Mô hình Vision-Language Y khoa cho Phân loại Có cấu trúc Bệnh lý Thoái hóa Cột sống Thắt lưng

*Một bản fine-tune dựa trên QLoRA của MedGemma 1.5 4B trên bộ dữ liệu RSNA 2024 Lumbar Spine Degenerative Classification.*

---

## Tóm tắt (Abstract)

Việc chẩn đoán các thay đổi thoái hóa của cột sống thắt lưng trên MRI yêu cầu bác sĩ X-quang gán 25 nhãn mức độ nghiêm trọng (severity) riêng biệt cho mỗi ca chụp — năm bệnh lý khác nhau được đánh giá tại năm tầng đốt sống. Việc gán nhãn thủ công cho từng ca là chậm và là một nút thắt cổ chai đã biết trong các quy trình phụ thuộc vào báo cáo chuẩn hóa. Chúng tôi nghiên cứu xem một mô hình vision-language y khoa có thể được fine-tune để sinh ra output có cấu trúc này trực tiếp từ ảnh MRI hay không. Cụ thể, chúng tôi fine-tune **MedGemma 1.5 4B** bằng QLoRA (base 4-bit + LoRA adapter bf16) trên bộ dữ liệu **RSNA 2024 Lumbar Spine Degenerative Classification**. Mô hình được huấn luyện để sinh ra một đối tượng JSON nén duy nhất chứa các mã severity (N=Normal/Mild, M=Moderate, S=Severe) cho mỗi trong 25 kết quả. Hai lựa chọn kỹ thuật giúp cấu hình trở nên trung thực: (i) một *multi-modality slice picker* chọn một slice sagittal T2 giữa, tối đa hai slice parasagittal T1, và tối đa năm slice axial T2 cho mỗi ca, sao cho tất cả 25 nhãn đều được bao phủ bởi hình ảnh thực sự được đưa vào mô hình; (ii) câu trả lời được nén từ dạng văn xuôi sang JSON ngắn (~100 token), giúp loss huấn luyện tập trung vào các token mang nhãn. Trên một validation set 198 ca được giữ riêng, Cohen's kappa tổng thể tăng từ **0.000** (mô hình base) lên **0.502** (đã fine-tune) với **tỷ lệ parse-failure JSON 0 %**. Bệnh lý quan trọng nhất về mặt lâm sàng — spinal canal stenosis — đạt kappa **0.54**; left và right subarticular stenosis đạt **0.44** và **0.50** tương ứng. Foraminal narrowing vẫn gần bằng 0 (kappa < 0.07) vì các ca Moderate/Severe của foraminal hiếm trong dữ liệu huấn luyện và lần chạy này không sử dụng class-rebalancing. Chúng tôi thảo luận giới hạn này và phác thảo oversampling cùng một biến thể classification-head làm các thí nghiệm tiếp theo hứa hẹn nhất.

---

## 1. Giới thiệu (Introduction)

### 1.1 Bối cảnh lâm sàng

Đau thắt lưng ảnh hưởng đến phần lớn người trưởng thành tại một thời điểm nào đó trong đời, và MRI cột sống thắt lưng là phương pháp chẩn đoán hình ảnh tiêu chuẩn khi nghi ngờ bệnh thoái hóa. Bác sĩ X-quang đánh giá từng tầng trong năm tầng đĩa đệm — L1/L2, L2/L3, L3/L4, L4/L5, và L5/S1 — cho một tập nhỏ các bệnh lý lặp lại: hẹp ống sống (spinal canal stenosis), hẹp lỗ liên hợp thần kinh (neural foraminal narrowing) ở mỗi bên, và hẹp ngách bên (subarticular stenosis) ở mỗi bên. Mỗi bệnh lý được phân loại theo thang severity thô, phổ biến nhất là *Normal/Mild*, *Moderate*, và *Severe*. Output của một ca chụp do đó là một bảng 25 ô gồm các nhãn phân loại. Việc điền bảng này có tính lặp lại nhưng quan trọng về mặt lâm sàng: lập kế hoạch phẫu thuật, quyết định điều trị bảo tồn, và báo cáo chuẩn hóa đều phụ thuộc vào nó.

### 1.2 Tại sao dùng vision-language model

Các pipeline computer-vision cổ điển cho bài toán này thường kết hợp CNN backbone với nhiều classification head. Những mô hình như vậy chính xác nhưng kém linh hoạt: chúng chỉ sinh ra nhãn và không gì khác. Một vision-language model về nguyên tắc có thể làm được nhiều hơn — sinh impression bằng văn bản tự do, tạo output có cấu trúc, hoặc chat về ca bệnh — sử dụng cùng một backbone. Đặc biệt, **MedGemma 1.5**, bản thích nghi y khoa của Google cho dòng Gemma 3, được phát hành như một mô hình đa phương thức SigLIP+Gemma3 4 B tham số đã được tiếp xúc với văn bản và hình ảnh y khoa trong quá trình pretraining. Điều này khiến nó trở thành một điểm khởi đầu tự nhiên cho fine-tuning y khoa.

Câu hỏi mà report này đặt ra cụ thể là: *liệu một bản fine-tune tiết kiệm tham số của MedGemma có thể sinh ra chẩn đoán có cấu trúc 25-label một cách đáng tin cậy trực tiếp từ ảnh MRI, trên một bộ dữ liệu công khai thực tế và mất cân bằng hay không?*

### 1.3 Đóng góp

Công việc này đóng góp những điều sau:

1. Một **pipeline fine-tuning QLoRA hoàn chỉnh** cho MedGemma 1.5 4B trên MRI thắt lưng, bao gồm tiền xử lý DICOM, chọn slice, xây dựng prompt, huấn luyện, và đánh giá.
2. Một **scheme multi-modality slice selection** giữ token budget trong phạm vi của một A100 40–80 GB đơn lẻ trong khi vẫn đảm bảo cả 25 nhãn đều có thể đánh giá được từ hình ảnh mà mô hình thực sự nhìn thấy.
3. Một **compact-JSON output schema** giảm số lượng token phía assistant khoảng 5× so với câu trả lời dạng văn xuôi, khiến gradient theo từng nhãn chiếm ưu thế trong supervised loss.
4. Một **protocol đánh giá nghiêm ngặt** tách biệt raw accuracy, weighted accuracy, và Cohen's kappa, đồng thời theo dõi JSON parse failure một cách rõ ràng để tránh việc inflation accuracy ngầm do fallback.
5. Một **so sánh thực nghiệm trung thực** giữa mô hình base và mô hình đã fine-tune trên split 198 ca được giữ riêng, cho thấy kappa tổng thể tăng từ 0.00 lên 0.50 và xác định class imbalance là failure mode chính còn lại.

Phần còn lại của report này được tổ chức như sau: §2 xây dựng kiến thức nền cần thiết cho phần còn lại của báo cáo; §3 mô tả bộ dữ liệu; §4 trình bày chi tiết phương pháp; §5 báo cáo experimental setup; §6 trình bày kết quả định lượng và định tính; §7 thảo luận về những gì hoạt động và những gì không; §8 liệt kê các giới hạn; §9 phác thảo công việc tương lai; §10 kết luận.

---

## 2. Kiến thức nền (Background knowledge)

Phần này giới thiệu ngắn gọn các khái niệm và kỹ thuật được sử dụng trong phần còn lại của báo cáo. Nó được giữ ngắn một cách có chủ đích — mục tiêu là làm cho công việc có thể tái tạo được và các quyết định thiết kế có thể hiểu được đối với một người đọc thoải mái với một lĩnh vực (chẩn đoán hình ảnh y khoa *hoặc* deep learning) nhưng không nhất thiết phải cả hai.

### 2.1 Giải phẫu cột sống thắt lưng và bệnh lý thoái hóa

Cột sống thắt lưng gồm năm đốt sống (L1–L5) và xương cùng (S1), ngăn cách bởi các đĩa đệm. Năm tầng đĩa đệm — viết là L1/L2 đến L5/S1 — là các đơn vị báo cáo tiêu chuẩn. Ba cấu trúc tại mỗi tầng có thể thoái hóa và chèn ép vào mô thần kinh:

- **Spinal canal (ống sống).** Ống trung tâm chứa đuôi ngựa. Hẹp (stenosis) có thể chèn ép các rễ thần kinh này; tầng L4/L5 là vị trí phổ biến nhất.
- **Neural foramen (lỗ liên hợp).** Các lỗ bên qua đó một rễ thần kinh đơn lẻ đi ra. Trái và phải được đánh giá riêng biệt. Foraminal narrowing được quan sát tốt nhất trên ảnh sagittal T1.
- **Subarticular recess (ngách bên).** Một túi bên nhỏ trong ống sống nơi một rễ thần kinh đi xuống trước khi thoát ra qua lỗ liên hợp kế tiếp. Stenosis ở đây được quan sát tốt nhất trên ảnh axial T2. Trái và phải được đánh giá riêng biệt.

Đối với mỗi trong năm tầng và mỗi trong năm cấu trúc này (canal, left foraminal, right foraminal, left subarticular, right subarticular), severity được phân loại thô: *Normal/Mild*, *Moderate*, *Severe*. Điều này cho 5 × 5 = 25 nhãn cho mỗi ca chụp.

### 2.2 Chụp MRI cột sống thắt lưng

Các ca MRI thắt lưng là multi-sequence và multi-plane. Ba loại series được sử dụng trong công việc này là:

- **Sagittal T2.** Một sequence nhìn nghiêng (side-view) trong đó dịch não tủy hiện sáng. Tốt nhất để đánh giá ống trung tâm vì tủy và rễ tối nằm bên trong một cột dịch não tủy sáng.
- **Sagittal T1.** Cùng góc nhìn nghiêng, nhưng mỡ hiện sáng và dịch não tủy tối. Tốt nhất để đánh giá lỗ liên hợp thần kinh, vì mỡ ngoài màng cứng sáng bao quanh rễ thần kinh đi ra tối.
- **Axial T2.** Một góc nhìn cắt ngang tại một tầng đĩa đệm cụ thể. Tốt nhất để đánh giá subarticular recess và sự bất đối xứng trái/phải.

Một ca chụp đơn lẻ thường chứa cả ba series. Mỗi series gồm nhiều slice song song xuyên qua giải phẫu. Giá trị pixel trong file DICOM thô không được chuẩn hóa giữa các máy quét; chuyển đổi sang ảnh có thể xem được yêu cầu sử dụng metadata *window centre / window width* của DICOM hoặc áp dụng chuẩn hóa percentile robust.

### 2.3 Vision-language models và MedGemma 1.5

Một vision-language model (VLM) hiện đại ghép một vision encoder với một language model:

1. **Vision encoder** chuyển đổi ảnh thành một chuỗi patch embedding. MedGemma sử dụng **SigLIP** (Sigmoid Loss for Image-Text Pretraining), một biến thể của CLIP xử lý input với patch size 14 pixel.
2. Một **projection layer** ánh xạ các patch embedding này vào không gian embedding của language model.
3. **Language model** — Gemma 3, một transformer decoder-only — nhận các vision token được chiếu xen kẽ với các text token và sinh phản hồi một cách autoregressive.

Tại độ phân giải ảnh $S \times S$, encoder sinh ra $(S/14)^2$ vision token cho mỗi ảnh. Tại $S=448$ là $1024$ token; tại $S=896$ là $4096$. Với $N$ ảnh mỗi example, **chi phí vision token** do đó là $N \cdot (S/14)^2$, chiếm phần lớn ngân sách sequence length.

**MedGemma 1.5 4B-IT** là phiên bản instruction-tuned y khoa của Google. Nó đã quen thuộc với từ vựng X-quang và cấu trúc báo cáo chuẩn, nhưng nó không được pre-train trên schema gán nhãn RSNA hoặc trên output JSON 25-label mà bài toán này yêu cầu. Do đó cần fine-tuning.

### 2.4 Parameter-efficient fine-tuning (LoRA và QLoRA)

Fine-tuning đầy đủ một mô hình 4 B tham số là khả thi nhưng tốn kém: nó yêu cầu lưu trữ optimizer state (Adam giữ $2 \times$ số tham số trong các momentum buffer) và toàn bộ trọng số bf16, dễ dàng vượt quá 60 GB VRAM. **Low-Rank Adaptation (LoRA)** giải quyết điều này bằng cách đóng băng trọng số base $W \in \mathbb{R}^{d \times d}$ và học một cập nhật low-rank

$$
\Delta W = B A, \qquad A \in \mathbb{R}^{r \times d}, \; B \in \mathbb{R}^{d \times r}, \; r \ll d.
$$

Chỉ $A$ và $B$ được huấn luyện. Trọng số hiệu dụng tại inference là $W + \alpha \Delta W$ với một scalar $\alpha$ nào đó. **QLoRA** kết hợp điều này với quantization 4-bit của base đã đóng băng: mô hình base được load ở 4 bit và các ma trận LoRA ở bf16 (hoặc fp16). Forward và backward pass qua base sử dụng giá trị đã de-quantize on the fly; chỉ các ma trận LoRA tích lũy gradient. Điều này giảm tổng chi phí VRAM xuống xấp xỉ kích thước của base 4-bit cộng với các buffer LoRA nhỏ — khả thi trên một A100 đơn lẻ.

### 2.5 Các metric đánh giá

Công việc này sử dụng nhiều metric; báo cáo chúng một cách riêng lẻ có thể gây hiểu lầm.

- **Raw accuracy.** Tỷ lệ nhãn được dự đoán đúng. Trên một bộ dữ liệu mất cân bằng nặng (≈ 85 % Normal/Mild) một classifier đơn giản luôn dự đoán lớp đa số đã đạt ≈ 85 % — accuracy đơn lẻ do đó không cung cấp thông tin.
- **F1 (weighted).** F1 theo từng lớp được trung bình với trọng số tỷ lệ với class support. Ít gây hiểu lầm hơn accuracy nhưng vẫn bị phồng lên một phần bởi lớp chiếm ưu thế.
- **Cohen's kappa, $\kappa$.** Accuracy *được hiệu chỉnh cho chance agreement*:
  $$
  \kappa = \frac{p_o - p_e}{1 - p_e},
  $$
  với $p_o$ là agreement quan sát được và $p_e$ là agreement kỳ vọng do ngẫu nhiên theo phân phối lớp biên. $\kappa = 0$ nghĩa là không tốt hơn ngẫu nhiên; $\kappa = 1$ nghĩa là đồng thuận hoàn hảo. *Đây là headline metric trong report này*: nó robust với class imbalance và là thước đo chuẩn cho các nghiên cứu đồng thuận trong X-quang.
- **Weighted accuracy.** Accuracy theo từng lớp được nhân trọng số $[1, 2, 4]$ cho $[N, M, S]$. Phạt lỗi trên các lớp hiếm hơn và nghiêm trọng hơn về mặt lâm sàng nặng hơn. Về mặt khái niệm tương tự như weighted log-loss chính thức của RSNA challenge, nhưng được định nghĩa trên hard prediction.
- **RSNA weighted score proxy.** Một scalar thay thế cho weighted log-loss chính thức của competition, được tính bằng cách dùng lại cùng class weight $[1, 2, 4]$. Vì một generative model không phát ra softmax probability được calibrate, đây là một proxy *hard-prediction*, không phải log-loss thực sự; nó là một weighted error rate, được scale bởi xấp xỉ 16.1. Do đó nó *không* trực tiếp so sánh được với RSNA public leaderboard.
- **Parse failure rate.** Tỷ lệ response của mô hình mà không thể extract được object JSON. Được track riêng vì parser fallback về "Normal/Mild" khi thất bại, điều này nếu không sẽ ngầm tăng accuracy trên một bộ dữ liệu mất cân bằng.

---

## 3. Bộ dữ liệu (Dataset)

### 3.1 Nguồn

Dữ liệu đến từ competition công khai **RSNA 2024 Lumbar Spine Degenerative Classification**, phân phối qua Kaggle. Phần training gồm xấp xỉ 2,000 ca MRI thắt lưng; mỗi ca chứa nhiều series (thường là sagittal T1, sagittal T2, axial T2) lưu trữ dưới dạng file DICOM thô cùng với các file CSV có cấu trúc:

- `train.csv` — một dòng mỗi study × condition × level, đưa ra nhãn severity được bác sĩ X-quang chấm điểm (target);
- `train_series_descriptions.csv` — metadata loại series (sagittal vs axial, T1 vs T2);
- `train_label_coordinates.csv` — các annotation (x, y, instance number) theo từng condition xác định slice và pixel minh họa tốt nhất cho mỗi nhãn.

Test set của competition được Kaggle giữ riêng và không sử dụng ở đây; chúng tôi giữ riêng 10 % của training set làm split validation của riêng mình.

### 3.2 Pipeline tiền xử lý

Cho mỗi ca, pipeline chuyển đổi các DICOM liên quan thành PNG 448 × 448. Metadata window centre / window width được sử dụng khi có sẵn; nếu không, áp dụng chuẩn hóa robust theo percentile (0.5 / 99.5 percentile). Photometric interpretation `MONOCHROME1` được phát hiện và đảo ngược để khớp với quy ước chuẩn "sáng = cường độ cao".

Script ghi ra:

- `train_dataset.jsonl` — một JSON example cho mỗi ca để huấn luyện,
- `val_dataset.jsonl` — cùng schema, giữ riêng 10 %,
- một thư mục flat chứa các PNG ở độ phân giải đã chọn.

Mỗi JSON example mang theo study id, danh sách các path PNG đã chọn, danh sách các loại series tương ứng, severity worst-case trên tất cả các nhãn (được sử dụng sau này cho oversampling theo lớp), và một conversation hai lượt (user instruction → compact-JSON assistant answer).

### 3.3 Khối lượng và split

| Đại lượng | Số lượng |
|---|---|
| Tổng số study RSNA training được sử dụng | 1,974 |
| Study bị bỏ (thiếu coordinate / vượt token budget) | ≈ 26 |
| Train split (90 %) | **1,776** |
| Validation split (10 %) | **198** |
| Tổng số label prediction cho mỗi lần evaluation | 198 × 25 = **4,950** |

Validation split 10 % được cố định ở mức study, nên không bệnh nhân nào xuất hiện ở cả hai partition.

### 3.4 Phân phối lớp

Bộ dữ liệu lệch nặng về phía outcome *Normal/Mild* — một phân phối thực tế về mặt lâm sàng, vì hầu hết bệnh nhân được chụp ảnh vì đau thắt lưng chỉ có thoái hóa nhẹ ở hầu hết các tầng. Tổng hợp trên cả 25 nhãn:

| Severity | Tỷ lệ xấp xỉ |
|---|---|
| Normal/Mild (N) | ≈ 85 % |
| Moderate (M) | ≈ 12 % |
| Severe (S) | ≈ 3 % |

Sự mất cân bằng này chịu trách nhiệm cho phần lớn chênh lệch theo condition trong §6 — các nhãn mà các ca Moderate/Severe đặc biệt hiếm (foraminal narrowing nhất là) cuối cùng bị under-trained.

### 3.5 Label schema

Schema được sử dụng nội bộ và trong output của mô hình được đưa ra bởi ba map được chia sẻ trên tất cả các script:

```python
COND_KEYS = {
    "spinal_canal_stenosis":            "canal",
    "left_neural_foraminal_narrowing":  "lf",
    "right_neural_foraminal_narrowing": "rf",
    "left_subarticular_stenosis":       "ls",
    "right_subarticular_stenosis":      "rs",
}
LEVEL_KEYS     = {"l1_l2": "L1L2", ..., "l5_s1": "L5S1"}
SEVERITY_CODES = {"Normal/Mild": "N", "Moderate": "M", "Severe": "S"}
```

Assistant target được sinh ra cho mỗi example do đó là một object JSON nén duy nhất chẳng hạn như:

```json
{
  "L1L2": {"canal": "N", "lf": "N", "rf": "N", "ls": "N", "rs": "N"},
  "L2L3": {"canal": "M", "lf": "N", "rf": "N", "ls": "N", "rs": "S"},
  "L3L4": {"canal": "N", "lf": "N", "rf": "N", "ls": "N", "rs": "N"},
  "L4L5": {"canal": "M", "lf": "M", "rf": "N", "ls": "N", "rs": "N"},
  "L5S1": {"canal": "N", "lf": "S", "rf": "M", "ls": "N", "rs": "N"}
}
```

---

## 4. Phương pháp (Method)

### 4.1 Base model

Điểm khởi đầu là `unsloth/medgemma-1.5-4b-it`, bản mirror đã được pre-quantize của Unsloth cho MedGemma 1.5 4B-IT của Google. Về mặt kiến trúc đây là một SigLIP vision encoder feed một Gemma 3 4 B decoder thông qua một multimodal projection layer. Vision tower sử dụng patch size 14, nên độ phân giải input ánh xạ tới số vision token là $(S/14)^2$ mỗi ảnh.

Mô hình base được load ở 4-bit sử dụng `bitsandbytes`, với các ma trận LoRA ở bf16 — setup QLoRA chuẩn. Mô hình expose 2,734,255,984 tham số tổng cộng; LoRA thêm 38,497,792 tham số có thể huấn luyện, tức **0.89 %** của mô hình.

### 4.2 Cấu hình LoRA

LoRA adapter được áp dụng cho cả phía language và vision của mô hình. Target module:

| Block | Module được target |
|---|---|
| Attention | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| MLP | `gate_proj`, `up_proj`, `down_proj` |
| Vision tower | cùng các pattern target bên trong SigLIP |
| Language layer | cùng các pattern target bên trong Gemma 3 |

Hyperparameter:

| LoRA parameter | Giá trị |
|---|---|
| Rank, $r$ | 16 |
| Scaling, $\alpha$ | 32 |
| Dropout | 0.05 |
| Bias | none |
| Dùng rsLoRA | no |

Adapt vision tower là thiết yếu: nếu không mô hình không thể học để định vị các cấu trúc nhỏ (foramen, subarticular recess) mà các nhãn mô tả.

### 4.3 Multi-modality slice selection

Một ca MRI đơn lẻ quá lớn để feed nguyên vẹn vào mô hình. Chúng ta phải chọn một tập con nhỏ, liên quan đến nhãn, của các slice.

Selector triển khai ba nguyên tắc:

1. **Một slice cho mỗi mục đích imaging.** Slice sagittal T2 *giữa* là slice hữu ích nhất để đánh giá ống trung tâm tại cả năm tầng. Hai slice T1 *parasagittal* hai bên sườn đường giữa cho thấy lỗ liên hợp thần kinh tốt nhất. Các slice *axial* T2, một cho mỗi tầng đĩa đệm khi có sẵn, cho thấy subarticular stenosis tốt nhất.
2. **Cả ba modality, luôn luôn.** Khi `train_label_coordinates.csv` có sẵn, picker truy vấn nó cho các slice index overlap với coordinate đã gán nhãn tại mỗi tầng, đảm bảo các nhãn foraminal và subarticular có thể đánh giá được một cách trung thực. Pipeline từ chối fabricate level-to-slice mapping khi coordinate bị thiếu, trừ khi cờ `--allow-no-coords` được set cho smoke test.
3. **Series priority và three-pass selection.** Một loop ngây thơ theo từng modality bỏ qua các modality cho các study có nhiều series cùng loại. Chúng tôi thay vào đó sử dụng một selector *three-pass*:

```
SERIES_PRIORITY = ["sagittal_t2", "sagittal_t1", "axial_t2"]

Pass 1: chọn một series mỗi loại theo thứ tự priority
Pass 2: nếu còn slot, fill với thêm series của bất kỳ modality nào đã biết
Pass 3: nếu vẫn còn slot, cho phép series unknown cuối cùng
```

Với giới hạn ba series mỗi study và ≤ 8 slice tổng cộng, điều này cho một example điển hình chứa một slice sagittal T2 giữa, hai slice parasagittal T1, và tối đa năm slice axial T2.

### 4.4 Compact JSON output schema

Một assistant target ngây thơ là một đoạn văn xuôi X-quang ngôn ngữ tự nhiên. Target như vậy có hai nhược điểm: nó dài (≈ 500 token) và tỷ lệ token *mang nhãn* nhỏ. Hầu hết loss được dành để học sinh ra filler.

Chúng tôi nén câu trả lời thành một object JSON duy nhất sử dụng key ngắn và mã severity một ký tự (N / M / S). Kết quả là khoảng **100 token** nội dung assistant, trong đó xấp xỉ 25 token mang thông tin nhãn thực tế. Vì supervised loss chỉ chạy trên các token assistant (xem §4.5), nén 5× này trực tiếp tăng tỷ lệ tín hiệu gradient mang nhãn.

Lợi ích thứ hai là độ tin cậy parse. Câu trả lời văn xuôi phải được match với regular expression mỏng manh; JSON nén được parse với một cuộc gọi `json.loads` duy nhất. Trên toàn bộ validation set 198 study, **parse failure rate là 0 %** (xem §6).

### 4.5 Định dạng prompt và supervised-loss masking

Cùng một prompt phía user được sử dụng trong huấn luyện, đánh giá, và inference live. Nó chứa:

1. Một mô tả về bối cảnh imaging: `"You are provided with N lumbar spine MRI image(s) (Sagittal T2, Sagittal T1, Axial T2). Classify all degenerative conditions at each spinal level from L1/L2 to L5/S1."`;
2. Một mô tả schema đánh vần đầy đủ cấu trúc JSON và ý nghĩa của mọi key và mã severity;
3. Các image token thực tế, một placeholder `<image>` cho mỗi slice đã load.

Assistant turn là JSON nén được hiển thị trong §3.5. Trong quá trình huấn luyện, `UnslothVisionDataCollator` mask mọi token user-turn thành nhãn `-100`, nên cross-entropy loss chỉ được tính trên các token assistant. Mô hình do đó được huấn luyện rõ ràng để *sinh ra* câu trả lời có cấu trúc thay vì tái tạo prompt.

Một hàm helper duy nhất `build_user_prompt(n_images, series_types)` xây dựng user turn này cho cả huấn luyện và đánh giá, đảm bảo rằng prompt tại thời điểm inference khớp verbatim với prompt SFT. Bất kỳ sự lệch nào ở đây sẽ ngầm làm giảm chất lượng đánh giá.

### 4.6 Lập luận token budget

Tại image size $448$ và patch size $14$, một ảnh tiêu thụ $1024$ vision token. Với multi-modality picker cung cấp tối đa 8 ảnh cho mỗi example:

$$
\text{image tokens} = 8 \times (448/14)^2 = 8 \times 1024 = 8192.
$$

Thêm ≈ 400 token cho user prompt, ≈ 100 token cho assistant JSON, và một margin nhỏ cho các token chat-template, sequence length **9,216** là đủ cho demo profile và tránh truncate nhãn assistant. Lựa chọn độ phân giải (448 thay vì native 896 của mô hình) được quyết định bởi budget này: cùng setup multi-modality 8 ảnh tại 896² sẽ yêu cầu ≈ 33k vision token, vượt quá những gì vừa với 40 GB ngay cả tại độ chính xác QLoRA.

### 4.7 Vòng lặp huấn luyện

Huấn luyện sử dụng `SFTTrainer` của TRL (v0.18+) trên `FastVisionModel` của Unsloth. Data collator là `UnslothVisionDataCollator`, xử lý chung ảnh và text và áp dụng user-turn masking được mô tả trong §4.5. Lazy image loading được kích hoạt qua `Dataset.set_transform`, nên các PIL image được materialize theo từng batch tại thời điểm `__getitem__` thay vì serialize vào PyArrow cache.

Supervised objective là cross-entropy chuẩn trên các token assistant. Optimizer là AdamW với lịch trình learning rate cosine, warmup, và weight decay. Một `EarlyStoppingCallback` với patience 3 theo dõi `eval_loss`, nhưng không trigger trong bất kỳ lần chạy nào của chúng tôi.

---

## 5. Experimental setup

### 5.1 Phần cứng

| Component | Giá trị |
|---|---|
| GPU | NVIDIA A100 80 GB PCIe |
| CUDA toolkit | 12.6 |
| Compute capability | 8.0 |
| Host | RunPod cloud GPU instance |
| OS | Linux (bản phái sinh Ubuntu) |

### 5.2 Software stack

| Library | Phiên bản |
|---|---|
| Python | 3.11 |
| PyTorch | 2.12.0+cu126 |
| Unsloth | 2026.5.7 |
| Transformers | 5.5.0 |
| TRL | ≥ 0.18.2, ≤ 0.24.0 |
| PEFT | ≥ 0.18.0 |
| BitsAndBytes | ≥ 0.43.1 |
| Triton | 3.7.0 |

Tập pin đầy đủ được encode trong `setup.sh` để đảm bảo khả năng tái tạo trên các pod.

### 5.3 Hyperparameter (profile `demo_a100_40g`)

| Parameter | Giá trị |
|---|---|
| Độ phân giải ảnh | 448 × 448 |
| Max sequence length | 9,216 token |
| Slice mỗi series (axial) | tối đa 5 (một cho mỗi level) |
| Series mỗi study | tối đa 3 (sag T2 + sag T1 + axial T2) |
| LoRA rank, $r$ / $\alpha$ | 16 / 32 |
| LoRA dropout | 0.05 |
| Per-device batch size | 1 |
| Gradient accumulation | 8 |
| Effective batch size | 8 |
| Epoch | 3 (không trigger early-stop) |
| Optimizer | AdamW |
| Learning rate | 2 × 10⁻⁴ |
| LR schedule | cosine, warmup ratio 0.05 |
| Weight decay | 0.01 |
| Precision (LoRA) | bf16 |
| Precision (base) | 4-bit (NF4) |
| Eval interval | mỗi 100 training step |
| Save interval | mỗi 200 training step |
| Random seed | 42 |

### 5.4 Thời lượng huấn luyện

Lần chạy 3-epoch trên 1,776 example huấn luyện đã sinh ra **666 optimizer step tổng cộng** ở effective batch size 8. Tổng wall-clock time: **14.28 giờ** (≈ 51,400 s). Throughput trung bình: 0.104 train sample / s; 0.013 train step / s.

---

## 6. Kết quả (Results)

### 6.1 Metric chính

Tất cả metric trong phần này được tính bởi `evaluate.py` trên validation set 198 study được giữ riêng (4,950 label prediction cho mỗi condition aggregate trên các level). Mô hình base là `unsloth/medgemma-1.5-4b-it` không có adapter và cùng định dạng prompt; mô hình fine-tune là LoRA checkpoint được save vào cuối epoch 3 trong §5.4.

**Bảng 6.1. Metric tổng hợp trên validation set 198 study.**

| Metric | Base | Fine-tuned | Δ |
|---|---:|---:|---:|
| Cohen's $\kappa$ (overall) | 0.000 | **0.502** | **+0.502** |
| Weighted accuracy ([1,2,4]) | 0.577 | **0.673** | +0.097 |
| F1 (weighted) | 0.683 | **0.777** | +0.095 |
| Raw accuracy | 0.779 | 0.814 | +0.035 |
| RSNA weighted score proxy (↓) | 6.826 | **5.268** | −1.558 |
| Parse failure rate | 0.0 % | **0.0 %** | 0 pp |

Kết quả then chốt là bước nhảy kappa từ **0.00 đến 0.50**. Raw accuracy chỉ di chuyển nhẹ (78 % → 81 %), điều này nhất quán với việc mô hình base khai thác class prior — dự đoán Normal/Mild cho mọi nhãn. Sau khi được kappa-correct, mô hình base về cơ bản đóng góp không tín hiệu chẩn đoán nào, trong khi mô hình đã fine-tune đạt được moderate agreement trên các nhãn của bác sĩ X-quang.

### 6.2 Cohen's kappa theo từng condition

**Bảng 6.2. Kappa và accuracy theo từng condition (fine-tuned, validation).**

| Condition | Avg. accuracy | Avg. $\kappa$ (Base) | Avg. $\kappa$ (Fine-tuned) |
|---|---:|---:|---:|
| Spinal canal stenosis | 0.907 | 0.000 | **0.537** |
| Right subarticular stenosis | 0.799 | 0.000 | **0.501** |
| Left subarticular stenosis | 0.772 | 0.000 | **0.435** |
| Right neural foraminal narrowing | 0.804 | 0.000 | 0.066 |
| Left neural foraminal narrowing | 0.786 | 0.000 | 0.033 |

Ba condition — spinal canal stenosis và cả hai subarticular stenosis — cải thiện đáng kể. Hai nhãn foraminal-narrowing vẫn gần với chance. Chúng tôi thảo luận lý do trong §7.

### 6.3 Training dynamics

Đường cong loss huấn luyện và validation được extract từ TRL log. Cross-entropy trên assistant JSON tự nhiên nhỏ (phần schema của câu trả lời có thể dự đoán cao), nên giá trị loss tuyệt đối nằm trong khoảng 0.02–0.05; điều quan trọng là xu hướng đi xuống.

**Bảng 6.3. Giá trị training loss được chọn (cosine schedule, 3 epoch, 666 step).**

| Step | Training loss |
|---:|---:|
| 10 | 1.372 |
| 20 | 0.494 |
| 30 | 0.084 |
| 40 | 0.042 |
| 100 | 0.0344 |
| 200 | 0.0327 |
| 300 | 0.0312 |
| 400 | 0.0294 |
| 500 | 0.0283 |
| 600 | 0.0280 |
| 666 (cuối) | 0.0277 |

**Bảng 6.4. Tiến triển evaluation-loss tại mỗi 100-step eval checkpoint.**

| Step | Eval loss |
|---:|---:|
| 100 | 0.0344 |
| 200 | 0.0327 |
| 300 | 0.0317 |
| 400 | 0.0294 |
| 500 | 0.0283 |
| 600 | 0.0280 |
| 666 (best) | 0.0278 |

Training loss sụp đổ nhanh trong khoảng ~30 step đầu khi mô hình học định dạng JSON, sau đó đi vào một regime dài chậm trong đó nó học các nhãn thực tế. Validation loss giảm đơn điệu qua mọi evaluation checkpoint, và `EarlyStoppingCallback(patience=3)` không bao giờ trigger. Không có dấu hiệu overfitting trong 3 epoch đã chạy.

### 6.4 Confusion matrix

Một lưới 5 × 5 confusion matrix (một cho mỗi condition × mỗi level) được sinh bởi `evaluate.plot_confusion_matrices`. File figure là `evaluation_results/finetuned_confusion_matrices.png`.

Định tính:
- Đối với các hàng **canal** và **subarticular**, đường chéo được populate rõ ràng cho *Moderate* và *Severe* cùng với sự thống trị kỳ vọng của *Normal/Mild*.
- Đối với các hàng **left foraminal** và **right foraminal**, prediction gần như tập trung hoàn toàn vào cột *Normal/Mild* ở mọi level, gần như không có entry đường chéo trong các hàng Moderate và Severe.

Pattern trực quan này khớp với các giá trị kappa theo condition trong Bảng 6.2.

### 6.5 Ví dụ định tính

Phần sau so sánh ground truth, mô hình base, và mô hình fine-tune trên ba validation study đại diện.

**Example A — Tất cả Normal/Mild (Study 1324569502)**

```
Ground truth :  N N N N N | N N N N N | N N N N N | N N N N N | N N N N N
Base         :  N N N N N | N N N N N | N N N N N | N N N N N | N N N N N   ✓
Fine-tuned   :  N N N N N | N N N N N | N N N N N | N N N N N | N N N N N   ✓
```

Cả hai mô hình match hoàn hảo. Đây là case phổ biến nhất trong bộ dữ liệu, đó là lý do tại sao raw accuracy trên cả hai mô hình khoảng 80 %.

**Example B — Moderate/Severe hỗn hợp (Study 4287160193)**

```
Ground truth (canal) :  N M N N N
Ground truth (lf)    :  N N N M N
Ground truth (ls)    :  N S M M N
Ground truth (rs)    :  N N M M N

Base (mọi nhãn)        :  N (uniform, mọi level)
Fine-tuned (mọi nhãn)  :  N (uniform, mọi level)
```

Case này có nhiều nhãn Moderate và một nhãn Severe. Cả hai mô hình default về Normal/Mild ở mọi nơi. Cải thiện của mô hình fine-tune là *trung bình* trên bộ dữ liệu — nó không cứu được mọi case khó, và class imbalance đặc biệt làm tổn thương performance trên các phát hiện severe đơn lẻ.

**Example C — Subarticular Moderate hai bên (Study 4189246764)**

```
Ground truth (ls) :  N N M M N
Ground truth (rs) :  M M M M N

Base       (ls / rs):  N N N N N / N N N N N
Fine-tuned (ls / rs):  N N N N N / N N N N N
```

Ở đây mô hình fine-tune lại dự đoán toàn-Normal. Pattern của các miss không phải ngẫu nhiên: mô hình tự tin nhất, và chính xác nhất, trên canal stenosis tại L4/L5; các call subarticular hai bên đơn lẻ vẫn khó khăn nếu không có class rebalancing rõ ràng.

Các ví dụ này được lấy từ `comparison/text_examples.txt`. Kappa tổng thể 0.50 headline đến từ các case study trong đó canal hoặc subarticular Moderate/Severe thực sự được phát hiện — figure confusion matrix (§6.4) cho thấy các diagonal hit này, nhưng chúng được trải ra trên hàng trăm validation prediction thay vì tập trung trong bất kỳ ví dụ đơn lẻ nào.

### 6.6 Kết quả compare.py small-sample

`compare.py` chạy cả hai mô hình trên một subset nhỏ (mặc định 20 study) và báo cáo kappa theo từng condition. Với 20 sample mỗi level, một số lớp Moderate/Severe có zero true positive tại các level nhất định, điều này khiến Cohen's kappa undefined (denominator 1 − p_e = 0, sinh ra NaN). Ví dụ, comparison small-sample báo cáo:

| Condition (20-sample) | Base $\kappa$ | Fine-tuned $\kappa$ |
|---|---:|---:|
| Spinal canal stenosis | 0.00 | 0.37 |
| Left foraminal narrowing | NaN | NaN |
| Right foraminal narrowing | NaN | NaN |
| Left subarticular stenosis | 0.00 | 0.45 |
| Right subarticular stenosis | 0.00 | 0.38 |

Các NaN là sample-size artefact, không phải model failure: với chỉ 20 study × 5 level mỗi condition, một số ô (level × condition) có tất cả nhãn truth hoặc prediction là Normal, khiến chance-corrected agreement undefined. Các con số có thẩm quyền là kết quả 198-sample `evaluate.py` trong Bảng 6.1 và 6.2.

---

## 7. Thảo luận (Discussion)

### 7.1 Fine-tune thực sự thay đổi điều gì

Mô hình base hành xử như một classifier *strong prior*: khi được yêu cầu điền JSON, nó sinh ra toàn *Normal/Mild* trên về cơ bản mọi study. Điều này hợp lý cục bộ — Normal/Mild là lớp đa số cho 25 trong 25 nhãn — nhưng nó cho Cohen's kappa chính xác bằng zero, vì dự đoán mode của phân phối không mang thông tin nào ngoài tần suất biên của mode đó.

Mô hình fine-tune khác biệt *về chất*: nó học để dự đoán *Moderate* và đôi khi *Severe* trên các condition và level mà những điều này phổ biến (chủ yếu là canal stenosis tại L4/L5 và subarticular hai bên tại L3–L5). Mỗi dự đoán non-trivial như vậy là một thắng lợi nhỏ cho kappa vì $p_o$ giờ đây vượt quá $p_e$. Cộng dồn trên validation set, mô hình đạt $\kappa = 0.50$ — moderate agreement theo interpretation Landis–Koch (1977) chuẩn.

Điều thiết yếu là đọc con số này cùng với §6.5: ngay cả tại $\kappa = 0.50$ mô hình vẫn miss nhiều instance Moderate/Severe cụ thể. Cải thiện là thực và có ý nghĩa thống kê, nhưng nó là một tín hiệu được trung bình trên hàng nghìn label prediction, không phải một lời hứa per-case.

### 7.2 Tại sao foraminal narrowing không cải thiện

Kappa theo condition cho left và right neural foraminal narrowing là 0.03 và 0.07 tương ứng — khó phân biệt với mô hình base. Ba lực kết hợp tạo ra kết quả này.

Thứ nhất, **tỷ lệ của Moderate/Severe ở các nhãn foraminal là thấp nhất trong bộ dữ liệu.** Foraminal narrowing ở người lớn tuổi tập trung tại L5/S1 và L4/L5, và ngay cả ở đó một bên thường bị ảnh hưởng nhiều hơn cả hai. Một nhãn *Moderate* foraminal non-trivial chỉ xuất hiện trong một phần nhỏ của các study huấn luyện.

Thứ hai, **tín hiệu lớp hiếm phải cạnh tranh với strong Normal/Mild prior trên cùng nhãn.** Không có bất kỳ reweighting nào, cross-entropy loss có nhiều gradient Normal/Mild hơn rất nhiều so với gradient Moderate/Severe trên các vị trí foraminal, nên mô hình được huấn luyện ngầm để dự đoán Normal/Mild ở đó.

Thứ ba, **foraminal narrowing tinh tế về mặt trực quan ngay cả trên modality sagittal T1 chuyên dụng.** Phân biệt Normal với Moderate yêu cầu nhận thấy sự thay đổi ở mỡ ngoài màng cứng giữa một foramen bình thường và một foramen bị chèn ép nhẹ. Đây là một cue trực quan tinh tế và có thể đòi hỏi độ phân giải input cao hơn 448² được sử dụng ở đây.

### 7.3 Thiết kế compact-JSON đã trả công

Tỷ lệ parse-failure 0 % quan sát thấy trong §6.1 là không bình thường. Ngay cả response của mô hình base, không có fine-tuning, sinh ra JSON parse được valid 100 % thời gian trên validation set này. Đây là hệ quả trực tiếp của hai lựa chọn thiết kế:

1. Mô tả schema xuất hiện verbatim trong *mỗi* user turn mà mô hình được prompt với. Đây là cùng một chuỗi tại thời điểm huấn luyện, đánh giá, và inference. Mô hình do đó rất có khả năng tái tạo một object JSON tuân theo schema đó.
2. Schema ngắn và cứng về cấu trúc. Có năm key level cố định, năm key condition cố định, và ba mã severity một ký tự. Ngay cả một mô hình base cũng có ít bậc tự do để lệch khỏi shape này.

Một hệ quả là không metric nào của chúng tôi dựa vào silent fallback — mọi dự đoán chúng tôi đếm thực sự được phát ra dưới dạng JSON parse được, không được sinh bởi một handler default-to-Normal/Mild downstream của một parse failure.

### 7.4 Caveat về RSNA weighted score

`rsna_weighted_score_proxy` được báo cáo ở trên *không phải* là RSNA log-loss chính thức. Metric competition yêu cầu calibrated softmax probability, mà một generative language model không phát ra một cách tự nhiên. Proxy được tính bằng cách interpret mỗi hard prediction là một one-hot probability và áp dụng cùng class weight $[1, 2, 4]$. Do đó nó là một *weighted error rate được scale bởi* $\approx 16.1$, không phải log-loss. Cải thiện từ 6.83 xuống 5.27 trong Bảng 6.1 chỉ ra một giảm tuyệt đối trong weighted error, nhưng nó không thể so sánh với public leaderboard.

### 7.5 Modality coverage quan trọng

Profile `demo_a100_40g` được thiết kế có chủ đích để bao gồm cả ba loại series cho mỗi study (sagittal T2, sagittal T1, và axial T2). Một profile đơn giản hơn chỉ sử dụng một series — chẳng hạn sagittal T2 — vẫn cho phép đánh giá trung thực canal stenosis, nhưng các nhãn foraminal và subarticular sẽ nhất thiết phải đoán từ class prior, vì mô hình sẽ không nhìn vào imaging liên quan tại thời điểm inference. Kết quả theo condition của chúng tôi trong Bảng 6.2 xác nhận thiết kế này: các condition mà modality chuyên dụng của chúng thực sự được bao gồm (canal trên sag T2, subarticular trên axial T2) di chuyển đáng kể; các condition mà modality chuyên dụng của chúng (parasag T1) được bao gồm nhưng ở độ phân giải thấp vẫn gặp khó khăn.

---

## 8. Giới hạn (Limitations)

Công việc này là một dự án nghiên cứu / học thuật và một số giới hạn nên được ghi nhớ khi diễn giải các kết quả.

- **Class imbalance.** Mô hình under-predict các lớp Moderate và Severe, đặc biệt cho foraminal narrowing. Không có class rebalancing (oversampling, class-weighted loss, hoặc focal loss) nào được áp dụng trong lần chạy này.
- **Generative classification có metric mong manh.** Vì mô hình sinh ra dự đoán phân loại thay vì xác suất được calibrate, RSNA-style weighted log-loss không thể được tính ở dạng competition của nó. So sánh với leaderboard đó do đó không có ý nghĩa.
- **Một bộ dữ liệu, một split.** Tất cả đánh giá đều trên một split 10 % được giữ riêng của RSNA 2024. Không có validation ngoài trên imagery từ một viện, vendor, hoặc population khác đã được thực hiện. Khả năng generalize off-distribution là không biết.
- **Phụ thuộc vào slice selection.** Chất lượng inference phụ thuộc vào việc user cung cấp các slice được chọn với cùng scheme multi-modality được sử dụng trong huấn luyện (sagittal T2 giữa + parasagittal T1 + axial T2 cho mỗi level). Slice ngẫu nhiên hoặc input single-modality sẽ làm giảm kết quả.
- **Không có validation lâm sàng.** Mô hình chưa được bác sĩ X-quang review như một phần của dự án này. Nó không phải là một thiết bị clinical-decision-support và không nên được sử dụng như vậy.
- **Compute budget.** Ba epoch trên một A100 80GB. Các profile lớn hơn (`a100_80g_multimodal`, độ phân giải 672², LoRA r=32, 5 epoch) đã được đề xuất trong `config.py` nhưng không được chạy do hạn chế thời gian.

---

## 9. Công việc tương lai (Future work)

Ba hướng follow-up đặc biệt hứa hẹn. Chúng được sắp xếp đại khái theo cải thiện kỳ vọng cho mỗi đơn vị nỗ lực.

### 9.1 Oversampling

Pipeline đã hỗ trợ oversampling theo từng severity qua `data_prep.py --oversample`. Điều này nhân đôi mỗi study huấn luyện theo tỷ lệ với worst-case severity của nó: Moderate × 2, Severe × 4. Trên dữ liệu hiện tại, điều này sẽ tăng train split từ 1,776 lên xấp xỉ 5,000 example trong khi giữ validation split không thay đổi. Tác động kỳ vọng, dựa trên phân tích trong §7.2, là một cải thiện non-trivial trong foraminal kappa (target $\kappa > 0.10$) với chi phí xấp xỉ 2.8× thời gian wall-clock huấn luyện nhiều hơn.

### 9.2 Profile multi-modality độ phân giải cao hơn

Profile `a100_80g_multimodal` giữ cùng multi-modality slice picker nhưng nâng độ phân giải từ 448² lên 672² (tức 1.5× tuyến tính), LoRA rank từ 16 lên 32, và số epoch từ 3 lên 5. Trên một A100 80 GB điều này sẽ tiêu thụ khoảng 65 GB VRAM trong huấn luyện. Độ phân giải cao hơn sẽ đặc biệt giúp foraminal narrowing, nơi tín hiệu fat-displacement liên quan nhỏ về mặt pixel. Tác động kỳ vọng: canal kappa $> 0.6$; overall kappa $> 0.55$.

### 9.3 Thay thế generative output bằng classification head

Một classification head fully-connected nhỏ trên pooled vision representation, được huấn luyện với cross-entropy class-balanced, sẽ sinh ra calibrated probability và cho phép weighted log-loss có ý nghĩa. Cùng backbone đã fine-tune có thể tiếp tục được sử dụng cho báo cáo ngôn ngữ tự nhiên, nhưng các con số headline sẽ được định nghĩa cùng cách như trong RSNA competition. Điều này invasive hơn hai thay đổi trước nhưng sẽ thu hẹp khoảng cách với các baseline CNN cổ điển.

### 9.4 Validation ngoài

Ngay cả một test set ngoài nhỏ (∼ 100 study) từ một viện khác sẽ tăng cường đáng kể bất kỳ tuyên bố nào về khả năng generalize. Không bộ dữ liệu lumbar MRI công khai nào hoàn hảo so sánh được với RSNA 2024, nhưng partial overlap trên ít nhất canal-stenosis grading là khả thi.

### 9.5 Deployment conversational

Mô hình fine-tune cũng có thể plug vào kiến trúc conversational hai-mô-hình: specialist LoRA-tuned phát ra JSON; một instance MedGemma base đọc JSON như context và converse với user bằng ngôn ngữ tự nhiên. Pattern này giữ accuracy classification mà không mất chat ergonomics mà base MedGemma cung cấp, và đã được prototype như một phần của dự án này (`demo_chat.py`).

---

## 10. Kết luận (Conclusion)

Chúng tôi đã chỉ ra rằng một bản fine-tune tiết kiệm tham số (QLoRA, $r = 16$) của MedGemma 1.5 4B có thể học để sinh ra chẩn đoán có cấu trúc 25-label về bệnh lý thoái hóa cột sống thắt lưng trực tiếp từ imagery MRI. Trên một validation split 198 study được giữ riêng của bộ dữ liệu RSNA 2024, Cohen's kappa tổng thể cải thiện từ **0.000** (mô hình base) lên **0.502** (đã fine-tune), với **tỷ lệ parse-failure JSON 0 %**. Ba trong số năm condition — spinal canal stenosis và cả hai subarticular stenosis — đạt giá trị kappa trong khoảng 0.43–0.54, chỉ ra agreement chẩn đoán thực sự. Hai condition foraminal-narrowing vẫn gần với chance, điều mà chúng tôi quy cho class imbalance và sự tinh tế trực quan của cue liên quan tại độ phân giải 448². Định dạng output compact-JSON, multi-modality slice picker, và label-only loss masking cùng nhau tạo ra một pipeline huấn luyện đủ nhỏ để chạy trên một A100 đơn lẻ dưới 15 giờ nhưng đủ trung thực để bao phủ cả 25 nhãn. Công việc này chỉ phục vụ nghiên cứu và chưa được validate lâm sàng; oversampling, một profile độ phân giải cao hơn, và một classification head được calibrate là ba thí nghiệm tiếp theo hứa hẹn nhất.

---

## 11. Tài liệu tham khảo (References)

Danh sách reference đã rút gọn; mở rộng với các entry BibTeX chính thức trong bản chuyển đổi LaTeX.

1. Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W. **LoRA: Low-Rank Adaptation of Large Language Models.** arXiv:2106.09685 (2021).
2. Dettmers, T., Pagnoni, A., Holtzman, A., Zettlemoyer, L. **QLoRA: Efficient Finetuning of Quantized LLMs.** arXiv:2305.14314 (2023).
3. Zhai, X., Mustafa, B., Kolesnikov, A., Beyer, L. **Sigmoid Loss for Language Image Pre-Training.** arXiv:2303.15343 (2023).
4. Google DeepMind. **MedGemma 1.5: A Vision-Language Model for Medical Image Understanding.** HuggingFace model card, 2025. https://huggingface.co/google/medgemma-1.5-4b-it
5. Google DeepMind. **Gemma 3 Technical Report.** 2025.
6. Cohen, J. **A Coefficient of Agreement for Nominal Scales.** *Educational and Psychological Measurement* 20(1):37–46 (1960).
7. Landis, J. R., Koch, G. G. **The Measurement of Observer Agreement for Categorical Data.** *Biometrics* 33(1):159–174 (1977).
8. Radiological Society of North America. **RSNA 2024 Lumbar Spine Degenerative Classification.** Kaggle competition (2024). https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification
9. von Werra, L., Belkada, Y., Tunstall, L., Beeching, E., Thrush, T., Lambert, N., Huang, S., Rasul, K., Gallouédec, Q., **TRL: Transformer Reinforcement Learning.** GitHub: huggingface/trl.
10. Han, J. & contributors. **Unsloth: 2× faster, 70 % less memory finetuning of LLMs.** GitHub: unslothai/unsloth.
11. Wolf, T., et al. **HuggingFace's Transformers: State-of-the-art Natural Language Processing.** EMNLP system demonstrations, 2020.

---

## Phụ lục A — Khả năng tái tạo (Reproducibility)

| Item | Giá trị |
|---|---|
| Repository | (local Git repo; SHA có sẵn qua `git rev-parse HEAD`) |
| HuggingFace adapter | `YOUR_USERNAME/medgemma-lumbar-finetune` (private/public theo model card) |
| Profile được sử dụng | `demo_a100_40g` |
| Random seed | 42 |
| Dependency lockfile | `setup.sh` (phiên bản đã pin) |
| Lệnh huấn luyện | `PROFILE=demo_a100_40g python train.py` |
| Lệnh đánh giá | `python evaluate.py --model_path /workspace/models/medgemma-lumbar/final` |
| Lệnh comparison | `python compare.py` |

## Phụ lục B — Metric tổng hợp ở dạng máy đọc được

Để có khả năng tái tạo, các con số headline từ §6.1 và §6.2 phản ánh các JSON dump được sinh bởi `evaluate.py` và `compare.py` (`evaluation_results/metrics.json`, `evaluation_results/ft_metrics.json`, `comparison/comparison_results.json`).

## Phụ lục C — Template inference

Inference tại thời điểm deployment sử dụng cùng user-prompt builder như huấn luyện:

```python
from data_prep import build_user_prompt

user_prompt = build_user_prompt(
    n_images=len(images),
    series_types=["sagittal_t2", "sagittal_t1", "axial_t2"],
)
# Pass `user_prompt` cộng với N <image> token qua tokenizer.apply_chat_template;
# decode response của mô hình và parse với json.loads.
```

Điều này giữ prompt tại thời điểm inference verbatim-giống hệt với prompt tại thời điểm SFT và ngăn ngừa distribution shift ngầm.
