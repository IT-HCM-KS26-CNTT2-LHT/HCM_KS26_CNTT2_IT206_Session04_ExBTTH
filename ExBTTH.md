# [Phân tích] So sánh Trade-off các giải pháp ra lệnh cho Kỹ thuật Prompting Nâng cao (Logic & Chuỗi)

## 1. Bài toán

**Tình huống:** Hệ thống tiếp nhận phản hồi khách hàng và tự động phân loại, đề xuất hướng xử lý, sau đó soạn câu trả lời.

**Mục tiêu:** Tạo chuỗi Prompt Chaining chạy tuần tự; đầu ra của bước trước là dữ liệu đầu vào của bước sau. Hệ thống kiểm tra dữ liệu và có phương án dự phòng khi AI trả về sai định dạng.

## 2. Thiết kế giải pháp

```text
Phản hồi khách hàng
        ↓
Bước 1 – Phân loại và trích xuất thông tin
        ↓ (JSON hợp lệ)
Bước 2 – Phân tích vấn đề và đề xuất xử lý
        ↓ (JSON hợp lệ)
Bước 3 – Soạn câu trả lời và tổng kết
        ↓
Kiểm tra đầu ra → Kết quả cuối
```

- **Chain-of-Thought (CoT):** Yêu cầu AI suy luận nội bộ theo từng tiêu chí; chỉ xuất kết luận và lý do ngắn gọn, không yêu cầu hiển thị toàn bộ suy nghĩ nội bộ.
- **Prompt Chaining:** Chia bài toán thành 3 bước nhỏ, truyền kết quả theo cấu trúc JSON.
- **Xử lý đứt gãy:** Kiểm tra kiểu dữ liệu, kiểm tra JSON, thử lại một lần. Nếu vẫn lỗi, dùng dữ liệu dự phòng và đánh dấu cần người kiểm tra.

## 3. Prompt thực thi từng bước

### Prompt 1 – Phân loại phản hồi

```text
Bạn là chuyên viên phân loại phản hồi khách hàng.
Đọc nội dung phản hồi, suy luận nội bộ và trả về DUY NHẤT JSON hợp lệ.
Không tự thêm thông tin không có trong dữ liệu.

Schema:
{
  "category": "nhóm vấn đề",
  "sentiment": "positive | neutral | negative",
  "priority": "low | medium | high",
  "summary": "tóm tắt tối đa 25 từ",
  "key_facts": ["thông tin quan trọng"]
}

Phản hồi: {feedback}
```

### Prompt 2 – Phân tích và đề xuất xử lý

```text
Bạn là nhân viên hỗ trợ khách hàng.
Dựa CHỈ trên dữ liệu JSON ở bước 1, phân tích vấn đề và đề xuất hành động thực tế.
Không khẳng định nguyên nhân khi chưa đủ bằng chứng.
Chỉ trả về JSON hợp lệ theo schema:
{
  "possible_causes": ["nguyên nhân có thể"],
  "actions": ["việc cần thực hiện"],
  "risk_note": "điểm cần xác minh"
}

Dữ liệu bước 1: {step1_json}
```

### Prompt 3 – Tạo câu trả lời cuối

```text
Bạn là nhân viên chăm sóc khách hàng, giọng lịch sự và rõ ràng.
Dựa trên dữ liệu bước 1 và bước 2, viết câu trả lời ngắn gọn.
Không hứa điều chưa chắc chắn, không bịa thông tin hoặc kết quả xử lý.
Chỉ trả về JSON hợp lệ:
{
  "reply": "câu trả lời gửi khách hàng",
  "internal_summary": "tóm tắt hướng xử lý cho nhân viên",
  "status": "ready | needs_review"
}

Dữ liệu bước 1: {step1_json}
Dữ liệu bước 2: {step2_json}
Nếu dữ liệu đầu vào có lỗi hoặc thiếu thông tin, đặt status = "needs_review".
```

## 4. Code Python mẫu

Code không phụ thuộc nhà cung cấp AI. Hàm `llm_call(prompt)` cần được kết nối với API/mô hình đang sử dụng và trả về chuỗi JSON hoặc dictionary.

```python
import json


def normalize_input(raw):
    """Chấp nhận chuỗi, dictionary, danh sách hoặc dữ liệu khác."""
    if raw is None:
        return ""
    if isinstance(raw, str):
        text = raw
    elif isinstance(raw, dict):
        value = next(
            (raw[k] for k in ("feedback", "message", "content")
             if isinstance(raw.get(k), str) and raw[k].strip()),
            None,
        )
        text = value if value is not None else json.dumps(
            raw, ensure_ascii=False, default=str
        )
    elif isinstance(raw, list):
        text = "\n".join(str(item) for item in raw if item is not None)
    else:
        text = str(raw)
    return text.strip()[:3000]  # Giới hạn token đầu vào


def ask_json(llm_call, prompt, required_keys, fallback):
    """Gọi AI, kiểm tra JSON, thử lại một lần rồi dùng dữ liệu dự phòng."""
    for attempt in range(2):
        try:
            result = llm_call(prompt)
            if isinstance(result, dict):
                data = result
            else:
                text = str(result).strip()
                if text.startswith("```"):
                    text = text.strip("`")
                    if text.lower().startswith("json"):
                        text = text[4:].strip()
                data = json.loads(text)

            if not isinstance(data, dict):
                raise ValueError("Đầu ra không phải JSON object")
            if not all(key in data for key in required_keys):
                raise ValueError("Thiếu trường dữ liệu bắt buộc")
            return data, True
        except Exception:
            # Thử lại với yêu cầu định dạng rõ ràng hơn.
            prompt += "\nChỉ trả về một JSON object hợp lệ, đủ tất cả trường bắt buộc."
    return fallback, False


def run_pipeline(raw_feedback, llm_call):
    feedback = normalize_input(raw_feedback)
    if not feedback:
        return {
            "status": "needs_review",
            "message": "Dữ liệu phản hồi trống hoặc không hợp lệ.",
        }

    # Bước 1: Phân loại
    p1 = f'''Phân loại phản hồi. Chỉ trả về JSON với các trường:
category, sentiment, priority, summary, key_facts.
Không suy đoán ngoài dữ liệu.
Phản hồi: {feedback}'''
    step1, ok1 = ask_json(
        llm_call, p1,
        ["category", "sentiment", "priority", "summary", "key_facts"],
        {"category": "unknown", "sentiment": "neutral", "priority": "low",
         "summary": "Chưa phân loại được", "key_facts": []},
    )

    # Bước 2: Phân tích – dùng trực tiếp kết quả bước 1
    p2 = "Phân tích JSON sau. Chỉ trả JSON gồm possible_causes, actions, risk_note. " \
         "Không khẳng định điều chưa xác minh.\n" + json.dumps(step1, ensure_ascii=False)
    step2, ok2 = ask_json(
        llm_call, p2, ["possible_causes", "actions", "risk_note"],
        {"possible_causes": [], "actions": ["Chuyển nhân viên kiểm tra"],
         "risk_note": "AI chưa phân tích được; cần xác minh thủ công."},
    )

    # Bước 3: Soạn phản hồi – dùng kết quả hai bước trước
    p3 = "Soạn câu trả lời lịch sự, không bịa thông tin. Chỉ trả JSON gồm " \
         "reply, internal_summary, status. Đặt status=needs_review nếu có dữ liệu dự phòng.\n" \
         + json.dumps({"step1": step1, "step2": step2}, ensure_ascii=False)
    step3, ok3 = ask_json(
        llm_call, p3, ["reply", "internal_summary", "status"],
        {"reply": "Chúng tôi đã ghi nhận phản hồi và sẽ kiểm tra thêm.",
         "internal_summary": "Chuyển nhân viên kiểm tra thông tin.",
         "status": "needs_review"},
    )

    # Nếu bất kỳ bước nào lỗi, không tự động coi kết quả là hoàn tất.
    if not (ok1 and ok2 and ok3):
        step3["status"] = "needs_review"

    return {
        "classification": step1,
        "analysis": step2,
        "final": step3,
        "pipeline_ok": ok1 and ok2 and ok3,
    }
```

> **Lưu ý kỹ thuật:** Khối bắt lỗi giúp pipeline không dừng đột ngột khi API/mô hình lỗi. Khi triển khai thật, nên ghi log ở phía máy chủ và không đưa dữ liệu cá nhân nhạy cảm vào prompt nếu không cần thiết.

## 5. Kết quả đầu ra minh họa

**Dữ liệu đầu vào:**

```text
Tôi đặt hàng 5 ngày chưa nhận. Ứng dụng báo đã giao nhưng tôi chưa thấy hàng.
```

**Kết quả AI sau chuỗi 3 bước:**

```json
{
  "classification": {
    "category": "Giao hàng",
    "sentiment": "negative",
    "priority": "high",
    "summary": "Khách chưa nhận hàng dù ứng dụng hiển thị đã giao.",
    "key_facts": ["Đặt hàng 5 ngày", "Ứng dụng báo đã giao", "Khách chưa nhận hàng"]
  },
  "analysis": {
    "possible_causes": ["Trạng thái giao hàng cập nhật chưa chính xác", "Giao nhầm địa chỉ hoặc người nhận"],
    "actions": ["Kiểm tra mã vận đơn và bằng chứng giao hàng", "Liên hệ đơn vị vận chuyển", "Cập nhật kết quả cho khách"],
    "risk_note": "Chưa đủ thông tin để xác định nguyên nhân."
  },
  "final": {
    "reply": "Chúng tôi xin lỗi vì sự bất tiện này. Chúng tôi sẽ kiểm tra trạng thái đơn hàng với đơn vị vận chuyển và cập nhật kết quả cho bạn sớm nhất.",
    "internal_summary": "Ưu tiên kiểm tra trạng thái và bằng chứng giao hàng trước khi kết luận.",
    "status": "ready"
  },
  "pipeline_ok": true
}
```

*Kết quả trên là ví dụ minh họa định dạng đầu ra; kết quả thực tế phụ thuộc mô hình và dữ liệu được gửi vào.*

## 6. Kết luận

Giải pháp chia bài toán lớn thành 3 prompt nhỏ, liên kết bằng JSON để dễ kiểm soát. CoT được dùng để hỗ trợ AI suy luận nội bộ; Prompt Chaining giúp quy trình chạy tuần tự. Kiểm tra dữ liệu, thử lại và dữ liệu dự phòng giúp hạn chế lỗi định dạng hoặc gián đoạn, đồng thời chuyển trường hợp không chắc chắn sang người kiểm tra.
