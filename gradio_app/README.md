# CXR Evidence — Gradio

Chuẩn bị môi trường và các artefact ngoài Git: xem [SETUP.md](../docs/SETUP.md).

Giao diện Gradio cho dự án hiện tại, theo bố cục trắng–xanh ngọc của
`D:\webKhoaLuan`. Không phụ thuộc runtime vào thư mục tham chiếu đó.

Chạy từ `D:\image-rag`:

```powershell
.\StartGradio.ps1
```

Hoặc nhấp đúp `StartGradio.bat`. Mở http://127.0.0.1:7860. Dừng bằng Ctrl+C.
Đổi cổng bằng `-Port 7861`, không mở trình duyệt bằng `-NoBrowser`, chỉ cài
thư viện bằng `-SetupOnly`.

Môi trường web riêng: `gradio_app/.venv` (Gradio 6.32.0). Worker nghiên cứu dùng
`.venv-bertscore/Scripts/python.exe` và CUDA hiện có. Có thể chỉ định interpreter
nghiên cứu bằng biến `CXR_RESEARCH_PYTHON`. Không thay thư viện của môi trường
nghiên cứu để cài giao diện. Khởi động và xem report lịch sử không gọi API.

Giao diện có ba mục:

- **Phân tích:** tải 1–10 ảnh PNG/JPEG; chọn DenseNet121/ResNet50, model GPT,
  reasoning, một phương pháp hoặc cả bốn. Mặc định Luna/high/8192/CUDA,
  local response cache tắt. Mỗi ảnh được suy luận độc lập.
- **Nghiên cứu:** đọc Table 5/6 của lượt high hoàn tất 01/11/2026, cùng bảng
  văn bản đánh giá lại 13/11/2026. Mở 4 report lịch sử theo study không sinh lại.
- **Lịch sử:** xem lại ca mới, report, classifier, support/non-support, route,
  gate, response ID và actual token usage; tải JSON gốc.

Mỗi ca mới nằm trong `outputs/gradio_cases/<uuid>/`. Không ghi đè output nghiên
cứu. Trên giao diện, nút phân tích sử dụng endpoint/key từ `.env` qua client hiện
có và có thể phát sinh phí API. Key không đưa vào frontend.

DenseNet121 là classifier mặc định. Checkpoint ResNet50 được sao chép nguyên
byte từ web tham chiếu vào `gradio_app/models/`; SHA-256:
`a304a6c73a2a0543314cace0fdd31666e1e1534584d0ae41c3a7bdf7e92f08a0`.
ResNet50 vẫn dùng ngưỡng/gate DenseNet đóng băng, **chưa hiệu chuẩn hoặc kiểm
định riêng**. Retrieval vẫn dùng bộ trích xuất DenseNet của nghiên cứu.

Ảnh IU-Xray gốc được nhận diện bằng SHA để loại chính study khỏi truy hồi. Với
ảnh đã đổi byte/re-encode, nhập Study ID nếu biết. Learned gate dùng đúng fold
held-out cho study trong folds.json; ca ngoài tập dùng fold 0 đóng băng và ghi
rõ không phải đánh giá OOF. Không huấn luyện lại gate.

Table 5: 206 study; Table 6: cohort chung 205 study, loại study 1615 vì hai
reference rỗng. Bảng văn bản: Findings 163, Impression 204 reference không rỗng.
Không tạo reference thay thế, không tính accuracy cho upload không có nhãn.

Kiểm tra backend không gọi API:

```powershell
.\.venv-bertscore\Scripts\python.exe -m unittest discover -s tests -p test_gradio_backend.py -v
```

`app.py` dùng native Gradio components và event callbacks. `styles.css` chỉ định
kiểu trình bày; không có frontend HTML/JavaScript riêng. `backend.py` quản lý ca
và subprocess; worker/classifiers/presentation kế thừa adapter tham chiếu,
không sửa classifiers/retriever/prompts/frozen gate trong nghiên cứu.
