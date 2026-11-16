# Ghi nhận sử dụng công cụ AI

**Dự án:** Phân loại loài gỗ từ ảnh vân gỗ - INT3401, lớp 11, Nhóm 18.

**Công cụ:** ChatGPT/Codex.

Bản ghi được tổng hợp khi hoàn thiện dự án từ các công việc có thể đối chiếu trong phiên hỗ trợ và repository. Đây không phải nhật ký được tạo đồng thời cho mọi hoạt động trước đó; không xác định tỷ lệ đóng góp hay thay đổi phân công của các thành viên.

## Mục đích và phạm vi hỗ trợ

| Công việc | Phần AI hỗ trợ | Sản phẩm để đối chiếu |
|---|---|---|
| Rà soát mã nguồn | Đọc luồng xử lý; phát hiện và chỉnh sửa sự khác biệt tiền xử lý giữa suy luận, benchmark và ảnh phủ Grad-CAM; xử lý vòng đời hook | `src/dataset.py`, `src/inference.py`, `src/gradcam.py`, `src/predict.py` |
| Huấn luyện và baseline | Kết nối cấu hình tăng cường dữ liệu với mã huấn luyện; bổ sung lưu mô hình SVM để sử dụng lại | `src/train.py`, `src/baseline_lbp_svm.py`, `configs/` |
| Giao diện minh họa | Tổ chức hai tab; dùng chung luồng suy luận; báo thiếu checkpoint thay vì trả dự đoán từ trọng số ngẫu nhiên | `app.py` |
| Kiểm chứng | Xây dựng và chạy kiểm thử; chạy lại suy luận trên tập Test, đối chiếu ma trận nhầm lẫn đã lưu | `tests/`, `outputs/metrics/resnet18_color_aug_test_metrics.json` |
| Tổ chức và đóng gói | Rà soát tệp không dùng; loại dữ liệu, checkpoint và thư viện cài sẵn khỏi phần theo dõi Git và gói nộp; viết hướng dẫn chạy | `.gitignore`, `scripts/make_submission_zip.py`, `README.md` |
| Biên tập báo cáo | Rà soát logic trình bày, định dạng LaTeX; đối chiếu số liệu; điều chỉnh nhận định thiếu minh chứng và nêu giới hạn triển khai HF | `report/main.tex`, `report/main.pdf` |

Các nội dung và mã do AI đề xuất đều cần được các thành viên kiểm tra trước khi sử dụng trong bản nộp.

## Minh chứng và kết quả kiểm tra

- [Commit hoàn thiện mã nguồn và báo cáo](https://github.com/Anomal-gre/Y3S1_Wood_Grain_Classification/commit/0b6676c838fbbe9f967cefb144f376e7dc65e480) cho phép xem thay đổi cụ thể.
- **9 kiểm thử tự động đã chạy thành công**, gồm kiểm tra hình học tiền xử lý/Grad-CAM, suy luận khi thiếu checkpoint, huấn luyện head, lưu lại SVM, chia tập và đóng gói. Lệnh chạy lại: `python -m unittest discover -s tests -v`.
- **Suy luận lại 2.836 ảnh Test:** ma trận nhầm lẫn khớp kết quả đã lưu, **2.630 ảnh đúng**, tương ứng Accuracy 92,74%. Đây là đối chiếu suy luận từ checkpoint hiện có; không phải huấn luyện lại, kiểm thử trên dữ liệu mới hay bằng chứng độc lập về khả năng tổng quát hóa.
- Các kết quả gốc nằm trong `outputs/metrics/`. Số liệu chưa có minh chứng cho ResNet50, ConvNeXt-Tiny và các đối chứng không được trình bày như kết quả đã hoàn thành.
- Biên dịch PDF và kiểm tra bố cục được thực hiện sau khi cập nhật nội dung. Gói nộp được tạo lại bằng `python scripts/make_submission_zip.py --group 18`, không chứa dataset hay checkpoint.

## Trách nhiệm và giới hạn

Các thành viên chịu trách nhiệm đọc, kiểm tra, hiểu, giải thích và xác nhận mã nguồn, số liệu, tài liệu tham khảo và nội dung cuối cùng trước khi nộp. Kiểm thử tự động và bản ghi AI không thay thế việc nghiệm thu của nhóm; bản ghi này không khẳng định các thành viên đã hoàn tất bước đó.

Không khai báo tỷ lệ đóng góp AI hoặc thời lượng công việc khi chưa có căn cứ. Việc đối chiếu checkpoint không xác nhận đã tái lập toàn bộ huấn luyện; việc có mã triển khai không xác nhận Hugging Face đang hoạt động đầu cuối.

## Ghi nhận trên CoReach

Reach do người dùng cung cấp: [Thảo luận của Nhóm 18](https://coreach.ai/app/reaches/3508e61c-b8af-4200-9568-3507a1cc9303/discussion).

Nội dung dùng để ghi nhận trên Reach:

> **Ghi nhận sử dụng AI khi hoàn thiện dự án - Nhóm 18**
>
> Công cụ: ChatGPT/Codex. AI hỗ trợ rà soát và chỉnh sửa mã nguồn, thống nhất tiền xử lý và Grad-CAM, tổ chức dự án, xây dựng kiểm thử, biên tập README/báo cáo LaTeX và đối chiếu số liệu với kết quả thực nghiệm đã lưu.
>
> Kiểm chứng: 9 kiểm thử tự động chạy thành công; suy luận lại trên 2.836 ảnh Test cho ma trận nhầm lẫn khớp kết quả đã lưu, với 2.630 ảnh đúng. Đây là kiểm tra checkpoint hiện có, không phải một lần huấn luyện lại hay đánh giá trên dữ liệu mới. PDF được biên dịch lại và gói nộp được cập nhật, loại dataset/checkpoint.
>
> Minh chứng mã nguồn: https://github.com/Anomal-gre/Y3S1_Wood_Grain_Classification/commit/0b6676c838fbbe9f967cefb144f376e7dc65e480
>
> Bản ghi chi tiết: https://github.com/Anomal-gre/Y3S1_Wood_Grain_Classification/blob/main/docs/AI_USAGE.md
>
> Các thành viên chịu trách nhiệm rà soát, hiểu, giải thích và xác nhận bản cuối trước khi nộp. Bản ghi này tổng hợp các hỗ trợ đã có minh chứng, không thay đổi phân công thành viên và không tự xác nhận nhóm đã nghiệm thu.
