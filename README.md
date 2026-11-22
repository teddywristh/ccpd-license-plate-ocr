# ccpd-license-plate-ocr

Nhận dạng chuỗi ký tự biển số từ ảnh CCPD2020 Green bằng PyTorch.

Mô hình dùng kiến trúc:

```text
ảnh [1, 48, 192]
    -> CNN Encoder
    -> Attention
    -> LSTM Decoder
    -> chuỗi ký tự
```

Decoder dùng ba token đặc biệt:

- `BOS`: bắt đầu chuỗi.
- `EOS`: kết thúc chuỗi.
- `PAD`: phần đệm cho các chuỗi trong cùng batch.

Khi huấn luyện, decoder nhận chuỗi đầu vào đúng. Khi đánh giá hoặc dự đoán,
decoder bắt đầu từ `BOS` và tự dùng ký tự vừa dự đoán cho bước tiếp theo.

## Cấu trúc chính

- `src/labels.py`: mã hóa và giải mã nhãn CCPD.
- `src/dataset.py`: đọc ảnh, cắt biển số và tạo batch Seq2Seq.
- `src/model.py`: CNN Encoder, Attention, LSTM Decoder.
- `src/train.py`: huấn luyện và lưu checkpoint tốt nhất.
- `src/predict.py`: dự đoán một ảnh.
- `src/evaluate.py`: đánh giá trên tập test.

## Kiểm tra dữ liệu và model

Chạy từ thư mục gốc project:

```powershell
python src/check_data.py
```

## Huấn luyện

```powershell
python src/train.py
```

Checkpoint có validation loss thấp nhất được lưu tại:

```text
checkpoints/best_model.pt
```

Log huấn luyện được ghi tại `checkpoints/training.log`. Có thể theo dõi log
trên PowerShell bằng:

```powershell
.\view_training_log.ps1
```

## Dự đoán một ảnh

```powershell
python src/predict.py data/CCPD2020/ccpd_green/test/ten_anh.jpg
```

Có thể chỉ định checkpoint khác:

```powershell
python src/predict.py data/CCPD2020/ccpd_green/test/ten_anh.jpg --checkpoint checkpoints/best_model.pt
```

## Đánh giá trên test

```powershell
python src/evaluate.py
```

Kết quả được lưu tại `outputs/test_predictions.csv` và
`outputs/evaluation_metrics.csv`. Các chỉ số chính gồm Exact Plate Accuracy,
Character Accuracy, CER, MAE, RMSE, R² và NSE.

