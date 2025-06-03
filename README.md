# Depth Anything V2 - TensorRT 擴充版

本專案是基於原始 [Depth Anything V2](https://github.com/isl-org/Depth-Anything) 所延伸，**新增支援 ONNX / TensorRT 的壓縮與影片推論功能**，可更高效地部署於 GPU 加速推論場景中。

> 🎥 支援即時影片輸入推論（Webcam 或 MP4）
> ⚙️ 支援 PyTorch → ONNX → TensorRT 引擎轉換流程
> 🚀 支援 FP32 / FP16 模式選擇與自動化權重轉換

---

## 📁 專案結構與主要程式說明

| 檔案名稱       | 說明 |
|----------------|------|
| `demo_onnx.py` | 最基礎推論腳本，支援載入原始 PyTorch 或 ONNX 權重模型，並可進行 ONNX / TensorRT 引擎壓縮。 |
| `trt.py`       | 以 TensorRT engine 執行的高效率推論腳本，使用 PyCUDA 進行記憶體與流控制管理。 |
| `export.py`    | 專門用來匯出模型為 ONNX 並建立 TensorRT 引擎，支援 FP16 / FP32 模式。 |
| `demo_trt.py`  | 綜合推論腳本（正在整合中），目標是整合原始模型 → ONNX → TensorRT 並支援影片推論與自動建構流程。 |

---

## 🧠 模型權重下載

請至以下 Google Drive 下載對應的模型與 TensorRT 引擎檔案：

🔗 [Depth Anything V2 Weights - Google Drive](https://drive.google.com/drive/folders/1FIeJFCWv2RBRNA9CVut9nqfSIA7yKgyH?usp=drive_link)

建議將下載後的檔案放置於 `./checkpoints` 目錄下。

---

## 🚀 推論流程快速開始

### 🔹 使用 ONNX 推論 / 建立.onnx and .engine weights：
```bash
python demo_onnx.py \
  --encoder vitb \
  --trt True \
  --precision fp16 \
  --video_path ./videos/test.mp4
  
```

### 🔹 使用 engine 推論 ：
```bash

python trt.py \
  --encoder vitb \
  --precision fp16 \
  --video_path ./videos/test.mp4

```
