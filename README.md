# Depth Anything V2 - TensorRT 擴充版

本專案是基於原始 [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) 所延伸，**新增支援 ONNX / TensorRT 的壓縮與影片推論功能**，可更高效地部署於 GPU 加速推論場景中。

> 🎥 支援讀取影像ROS2 Image 格式 topic 進行推論
> ⚙️ 支援 PyTorch → ONNX → TensorRT 引擎轉換流程
> 🚀 支援 FP32 / FP16 模式選擇與自動化權重轉換

---
## 📊 模型推論效能比較 (FPS)
**🟠 原始 PyTorch 模型（vit-s）**
![](assets/vits.gif)

**🟡 ONNX 模型推論（vit-s）**
![](assets/vits_onnx.gif)

**🔵 TensorRT FP32 推論（vit-s）**
![](assets/vits_engine_fp32.gif)

**🟢 TensorRT FP16 推論（vit-s）**
![](assets/vits_engine_fp16.gif)

---
## 🎬 推論效果預覽

|模型架構       | 原始 PyTorch | ONNX | TensorRT FP32 | TensorRT FP16 |
|---------------|------|------|------|------|
| `vit-s`       |40–41 FPS|66–68 FPS|71–76 FPS|76–82 FPS |
| `vit-b`       |22–23 FPS|33–35 FPS|37–39 FPS|90–100 FPS|
| `vit-ly`      | 9–10 FPS|13–14 FPS|13–14 FPS|55–58 FPS |

---
## 💻 系統環境資訊 (環境建議)

| 套件名稱       | 版本 |
|---------------|------|
|Python|3.8.10|
|PyTorch|1.12.1+cu113|
|TorchVision|0.13.1+cu113|
|Torchaudio|0.12.1+cu113|
|TensorRT|10.10.0.31|
|ONNX|1.13.1|
|ONNXRuntime-GPU|1.14.1|
---
## 🧠 模型權重下載

請至以下 Google Drive 下載對應的模型與 TensorRT 引擎檔案：

🔗 [Depth Anything V2 Weights - Google Drive](https://drive.google.com/drive/folders/1FIeJFCWv2RBRNA9CVut9nqfSIA7yKgyH?usp=drive_link)

建議將下載後的檔案放置於 `./checkpoints` 目錄下。
建議將.pth檔案下載後的檔案放置於 `./checkpoints/torch` 目錄下。

---

## 🚀 推論流程快速開始

### 🔧 參數說明

| 參數名稱                 | 說明                                                                 |
|--------------------------|----------------------------------------------------------------------|
| `depth_model_type`       | 選擇模型來源，`onnx_hybrid` 使用 PyTorch 或 ONNX 模型，`tensorrt` 使用 .engine |
| `encoder`                | 模型大小選擇，支援 `vits`, `vitb`, `vitl`, `vitg` 等                 |
| `use_trt`                | 若為 `true` 則會以 ONNX 模型執行 TensorRT 壓縮與加速                 |
| `precision`              | 精度模式，支援 `fp16` 或 `fp32`（當 `use_trt=true` 或 `tensorrt` 必填） |
| `onnx_hybrid_workspace`  | TensorRT 壓縮工作記憶體空間（單位 GB，預設為 4）                        |
| `start_rviz`             | 是否啟動 RViz 顯示（`true` 或 `false`）  

---

### 🔹 使用 PyTorch 權重（onnx_hybrid + 不壓縮）
```bash
ros2 launch depth_Anything_V2 depth_anything.launch.py \
  depth_model_type:=onnx_hybrid \
  use_trt:=false \
  encoder:=vits
```

### 🔹 使用onnx 權重開始推論且同時檢查並壓縮.onnx與.engine權重
```bash
ros2 launch depth_Anything_V2 depth_anything.launch.py \
  depth_model_type:=onnx_hybrid \
  use_trt:=true \
  encoder:=vits \
  precision:=fp32
```


## 🔹 使用TensorRT fp16精度的.enine權重開始推論
```bash
ros2 launch depth_Anything_V2 depth_anything.launch.py \
  depth_model_type:=tensorrt \
  encoder:=vits \
  precision:=fp16
```

## 🔹 使用TensorRT fp32精度的.enine權重開始推論
```bash
ros2 launch depth_Anything_V2 depth_anything.launch.py \
  depth_model_type:=tensorrt \
  encoder:=vits \
  precision:=fp32\
```

---

##  Citation

本專案基於以下優秀開源研究成果延伸開發，特此感謝原始作者社群的貢獻：

- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)
- [Depth Anything ONNX](https://github.com/fabio-sim/Depth-Anything-ONNX)

若您在研究或應用上受本專案或其基礎模型啟發，請考慮引用以下論文：

```bibtex
@article{depth_anything_v2,
  title={Depth Anything V2},
  author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and Zhao, Zhen and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
  journal={arXiv:2406.09414},
  year={2024}
}

@inproceedings{depth_anything_v1,
  title={Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data}, 
  author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
  booktitle={CVPR},
  year={2024}
}

@misc{oquab2023dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime et al.},
  journal={arXiv:2304.07193},
  year={2023}
}
