import argparse
import numpy as np
import os
import cv2
import time
import tensorrt as trt
import pycuda.driver as cuda
import traceback # 引入 traceback
import pycuda.autoinit # 自動初始化 CUDA Context

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class DepthAnythingTRT(): 
    def __init__(self) -> None:
        
        self.augments() # 沿用參數設定
        self.root_weights = self.args.root_weights

        self.engine_path = os.path.join(self.root_weights, "onnx", f"depth_anything_v2_{self.args.encoder}_{self.args.precision}.engine")
        # self.engine_path = f"D:\Depth-Anything-V2\checkpoints\onnx\depth_anything_v2_{self.args.encoder}_{self.args.precision}.engine"

        # --- TensorRT 初始化 ---
        # self.trt_runtime = trt.Runtime(TRT_LOGGER)
        self.engine = self.load_engine()
        self.context = self.engine.create_execution_context()

        print(f"訊息: TensorRT Version {trt.__version__} ")

        # 依照TensorRT 版本檢查設定 context
        if hasattr(self.context, 'execute_async_v3'):
            print("訊息: 偵測到 TensorRT execute_async_v3 API。")
            self.execution_api_type = 'v3_async'
        elif hasattr(self.context, 'execute_async_v2'):
            print("訊息: 偵測到 TensorRT execute_async_v2 API。")
            self.execution_api_type = 'v2_async' 
        elif hasattr(self.context, 'execute_v2'): # 作為同步的備選方案
            print("訊息: 偵測到 TensorRT execute_v2 (同步) API。")
            self.execution_api_type = 'v2_sync'

        #=====如有需要可以增加 elif hasattr(self.context, 'enqueue'): 檢查其他版本
        else:
            # 如果上面都沒有找到，拋出錯誤
            if self.context: del self.context # 嘗試清理
            if self.engine: del self.engine
            raise RuntimeError("在 IExecutionContext 上找不到支援的執行方法 "
                                "(如 execute_async_v3, execute_async_v2, execute_v2)。"
                                "請檢查 TensorRT 版本或 Python bindings 是否正確。")
        
        # --- 分配 Host 和 Device 緩衝區 ---
        self.host_inputs = []
        self.host_outputs = []
        self.device_inputs = []
        self.device_outputs = []
        self.bindings = [] # 存放 device buffer 的指標
        self.stream = cuda.Stream() # 創建 CUDA stream

        self.input_shapes = {} # 儲存輸入的形狀
        self.input_dtypes = {} # 儲存輸入的資料類型
        self.output_shapes = {} # 儲存輸出的形狀

        print("--- 初始化引擎綁定 (Inputs/Outputs) ---")
        for binding in self.engine:
            # binding_idx = self.engine.get_binding_index(binding)

            # 獲取引擎定義的形狀和資料類型
            # 注意: get_binding_shape 對於有動態軸的引擎可能只回傳構建時的 profile 形狀
            # 如果使用動態形狀，實際推論時可能需要從 context.get_binding_shape(binding_idx) 獲取
            shape = tuple(self.engine.get_tensor_shape(binding)) # 轉換為 tuple
            dtype = trt.nptype(self.engine.get_tensor_dtype(binding))

            # 關於動態維度的警告
            if -1 in shape: # <--- 關鍵點2
                print(f"警告: 綁定 '{binding}' 包含動態維度 {shape}...")

            size = trt.volume(shape) # <--- 關鍵點3
            if size < 0:
                raise RuntimeError(f"計算綁定 '{binding}' 大小時出錯... 形狀: {shape}")
    
            try:
                # 分配 Host 和 Device 記憶體
                host_mem = cuda.pagelocked_empty(size, dtype) # 使用鎖頁記憶體加速 H2D/D2H
                device_mem = cuda.mem_alloc(host_mem.nbytes)
            except Exception as e:
                raise RuntimeError(f"記憶體分配失敗 for binding {binding}") from e

            # 將 device buffer 指標加入 bindings 列表
            self.bindings.append(int(device_mem))

            tensor_io_mode = self.engine.get_tensor_mode(binding)

            # 根據是 input 還是 output 分類
            if tensor_io_mode == trt.TensorIOMode.INPUT:
                self.host_inputs.append(host_mem)
                self.device_inputs.append(device_mem)
                self.input_shapes[binding]= shape
                self.input_dtypes[binding] = dtype
                print(f"Input Binding: {binding}, Shape: {shape}, Dtype: {dtype}")
            elif tensor_io_mode == trt.TensorIOMode.OUTPUT: 
                self.host_outputs.append(host_mem)
                self.device_outputs.append(device_mem)
                self.output_shapes[binding] = shape # 記住輸出形狀
                print(f"Output Binding: {binding}, Shape: {shape}, Dtype: {dtype}")
            else:
                print(f"訊息: 張量 '{binding}' 模式為 {tensor_io_mode}，非 I/O 張量，跳過加入 Host/Device 列表。")

        if not self.host_inputs:
            raise RuntimeError("錯誤: TensorRT 引擎中未找到任何輸入綁定。")
        if not self.host_outputs:
            raise RuntimeError("錯誤: TensorRT 引擎中未找到任何輸出綁定。")
        
        try:
            self.primary_input_name = next(name for name in self.engine 
                                            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT)
        except StopIteration:
            raise RuntimeError("未能識別主要輸入張量。")
        
        try:
            self.primary_output_name = next(name for name in self.engine 
                                            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT)
        except StopIteration:
            raise RuntimeError("未能識別主要輸出張量。")
        
        print("--- Bindings initialized ---")

        print(f"✅ TensorRT 引擎成功從 '{self.engine_path}' 載入並初始化。")
        print(f"✅ 主要輸入綁定: '{self.primary_input_name}', "
            f"形狀: {self.input_shapes[self.primary_input_name]}, "
            f"Dtype: {self.input_dtypes[self.primary_input_name]}")
        print(f"✅ 主要輸出綁定: '{self.primary_output_name}', "
            f"形狀: {self.output_shapes[self.primary_output_name]}")

    def load_engine(self):
        """載入 TensorRT 引擎檔案"""
        if not os.path.exists(self.engine_path):
            print(f"錯誤: 引擎檔案不存在 {self.engine_path}")
            return None
        print(f"正在從 {self.engine_path} 載入 TensorRT 引擎...")
        with open(self.engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            print("錯誤: 反序列化引擎失敗。")
            return None
        print("引擎載入完成。")
        return engine

    def predictions(self, frame_rgb:np.ndarray):
        """
        使用已載入的 TensorRT 引擎對單張 RGB 影像執行深度估測。

        Args:
            frame_rgb (np.ndarray): 輸入的 RGB 影像 (H, W, C) 格式，NumPy 陣列。

        Returns:
            np.ndarray | None: 預測的深度圖 (H_out, W_out)，如果成功則為 NumPy 陣列，否則為 None。
        """
        if not self.context or not self.host_inputs or not self.device_inputs or not self.host_outputs or not self.device_outputs:
            print("錯誤: TensorRT 元件在 predictions 方法中未正確初始化。")
            return None
        
        p_start = time.time()

        # --- 1. 準備輸入資料 (須與 ONNX/引擎建立時一致) ---
        #    - Resize (如果需要，但通常引擎建立時已固定尺寸)
        #    - BGR -> RGB (已傳入 frame_rgb)
        #    - HWC -> CHW
        #    - Normalize (例如 / 255.0)
        #    - 轉為 float32
        #    - 增加 Batch 維度 (如果引擎是這樣建立的，你的 ONNX 匯出是 [1, 3, H, W])

        input_name = self.primary_input_name
        output_name = self.primary_output_name
        host_input_buffer = self.host_inputs[0]

        # 取得 Host 輸入緩衝區 (假設只有一個輸入)
        try:
            input_shape = self.input_shapes[input_name]  # [1, 3, H, W]
            dtype = self.input_dtypes[input_name] # dtype: np.float32
        except KeyError:
            print(f"錯誤: 無法獲取主要輸入 '{input_name}' 的形狀或資料類型。")
            return None

        # 預處理範例 (請根據你的模型調整)
        if frame_rgb.shape[0] != input_shape[2] or frame_rgb.shape[1] != input_shape[3]:
            img = cv2.resize(frame_rgb, (input_shape[3], input_shape[2])) # Resize 到引擎尺寸 W, H 
            print(f"resize image as {input_shape[2]}x{input_shape[3]}")
        else:
            img = frame_rgb

        img = img.astype(dtype) / 255.0 # Normalize & Type
        img = img.transpose((2, 0, 1)) # HWC to CHW
        img = np.expand_dims(img, axis=0) # Add Batch dimension -> [1, 3, H, W]
        img = np.ascontiguousarray(img) # 確保記憶體連續

        if img.shape != input_shape: # input_shape 是引擎期望的 NCHW 形狀
            print(f"錯誤: 最終預處理後的影像形狀 {img.shape} 與引擎期望的輸入形狀 {input_shape} 不符。")
            return None

        # 增加了檢查，且通常是 ravel 後複製到一維 buffer
        if host_input_buffer.size != img.size:
            print(f"錯誤: Host 輸入緩衝區大小 ({host_input_buffer.size}) 與預處理後影像大小 ({img.size}) 不符。")
            return None
        if host_input_buffer.dtype != img.dtype:
            print(f"錯誤: Host 輸入緩衝區型別 ({host_input_buffer.dtype}) 與預處理後影像型別 ({img.dtype}) 不符。")
            return None
        
        # 將處理好的資料複製到 Host 輸入緩衝區
        # np.copyto(host_input_buffer, img.ravel()) # ravel() 將多維陣列攤平成一維
        # 或者直接賦值 (如果形狀匹配且使用鎖頁記憶體)
        np.copyto(host_input_buffer, img.ravel()) 

        # debug 使用檢查context 相關資訊
        '''
        print(f"類型 self.context: {type(self.context)}") # 打印 context 的確切類型
        print("可用的 self.context 方法和屬性:")
        print(dir(self.context))
        '''

        # --- 2. 執行推論 ---
        try:
            # H2D: Host -> Device
            cuda.memcpy_htod_async(self.device_inputs[0], host_input_buffer, self.stream)
            
            execution_successful = False # 用於追蹤 v3 API 的回傳值

            if self.execution_api_type == 'v3_async':
                # 設定主要輸入張量的位址
                self.context.set_tensor_address(self.primary_input_name, self.device_inputs[0])
                # 設定主要輸出張量的位址
                self.context.set_tensor_address(self.primary_output_name, self.device_outputs[0])

                if self.context.execute_async_v3(stream_handle=self.stream.handle):
                    execution_successful = True
                else:
                    print("錯誤: self.context.execute_async_v3() 回傳 False。")
            
            elif self.execution_api_type == 'execute_async_v2':
                self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
                execution_successful = True # 假設無異常即成功

            elif self.execution_api_type == 'v2_sync':
                # --- 使用 execute_v2 (同步) ---
                # 同步執行，不需要 stream_handle 參數給 execute_v2 本身
                print("警告: 正在使用同步的 execute_v2 API 進行推論。")
                self.context.execute_v2(bindings=self.bindings)
                execution_successful = True # 假設無異常即成功

            else: #  __init__ 中已經有檢查了
                print(f"錯誤: 未知的 self.execution_api_type: {self.execution_api_type}")
                return None

            if not execution_successful and self.execution_api_type == 'v3_async': 
                return None # 如果 v3 執行失敗
            
            # D2H: Device -> Host
            cuda.memcpy_dtoh_async(self.host_outputs[0], self.device_outputs[0], self.stream) 
            # Synchronize
            self.stream.synchronize()

        except Exception as e:
            print(f"錯誤: TensorRT 推論過程中 (API: {self.execution_api_type}) 發生錯誤: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        # --- 3. 取得輸出 ---
        # 結果在 self.host_outputs[0] (假設只有一個輸出)
        # 需要 reshape 回原本的輸出維度
        output_shape = self.output_shapes[output_name]
        depths = self.host_outputs[0].reshape(output_shape) # 已經是一維陣列
        # 假設輸出是 [1, H_out, W_out] 或 [1, 1, H_out, W_out] 之類的

        # 如果有 Batch 維度，去掉
        depths = depths.squeeze(0)

        if depths.ndim != 2:
            print(f"警告: 推論結果 'squeeze' 後的維度為 {depths.ndim} (期望是 2D)。\n原始輸出形狀: {output_shape}, Squeezed 後形狀: {depths.shape}")


        p_end = time.time()
        print(f"[predictions TRT] elapsed: {(p_end - p_start) * 1000:.2f} ms")

        return depths

    def run(self):
        # ... (之前的 run 方法大部分邏輯可以保留) ...
        # 主要修改是調用 self.predictions(frame_rgb)

        src = 0 if self.args.video_path == '0' else self.args.video_path
        print(f"Images Source {self.args.video_path}")
        cap = cv2.VideoCapture(src)
        assert cap.isOpened(), f'Cannot Open {src}'

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 保留原始尺寸 W, H (如果視覺化需要)
            # W_orig, H_orig = frame.shape[:2]
            # print(f"Source Images Size{W_orig, H_orig}")

            # 預處理通常在 predictions 內部完成，這裡只需要 BGR -> RGB
            frame_rgb = frame[:, :, ::-1].copy()

            #--- 單張推論 ---
            run_start = time.time()
            depth = self.predictions(frame_rgb) # 這裡會調用新的 TRT predictions
            print(f"Predictions Shape : {depth.shape}") # 應該是 [H_out, W_out]

            #--- 視覺化 (大致不變) ---
            # 注意：深度圖的尺寸可能是引擎的輸出尺寸，需要 resize 回顯示尺寸
            vis_width = int(frame.shape[1] * self.args.video_scale / 100)
            vis_height = int(frame.shape[0] * self.args.video_scale / 100)
            vis_dim = (vis_width, vis_height)

            # --- normalize depth to [0-1] then convert to [0-225]
            depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            depth_colormap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_MAGMA) # 或其他 colormap

            # --- resize image and depth map to visualization size ---
            frame_vis = cv2.resize(frame, vis_dim, interpolation=cv2.INTER_AREA)
            depth_colormap_vis = cv2.resize(depth_colormap, vis_dim, interpolation=cv2.INTER_AREA)

            latency = (time.time() - run_start) # 這包含了預處理+推論+取得輸出
            print(f"[run TRT] elapsed: {latency * 1000:.2f} ms") # 注意這個時間包含的範圍
            Result = np.hstack((frame_vis, depth_colormap_vis))
            fps = 1.0 / latency if latency > 0 else 0
            cv2.putText(Result, f"FPS : {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.imshow("Result", Result)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break # 不再返回 True，直接 break

        cap.release()
        cv2.destroyAllWindows()

    def augments(self):
        parser = argparse.ArgumentParser(description='Depth Anything V2 - TensorRT Inference')
        parser.add_argument('--video_path', type=str, default="./videos/4.mp4",  help='Image source: video_path or 0 for webcam')
        parser.add_argument('--outdir', type=str, default='./vis_video_depth_trt') 
        parser.add_argument('--video_scale', type=int , default=40 ,help='Percentage to shrink display window (e.g., 50)')

        parser.add_argument('--root_weights', type=str, default="./checkpoints", help='The root directory of weights')


        # 讀取 engine 參數
        parser.add_argument('--encoder', type=str, default='vitb', choices=['vits', 'vitb', 'vitl', 'vitg'])
        parser.add_argument('--precision', type=str, default='fp16', choices=['fp32', 'fp16'], help='Precision mode for TensorRT engine')

        
        self.args = parser.parse_args()

if __name__=="__main__":
    # 建立實例並傳入 engine 路徑
    DAv2_trt = DepthAnythingTRT()
    DAv2_trt.run()