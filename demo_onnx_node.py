#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import numpy as np
import os
import torch
import cv2
import time
import onnxruntime as ort
import sys

# 假設您的環境中已安裝 TensorRT。如果沒有，ROS 節點仍可使用 ONNX 或 PyTorch 運行。
try:
    import tensorrt as trt
except ImportError:
    rospy.logwarn("TensorRT library not found. TensorRT functionalities will be disabled.")
    trt = None

# --- 核心深度估計類別 ---
# 這部分是從您提供的 demo_onnx.py 移植並修改而來，以適應 ROS 環境。

# 假設 depth_anything_v2/dpt.py 與此腳本在同一個 Python 路徑下
# 在 ROS 中，建議將 depth_anything_v2 資料夾放在此腳本的同一目錄下
try:
    from depth_anything_v2.dpt import DepthAnythingV2
except ImportError:
    rospy.logerr("Failed to import DepthAnythingV2. Make sure the 'depth_anything_v2' directory is in the same folder as this script or in your PYTHONPATH.")
    sys.exit(1)

class DepthAnything:
    def __init__(self, args) -> None:
        """
        初始化深度估計模型。
        Args:
            args (dict): 包含所有設定的字典，從 ROS 參數伺服器讀取。
        """
        self.args = args
        self.model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        
        # 定義模型和權重路徑
        self.checkpoints_path = self.args.get('checkpoints_path', './checkpoints')
        self.onnx_path = os.path.join(self.checkpoints_path, 'onnx', f"depth_anything_v2_{self.args['encoder']}.onnx")
        self.engine_path = os.path.join(self.checkpoints_path, 'onnx', f"depth_anything_v2_{self.args['encoder']}_{self.args['precision']}.engine")
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cpu':
            rospy.logwarn("CUDA not available, running on CPU. This will be very slow.")

        self.model = self.load_model()

    def load_model(self):
        """根據設定載入 PyTorch、ONNX 或 TensorRT 模型。"""
        model_name = self.args['encoder']
        
        # 檢查 PyTorch 權重檔案是否存在
        pytorch_model_path = os.path.join(self.checkpoints_path, f'torch/depth_anything_v2_{model_name}.pth')
        if not os.path.exists(pytorch_model_path):
            rospy.logerr(f"PyTorch model file not found at: {pytorch_model_path}")
            rospy.logerr("Please download the model weights and place them in the 'checkpoints' directory.")
            sys.exit(1)

        # 載入基礎 PyTorch 模型 (用於導出或直接推論)
        depth_anything = DepthAnythingV2(**self.model_configs[model_name])
        depth_anything.load_state_dict(torch.load(pytorch_model_path, map_location=self.device), strict=True)
        depth_anything = depth_anything.to(self.device).eval()

        # 如果啟用 TRT，則載入或建立 TRT 引擎
        if self.args['use_trt']:
            if trt is None:
                rospy.logerr("TensorRT is enabled in args, but the library failed to import.")
                sys.exit(1)

            # 確保 onnx 資料夾存在
            os.makedirs(os.path.dirname(self.onnx_path), exist_ok=True)

            # 檢查 ONNX 模型是否存在，若否，則導出
            if not os.path.exists(self.onnx_path):
                rospy.loginfo(f"ONNX model not found. Exporting from PyTorch to {self.onnx_path}...")
                self.export_model_to_onnx(depth_anything, self.onnx_path)
            else:
                rospy.loginfo(f"✅ ONNX model found at: {self.onnx_path}")

            # 檢查 TensorRT 引擎是否存在，若否，則建立
            if not os.path.exists(self.engine_path):
                rospy.loginfo(f"TensorRT engine not found. Building engine at {self.engine_path}...")
                self.build_trt_engine()
            else:
                rospy.loginfo(f"✅ TensorRT engine found at: {self.engine_path}")

            # 載入 ONNX Runtime Session (作為備用或直接使用)
            rospy.loginfo(f"Loading ONNX Runtime session for {model_name}...")
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            onnx_session = ort.InferenceSession(self.onnx_path, providers=providers)
            rospy.loginfo(f"✅ Successfully loaded ONNX model.")
            return onnx_session
        else:
            rospy.loginfo(f"✅ Successfully loaded PyTorch model for {model_name}. TRT is disabled.")
            return depth_anything

    def export_model_to_onnx(self, model, onnx_path):
        """將 PyTorch 模型導出為 ONNX 格式。"""
        rospy.loginfo("Starting ONNX export...")
        dummy_input = torch.randn(1, 3, self.args['height'], self.args['width'], device=self.device)
        try:
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                opset_version=11,
                input_names=["input"],
                output_names=["output"],
                do_constant_folding=True,
                verbose=False
            )
            rospy.loginfo(f"✅ Model successfully exported to {onnx_path}")
        except Exception as e:
            rospy.logerr(f"Failed to export ONNX model: {e}")
            sys.exit(1)

    def predictions(self, frame_rgb):
        """
        對單一影像幀進行深度預測。
        """
        p_start = time.time()
        
        # 根據模型類型進行推論
        if isinstance(self.model, ort.InferenceSession):
            # ONNX/TensorRT 推論
            img = frame_rgb.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))
            img = np.expand_dims(img, axis=0)
            
            ort_inputs = {self.model.get_inputs()[0].name: img}
            ort_outs = self.model.run(None, ort_inputs)
            
            depth = ort_outs[0].squeeze() # 去掉批次和通道維度
        else:
            # PyTorch 推論
            depth = self.model.infer_image(frame_rgb, input_size=(self.args['height'], self.args['width']))
        
        p_end = time.time()
        rospy.logdebug(f"[Prediction] elapsed: {(p_end - p_start) * 1000:.2f} ms")

        return depth
    
    def build_trt_engine(self):
        """從 ONNX 檔案建立 TensorRT 引擎。"""
        if trt is None:
            rospy.logerr("Cannot build TRT engine, TensorRT library not found.")
            return

        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(network_flags)
        parser = trt.OnnxParser(network, logger)

        rospy.loginfo(f"Parsing ONNX model from {self.onnx_path}...")
        with open(self.onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    rospy.logerr(f"TRT ONNX Parser Error: {parser.get_error(error)}")
                sys.exit(1)
        rospy.loginfo("✅ ONNX model parsed successfully.")

        config = builder.create_builder_config()
        workspace_bytes = self.args['workspace'] * (1 << 30)
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
        
        if self.args['precision'] == "fp16":
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                rospy.loginfo("⚡ Building TensorRT engine in FP16 mode.")
            else:
                rospy.logwarn("Platform does not support FP16, falling back to FP32.")
        else:
            rospy.loginfo("⚡ Building TensorRT engine in FP32 mode.")
            
        rospy.loginfo("Building serialized network. This may take a few minutes...")
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            rospy.logerr("Failed to build serialized TensorRT engine.")
            sys.exit(1)
        
        with open(self.engine_path, 'wb') as f:
            f.write(serialized_engine)
        rospy.loginfo(f"✅ TensorRT engine saved to: {self.engine_path}")


# --- ROS 節點類別 ---
class DepthAnythingROS:
    def __init__(self):
        """初始化 ROS 節點、參數、訂閱者和發布者。"""
        rospy.init_node('depth_anything_node', anonymous=True)
        
        # 從 ROS 參數伺服器讀取設定
        self.get_ros_params()
        
        # 初始化核心深度估計模型
        rospy.loginfo("Initializing Depth Anything model...")
        self.depth_estimator = DepthAnything(self.args)
        
        # 初始化 ROS 影像轉換工具
        self.bridge = CvBridge()
        
        # 建立訂閱者和發布者
        # 增加 buff_size 以避免因影像過大而丟失訊息
        self.image_sub = rospy.Subscriber(self.args['input_topic'], Image, self.image_callback, queue_size=1, buff_size=2**24)
        self.result_pub = rospy.Publisher(self.args['output_topic'], Image, queue_size=1)
        
        rospy.loginfo("✅ Depth Anything ROS node initialized and waiting for images...")
        rospy.loginfo(f"Subscribing to topic: {self.args['input_topic']}")
        rospy.loginfo(f"Publishing to topic: {self.args['output_topic']}")

    def get_ros_params(self):
        """從 ROS 參數伺服器讀取設定，並提供預設值。"""
        self.args = {
            'input_topic': rospy.get_param('~input_topic', '/camera/image_raw'),
            'output_topic': rospy.get_param('~output_topic', '/depth_anything/result_image'),
            'checkpoints_path': rospy.get_param('~checkpoints_path', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')),
            'encoder': rospy.get_param('~encoder', 'vits'),
            'width': rospy.get_param('~width', 518),
            'height': rospy.get_param('~height', 518),
            'use_trt': rospy.get_param('~use_trt', True),
            'precision': rospy.get_param('~precision', 'fp16'),
            'workspace': rospy.get_param('~workspace', 4)
        }
        rospy.loginfo("--- ROS Parameters ---")
        for key, value in self.args.items():
            rospy.loginfo(f"{key}: {value}")
        rospy.loginfo("--------------------")

    def image_callback(self, ros_image):
        """
        處理傳入影像的 ROS 回呼函式。
        """
        rospy.logdebug("Received an image.")
        try:
            # 將 ROS Image 訊息轉換為 OpenCV 影像 (BGR 格式)
            cv_image = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return

        run_start = time.time()
        
        # --- 預處理 ---
        # 調整影像尺寸以符合模型輸入要求
        frame_resized = cv2.resize(cv_image, (self.args['width'], self.args['height']), interpolation=cv2.INTER_AREA)
        # 將 BGR 影像轉換為 RGB
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        # --- 進行深度預測 ---
        depth = self.depth_estimator.predictions(frame_rgb)
        
        # --- 視覺化 ---
        # 將深度圖正規化到 0-255 範圍
        depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # 應用色彩映射 (COLORMAP_MAGMA 看起來效果不錯)
        depth_colormap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_MAGMA)
        
        # 將原始影像(調整尺寸後)和深度圖水平拼接
        result_image = np.hstack((frame_resized, depth_colormap))

        # --- 發布結果 ---
        try:
            # 在影像上繪製 FPS
            latency = (time.time() - run_start)
            fps = 1.0 / latency if latency > 0 else 0
            cv2.putText(result_image, f"FPS: {fps:.2f}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # 將結果影像轉換回 ROS Image 訊息並發布
            result_ros_image = self.bridge.cv2_to_imgmsg(result_image, "bgr8")
            # 保持與輸入影像相同的時間戳和座標系
            result_ros_image.header = ros_image.header
            self.result_pub.publish(result_ros_image)
            
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error on publishing: {e}")

    def run(self):
        """啟動 ROS 節點的主循環。"""
        rospy.spin()

if __name__ == '__main__':
    try:
        ros_node = DepthAnythingROS()
        ros_node.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS node interrupted.")
    except Exception as e:
        rospy.logfatal(f"An unhandled exception occurred: {e}")

