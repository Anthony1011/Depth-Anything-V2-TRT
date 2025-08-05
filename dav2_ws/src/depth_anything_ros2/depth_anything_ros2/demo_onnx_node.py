#!/usr/bin/env /home/nvidia/miniconda3/envs/yolov8/bin/python
# -*- coding: utf-8 -*-


import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from rclpy.logging import get_logger

import numpy as np
import os
import torch
import cv2
import time
import onnxruntime as ort
import sys
import traceback

try:
    import tensorrt as trt
except ImportError:
    get_logger("imports").warn("TensorRT library not found. TensorRT functionalities will be disabled.")
    trt = None

try:
    from depth_anything_ros2.depth_anything_v2_tool.dpt import DepthAnythingV2
except ImportError:
    get_logger("imports").fatal("Failed to import DepthAnythingV2. Make sure the 'depth_anything_v2' directory is accessible.")
    sys.exit(1)

# --- 核心深度估計類別 (ROS2 適應版) ---
class DepthAnything:
    def __init__(self, args: dict, logger) -> None:
        self.args = args
        self.logger = logger
        self.model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        
        self.root_weights = self.args['root_weights']
        self.onnx_path = os.path.join(self.root_weights, 'onnx', f"depth_anything_v2_{self.args['encoder']}.onnx")
        self.engine_path = os.path.join(self.root_weights, 'onnx', f"depth_anything_v2_{self.args['encoder']}_{self.args['precision']}.engine")
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cpu':
            self.logger.warn("CUDA not available, running on CPU. This will be very slow.")

        self.model = self.load_model()

    def load_model(self):
        model_name = self.args['encoder']
        pytorch_model_path = os.path.join(self.root_weights, f'torch/depth_anything_v2_{model_name}.pth')
        
        if not os.path.exists(pytorch_model_path):
            self.logger.fatal(f"PyTorch model file not found at: {pytorch_model_path}")
            sys.exit(1)

        depth_anything = DepthAnythingV2(**self.model_configs[model_name])
        depth_anything.load_state_dict(torch.load(pytorch_model_path, map_location=self.device), strict=True)
        depth_anything = depth_anything.to(self.device).eval()

        if self.args['use_trt']:
            if trt is None:
                self.logger.error("TensorRT is enabled in args, but the library failed to import.")
                sys.exit(1)

            # 檢查是否有onnx權重 沒有則壓縮
            os.makedirs(os.path.dirname(self.onnx_path), exist_ok=True)
            if not os.path.exists(self.onnx_path):
                self.logger.info(f"ONNX model not found. Exporting from PyTorch to {self.onnx_path}...")
                self.export_model_to_onnx(depth_anything, self.onnx_path)
            
            # 檢查是否有engine權重 沒有則壓縮
            if not os.path.exists(self.engine_path):
                self.logger.info(f"TensorRT engine not found. Building engine at {self.engine_path}...")
                print("AAAAAAAAAAAAAAAAAAAAAAAAA")
                self.build_trt_engine()
            
            # 這裡紙漿onnx 的模型實例化
            self.logger.info(f"Loading ONNX Runtime session for {model_name}...")
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            return ort.InferenceSession(self.onnx_path, providers=providers)
        else:
            self.logger.info(f"✅ Successfully loaded PyTorch model for {model_name}. TRT is disabled.")
            return depth_anything

    def export_model_to_onnx(self, model, onnx_path):
        self.logger.info("Starting ONNX export...")
        dummy_input = torch.randn(1, 3, self.args['height'], self.args['width'], device=self.device)
        try:
            torch.onnx.export(model, dummy_input, onnx_path, opset_version=11,
                              input_names=["input"], output_names=["output"],
                              do_constant_folding=True, verbose=False)
            self.logger.info(f"✅ Model successfully exported to {onnx_path}")
        except Exception as e:
            self.logger.error(f"Failed to export ONNX model: {e}")
            sys.exit(1)

    def predictions(self, frame_rgb):
        p_start = time.time()
        if isinstance(self.model, ort.InferenceSession):
            img = frame_rgb.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))
            img = np.expand_dims(img, axis=0)
            ort_inputs = {self.model.get_inputs()[0].name: img}
            ort_outs = self.model.run(None, ort_inputs)
            depth = ort_outs[0].squeeze()
        else:
            # print(type(self.args['height']), type((self.args['height'],self.args['width'])))
            depth = self.model.infer_image(frame_rgb, input_size = 518)
        
        self.logger.debug(f"[Prediction] elapsed: {(time.time() - p_start) * 1000:.2f} ms")
        return depth
    
    def build_trt_engine(self):
        if trt is None: return
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        
        self.logger.info(f"Parsing ONNX model from {self.onnx_path}...")
        with open(self.onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    self.logger.error(f"TRT ONNX Parser Error: {parser.get_error(error)}")
                sys.exit(1)
        
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, self.args['workspace'] * (1 << 30))
        if self.args['precision'] == "fp16" and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            self.logger.info("⚡ Building TensorRT engine in FP16 mode.")
        
        serialized_engine = builder.build_serialized_network(network, config)
        if not serialized_engine:
            self.logger.error("Failed to build serialized TensorRT engine.")
            sys.exit(1)
        
        with open(self.engine_path, 'wb') as f: f.write(serialized_engine)
        self.logger.info(f"✅ TensorRT engine saved to: {self.engine_path}")

# --- ROS2 節點類別 ---
class DepthAnythingROS2Node(Node):
    def __init__(self):
        super().__init__('depth_anything_onnx_ros2_node')

        self.logger = self.get_logger()
        self.bridge = CvBridge()

        self.declare_parameter('input_image','/zedx_one_0/image')

        self.declare_parameter('depth_np','/zedx_0_depth_np_image')
        self.declare_parameter('depth_colormap','/zedx_0_depth_colormap_image')
        self.declare_parameter('pred_depth_colormap','/zedx_0_pred_depth_colormap_image')

        self.declare_parameter('root_weights','/home/nvidia/hard_disk/dav2_ws/src/depth_anything_ros2/checkpoints')
        self.declare_parameter('encoder','vits')
        self.declare_parameter('width',518)
        self.declare_parameter('height',518)

        self.declare_parameter('max_depth',120.0)
        self.declare_parameter('min_depth',0.1)

        self.declare_parameter('use_trt',False)
        self.declare_parameter('precision','fp32')
        self.declare_parameter('workspace',4)

        input_image = self.get_parameter('input_image').get_parameter_value().string_value

        depth_np = self.get_parameter('depth_np').get_parameter_value().string_value
        depth_colormap = self.get_parameter('depth_colormap').get_parameter_value().string_value
        pred_depth_colormap = self.get_parameter('pred_depth_colormap').get_parameter_value().string_value

        root_weights = self.get_parameter('root_weights').get_parameter_value().string_value
        encoder = self.get_parameter('encoder').get_parameter_value().string_value
        width = self.get_parameter('width').get_parameter_value().integer_value
        height = self.get_parameter('height').get_parameter_value().integer_value
        max_depth = self.get_parameter('max_depth').get_parameter_value().double_value
        min_depth = self.get_parameter('min_depth').get_parameter_value().double_value
        use_trt = self.get_parameter('use_trt').get_parameter_value().bool_value
        precision = self.get_parameter('precision').get_parameter_value().string_value
        workspace = self.get_parameter('workspace').get_parameter_value().integer_value

        self.args = {
            'input_image': input_image,
            'depth_np': depth_np,
            'depth_colormap': depth_colormap,
            'pred_depth_colormap': pred_depth_colormap,
            'root_weights': root_weights,
            'encoder': encoder,
            'width': width,
            'height': height,
            'max_depth': max_depth,
            'min_depth': min_depth,
            'use_trt': use_trt,
            'precision': precision,
            'workspace': workspace
        }

        self.get_logger().info("--- ROS2 Parameters ---")
        for key, value in self.args.items():
            self.get_logger().info(f"{key}: {value}")
        self.get_logger().info("--------------------")

        self.logger.info("Initializing Depth Anything model...")
        # 建立Depth AnthingV2 Model
        self.depth_estimator = DepthAnything(self.args, self.logger)
        
        self.image_sub = self.create_subscription(Image, self.args['input_image'], self.image_callback, 1)
        
        self.depth_np = self.create_publisher(Image, self.args['depth_np'], 1)
        self.depth_colormap = self.create_publisher(Image, self.args['depth_colormap'], 1)
        self.pred_depth_colormap = self.create_publisher(Image, self.args['pred_depth_colormap'], 1)

        self.logger.info("✅ Depth Anything ROS2 node initialized and waiting for images...")
        self.logger.info(f"Subscribing to topic: {self.args['input_image']}")
        self.logger.info(f"Publishing to topic: {self.args['depth_np']}, {self.args['depth_colormap']}, {self.args['pred_depth_colormap']}")

    def disp_to_depth(self, disp, min_depth, max_depth):
        """Convert network's sigmoid output into depth prediction
        The formula for this conversion is given in the 'additional considerations'
        section of the paper.
        """
        min_disp = 1 / max_depth
        max_disp = 1 / min_depth
        scaled_disp = min_disp + (max_disp - min_disp) * disp
        depth = 1 / scaled_disp
        return scaled_disp, depth

    def image_callback(self, ros_image: Image):

        self.logger.debug("Received an image.")
        header = ros_image.header
        try:
            cv_image = self.bridge.imgmsg_to_cv2(ros_image, desired_encoding="bgr8")
        except CvBridgeError as e:
            self.logger.error(f"CvBridge Error: {e}")
            return

        
        frame_resized = cv2.resize(cv_image, (self.args['width'], self.args['height']), interpolation=cv2.INTER_AREA)
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        inference_start_time = time.time()
        disp = self.depth_estimator.predictions(frame_rgb)
        scaled_disp, depth = self.disp_to_depth(disp, self.args['min_depth'], self.args['max_depth'])
        inference_ms = (time.time() - inference_start_time) * 1000

        depth *= 1.0

        try:
            inference_fps = 1000.0 / inference_ms if inference_ms > 0 else 0
            self.get_logger().info(f"Inference FPS: {inference_fps:.2f} | Latency: {inference_ms:.2f} ms")

            # 轉 numpy 與 flaot point 32
            depth_np = depth.astype(np.float32)
            scaled_disp_np = scaled_disp.astype(np.float32)
            disp_resized_np = disp.astype(np.float32)

            # 0-255 gracy depth image
            pred_depth = (disp_resized_np - np.min(disp_resized_np)) / (np.max(disp_resized_np) - np.min(disp_resized_np)) * 255
            pred_depth = pred_depth.astype(np.uint8)

            # Color depth image
            depth_colormap = cv2.applyColorMap(pred_depth, cv2.COLORMAP_MAGMA)

            cv2.putText(depth_colormap, f"FPS: {inference_fps:.2f}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            depth_np_msg = self.bridge.cv2_to_imgmsg(depth_np, encoding="32FC1")
            depth_np_msg.header = header
            pred_depth_msg = self.bridge.cv2_to_imgmsg(pred_depth, encoding="mono8")
            pred_depth_msg.header = header
            depth_colormap_msg = self.bridge.cv2_to_imgmsg(depth_colormap, encoding="bgr8")
            depth_colormap_msg.header = header

            self.depth_np.publish(depth_np_msg)
            self.depth_colormap.publish(pred_depth_msg) 
            self.pred_depth_colormap.publish(depth_colormap_msg) 
            
        except CvBridgeError as e:
            self.logger.error(f"CvBridge Error on publishing: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = DepthAnythingROS2Node()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        get_logger("main").fatal(f"An unhandled exception occurred: {e}\n{traceback.format_exc()}")
    finally:
        if node:
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
