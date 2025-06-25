#!/usr/bin/env python3
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
    from Depth_Anything_V2.depth_anything_v2.dpt import DepthAnythingV2
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
        
        self.checkpoints_path = self.args['checkpoints_path']
        self.onnx_path = os.path.join(self.checkpoints_path, 'onnx', f"depth_anything_v2_{self.args['encoder']}.onnx")
        self.engine_path = os.path.join(self.checkpoints_path, 'onnx', f"depth_anything_v2_{self.args['encoder']}_{self.args['precision']}.engine")
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cpu':
            self.logger.warn("CUDA not available, running on CPU. This will be very slow.")

        self.model = self.load_model()

    def load_model(self):
        model_name = self.args['encoder']
        pytorch_model_path = os.path.join(self.checkpoints_path, f'torch/depth_anything_v2_{model_name}.pth')
        
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

            os.makedirs(os.path.dirname(self.onnx_path), exist_ok=True)
            if not os.path.exists(self.onnx_path):
                self.logger.info(f"ONNX model not found. Exporting from PyTorch to {self.onnx_path}...")
                self.export_model_to_onnx(depth_anything, self.onnx_path)
            
            if not os.path.exists(self.engine_path):
                self.logger.info(f"TensorRT engine not found. Building engine at {self.engine_path}...")
                print("AAAAAAAAAAAAAAAAAAAAAAAAA")
                self.build_trt_engine()
            
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
            depth = self.model.infer_image(frame_rgb, input_size=(self.args['height'], self.args['width']))
        
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
        self.get_ros_params()
        self.logger = self.get_logger()
        
        self.logger.info("Initializing Depth Anything model...")
        self.depth_estimator = DepthAnything(self.args, self.logger)
        
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, self.args['input_topic'], self.image_callback, 1)
        self.result_pub = self.create_publisher(Image, self.args['output_topic'], 1)
        
        self.logger.info("✅ Depth Anything ROS2 node initialized and waiting for images...")
        self.logger.info(f"Subscribing to topic: {self.args['input_topic']}")
        self.logger.info(f"Publishing to topic: {self.args['output_topic']}")

    def get_ros_params(self):
        default_checkpoints_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')
        print("default_checkpoints_path: ", default_checkpoints_path)
        
        self.declare_parameter('input_topic', '/camera/image_raw')
        self.declare_parameter('output_topic', '/depth_anything/result_image')
        self.declare_parameter('checkpoints_path', default_checkpoints_path)
        self.declare_parameter('encoder', 'vits')
        self.declare_parameter('width', 518)
        self.declare_parameter('height', 518)
        self.declare_parameter('use_trt', True)
        self.declare_parameter('precision', 'fp16')
        self.declare_parameter('workspace', 4) # GB

        self.args = {
            'input_topic': self.get_parameter('input_topic').value,
            'output_topic': self.get_parameter('output_topic').value,
            'checkpoints_path': self.get_parameter('checkpoints_path').value,
            'encoder': self.get_parameter('encoder').value,
            'width': self.get_parameter('width').value,
            'height': self.get_parameter('height').value,
            'use_trt': self.get_parameter('use_trt').value,
            'precision': self.get_parameter('precision').value,
            'workspace': self.get_parameter('workspace').value
        }
        self.get_logger().info("--- ROS2 Parameters ---")
        for key, value in self.args.items():
            self.get_logger().info(f"{key}: {value}")
        self.get_logger().info("--------------------")

    def image_callback(self, ros_image: Image):
        self.logger.debug("Received an image.")
        try:
            cv_image = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")
        except CvBridgeError as e:
            self.logger.error(f"CvBridge Error: {e}")
            return

        run_start = time.time()
        
        frame_resized = cv2.resize(cv_image, (self.args['width'], self.args['height']), interpolation=cv2.INTER_AREA)
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        
        depth = self.depth_estimator.predictions(frame_rgb)
        
        depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_colormap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_MAGMA)
        
        result_image = np.hstack((frame_resized, depth_colormap))
        
        try:
            latency = time.time() - run_start
            fps = 1.0 / latency if latency > 0 else 0
            cv2.putText(result_image, f"FPS: {fps:.2f}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            result_ros_image = self.bridge.cv2_to_imgmsg(result_image, "bgr8")
            result_ros_image.header = ros_image.header
            self.result_pub.publish(result_ros_image)
            
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
