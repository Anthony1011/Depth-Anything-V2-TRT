#!/usr/bin/env /home/nvidia/miniconda3/envs/yolov8/bin/python
# -*- coding: utf-8 -*-

# ROS 2 Core Libraries
import rclpy
import rclpy.logging
from rclpy.node import Node
from rclpy.qos import QoSProfile

# ROS 2 Message Types
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

# Standard Libraries
import numpy as np
import os
import cv2
import time
import tensorrt as trt
import traceback

# PyCUDA
import pycuda.driver as cuda

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# =================================================================================
#  DepthAnythingTRT 類別 (此部分與前一版本相同，保持不變)
# =================================================================================
class DepthAnythingTRT():
    def __init__(self, encoder='vitb', precision='fp16', root_weights='./checkpoints', logger=None) -> None:
        self.logger = logger if logger is not None else rclpy.logging.get_logger("DepthAnythingTRT_standalone")
        self.logger.info("Initializing DepthAnythingTRT Engine...")
        self.encoder = encoder
        self.precision = precision
        self.root_weights = root_weights
        self.engine_path = os.path.join(self.root_weights, "onnx", f"depth_anything_v2_{self.encoder}_{self.precision}.engine")
        self.engine = self.load_engine()
        if not self.engine:
            raise RuntimeError("Failed to load TensorRT engine.")
        self.context = self.engine.create_execution_context()
        self.logger.info(f"TensorRT Version {trt.__version__}")

        if hasattr(self.context, 'execute_async_v3'):
            self.execution_api_type = 'v3_async'
        elif hasattr(self.context, 'execute_async_v2'):
            self.execution_api_type = 'v2_async'
        elif hasattr(self.context, 'execute_v2'):
            self.execution_api_type = 'v2_sync'
        else:
            if self.context: del self.context
            if self.engine: del self.engine
            raise RuntimeError("Unsupported TensorRT execution method found.")
        
        self.logger.info(f"Using TensorRT execution API: {self.execution_api_type}")
        self.host_inputs, self.host_outputs = [], []
        self.device_inputs, self.device_outputs = [], []
        self.bindings = []
        self.stream = cuda.Stream()
        self.input_shapes, self.input_dtypes, self.output_shapes = {}, {}, {}

        self.logger.info("--- Initializing Engine Bindings (Inputs/Outputs) ---")
        for binding in self.engine:
            shape = tuple(self.engine.get_tensor_shape(binding))
            dtype = trt.nptype(self.engine.get_tensor_dtype(binding))
            if -1 in shape:
                self.logger.warn(f"Binding '{binding}' has dynamic dimensions: {shape}")
            max_shape = list(shape)
            for i, dim in enumerate(max_shape):
                if dim == -1:
                    max_shape[i] = 518
            num_elements = trt.volume(max_shape)
            host_mem = cuda.pagelocked_empty(num_elements, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            tensor_io_mode = self.engine.get_tensor_mode(binding)
            if tensor_io_mode == trt.TensorIOMode.INPUT:
                self.host_inputs.append(host_mem)
                self.device_inputs.append(device_mem)
                self.input_shapes[binding] = shape
                self.input_dtypes[binding] = dtype
            elif tensor_io_mode == trt.TensorIOMode.OUTPUT:
                self.host_outputs.append(host_mem)
                self.device_outputs.append(device_mem)
                self.output_shapes[binding] = shape
        self.primary_input_name = next(name for name in self.engine if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT)
        self.primary_output_name = next(name for name in self.engine if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT)
        
        self.logger.info("--- Bindings initialized ---")
        self.logger.info(f"✅ TensorRT engine successfully loaded and initialized from '{self.engine_path}'.")

    def load_engine(self):
        if not os.path.exists(self.engine_path):
            self.logger.error(f"Engine file not found: {self.engine_path}")
            return None
        self.logger.info(f"Loading TensorRT engine from {self.engine_path}...")
        with open(self.engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            self.logger.error("Failed to deserialize the engine.")
            return None
        self.logger.info("Engine loaded successfully.")
        return engine

    def predictions(self, frame_rgb: np.ndarray):
        if not self.context:
            self.logger.error("TensorRT context is not initialized in predictions.")
            return None
        input_name = self.primary_input_name
        output_name = self.primary_output_name
        dtype = self.input_dtypes[input_name]
        H_in, W_in = 518, 518
        img = cv2.resize(frame_rgb, (W_in, H_in), interpolation=cv2.INTER_AREA) if frame_rgb.shape[:2] != (H_in, W_in) else frame_rgb
        img = img.astype(dtype) / 255.0
        img = img.transpose((2, 0, 1))
        img = np.expand_dims(img, axis=0)
        img = np.ascontiguousarray(img)
        host_input_buffer = self.host_inputs[0]
        np.copyto(host_input_buffer, img.ravel())
        try:
            cuda.memcpy_htod_async(self.device_inputs[0], host_input_buffer, self.stream)
            if self.execution_api_type == 'v3_async':
                if not self.context.set_input_shape(input_name, img.shape):
                    self.logger.error(f"Failed to set input shape to {img.shape} for binding '{input_name}'")
                    return None
                self.context.set_tensor_address(input_name, int(self.device_inputs[0]))
                self.context.set_tensor_address(output_name, int(self.device_outputs[0]))
                if not self.context.execute_async_v3(stream_handle=self.stream.handle):
                    self.logger.error("execute_async_v3() returned False.")
                    return None
            else:
                self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            output_shape = self.context.get_tensor_shape(output_name)
            output_size_in_elements = trt.volume(output_shape)
            host_output_buffer = self.host_outputs[0]
            cuda.memcpy_dtoh_async(host_output_buffer.ravel()[:output_size_in_elements], self.device_outputs[0], self.stream)
            self.stream.synchronize()
        except Exception as e:
            self.logger.error(f"TensorRT inference failed (API: {self.execution_api_type}): {e}")
            traceback.print_exc()
            return None
        depths = host_output_buffer[:output_size_in_elements].reshape(output_shape)
        return depths.squeeze()

# =================================================================================
#  ROS 2 節點包裝類別 (FPS 依據推論延遲計算)
# =================================================================================
class DepthAnythingROS2(Node):
    def __init__(self):
        super().__init__('depth_anything_ros2_node')
        self.get_logger().info("Starting Depth Anything ROS 2 Node...")

        self.cuda_ctx = None
        self.init_cuda()
        self.cuda_ctx.push()
        
        # 設定接收launch parameter
        self.declare_parameter('input_image','/zedx_one_0/image')

        self.declare_parameter('depth_np','/zedx_0_depth_np_image')
        self.declare_parameter('depth_colormap','/zedx_0_depth_colormap_image')
        self.declare_parameter('pred_depth_colormap','/zedx_0_pred_depth_colormap_image')

        self.declare_parameter('max_depth',120.0)
        self.declare_parameter('min_depth',0.1)

        self.declare_parameter('root_weights','/home/nvidia/hard_disk/dav2_ws/src/depth_anything_ros2/checkpoints')
        self.declare_parameter('encoder','vits')
        self.declare_parameter('precision','fp32')

        # 定義取得從parameter 去得 /topic name
        self.input_image = self.get_parameter('input_image').get_parameter_value().string_value

        self.depth_np = self.get_parameter('depth_np').get_parameter_value().string_value
        self.depth_colormap = self.get_parameter('depth_colormap').get_parameter_value().string_value
        self.pred_depth_colormap = self.get_parameter('pred_depth_colormap').get_parameter_value().string_value

        self.max_depth = self.get_parameter('max_depth').get_parameter_value().double_value
        self.min_depth = self.get_parameter('min_depth').get_parameter_value().double_value

        root_weights = self.get_parameter('root_weights').get_parameter_value().string_value
        encoder = self.get_parameter('encoder').get_parameter_value().string_value
        precision = self.get_parameter('precision').get_parameter_value().string_value

        self.bridge = CvBridge()
        self.model = DepthAnythingTRT(encoder, precision, root_weights, logger=self.get_logger())

        qos_profile = QoSProfile(depth=1)

        self.depth_np_pub = self.create_publisher(Image, self.depth_np, 1)
        self.depth_colormap_pub = self.create_publisher(Image, self.depth_colormap, 1)
        self.pred_depth_colormap_pub = self.create_publisher(Image, self.pred_depth_colormap, 1)
        
        self.image_sub = self.create_subscription(Image, self.input_image, self.image_callback, 10)

        self.get_logger().info("Depth Anything ROS 2 node is ready.")
        self.get_logger().info(f"Subscribing to '{self.input_image}'")
        self.get_logger().info(f"Publishing to: {self.depth_np}, {self.depth_colormap}, {self.pred_depth_colormap}")

    def init_cuda(self):
        try:
            self.get_logger().info("Initializing CUDA...")
            cuda.init()
            device = cuda.Device(0)
            self.cuda_ctx = device.make_context()
            self.get_logger().info(f"CUDA initialized on device {device.name()}.")
        except Exception as e:
            self.get_logger().fatal(f"Failed to initialize CUDA: {e}")
            raise e

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

        self.get_logger().info("Received an image.")
        header = ros_image.header
        try:
            self.get_logger().debug("Received an image.")
            try:
                cv_image = self.bridge.imgmsg_to_cv2(ros_image, desired_encoding="bgr8")

            except CvBridgeError as e:
                self.get_logger().error(f"CvBridge Error: {e}")
                return

            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            
            inference_start_time = time.time()
            disp = self.model.predictions(rgb_image)
            scaled_disp, depth_map = self.disp_to_depth(disp, self.min_depth, self.max_depth)
            inference_ms = (time.time() - inference_start_time) * 1000.0

            depth_map *= 1.0

            if disp is not None:
                inference_fps = 1000.0 / inference_ms if inference_ms > 0 else 0
                self.get_logger().info(f"Inference FPS: {inference_fps:.2f} ms | Latency: {inference_ms:.2f} ms")

                depth_np = depth_map.astype(np.float32)
                scaled_disp_np = scaled_disp.astype(np.float32)
                disp_np = disp.astype(np.float32)
            
                depth_map = depth_map.astype(np.float32)

                # 0-255 gracy depth image
                pred_depth = cv2.normalize(disp_np, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                pred_depth = pred_depth.astype(np.uint8)

                # Color depth image
                depth_colormap = cv2.applyColorMap(pred_depth, cv2.COLORMAP_MAGMA)
                cv2.putText(depth_colormap, f"FPS: {inference_fps:.2f}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)


                try:
                    depth_np = self.bridge.cv2_to_imgmsg(depth_np, encoding="32FC1")
                    depth_np.header = header

                    depth_colormap = self.bridge.cv2_to_imgmsg(depth_colormap, encoding="bgr8")
                    depth_colormap.header = header

                    pred_depth = self.bridge.cv2_to_imgmsg(pred_depth, encoding="mono8")
                    pred_depth.header = header
                except CvBridgeError as e:
                    self.get_logger().error(f"CvBridge Error (Raw Depth): {e}")

                try:
                    self.depth_np_pub.publish(depth_np)
                    self.depth_colormap_pub.publish(depth_colormap)
                    self.pred_depth_colormap_pub.publish(pred_depth)
    
                except Exception as e:
                    self.get_logger().error(f"Failed to create or publish stitched image: {e}")

        except Exception as e:
            self.get_logger().error(f"An error occurred in image_callback: {e}")
            traceback.print_exc()

    def cleanup(self):
        self.get_logger().info("Cleaning up resources...")
        if self.cuda_ctx:
            self.cuda_ctx.pop()
            self.cuda_ctx.detach()
            self.cuda_ctx = None
            self.get_logger().info("CUDA context cleaned up.")

# =================================================================================
#  主執行函數 (ROS 2 模式)
# =================================================================================
def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = DepthAnythingROS2()
        rclpy.spin(node)
    except (KeyboardInterrupt, Exception) as e:
        if node:
            node.get_logger().error(f"An exception occurred: {e}")
            traceback.print_exc()
    finally:
        if node:
            node.cleanup()
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()