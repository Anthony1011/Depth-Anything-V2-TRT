#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import numpy as np
import os
import cv2
import time
import tensorrt as trt
import traceback

# ---【關鍵修改 1】---
# 移除 pycuda.autoinit，我們將手動管理 context
# import pycuda.autoinit
import pycuda.driver as cuda

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# =================================================================================
#  DepthAnythingTRT 類別 (保持我們修正後的版本)
# =================================================================================
class DepthAnythingTRT():
    # In the class DepthAnythingTRT:

    def __init__(self, encoder='vitb', precision='fp16', root_weights='./checkpoints') -> None:
        rospy.loginfo("Initializing DepthAnythingTRT Engine...")
        self.encoder = encoder
        self.precision = precision
        self.root_weights = root_weights
        self.engine_path = os.path.join(self.root_weights, "onnx", f"depth_anything_v2_{self.encoder}_{self.precision}.engine")

        self.engine = self.load_engine()
        if not self.engine:
            rospy.logerr(f"Failed to load TensorRT engine. Shutting down node.")
            rospy.signal_shutdown("Engine load failure")
            return

        self.context = self.engine.create_execution_context()
        rospy.loginfo(f"TensorRT Version {trt.__version__}")

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
        rospy.loginfo(f"Using TensorRT execution API: {self.execution_api_type}")

        self.host_inputs, self.host_outputs = [], []
        self.device_inputs, self.device_outputs = [], []
        self.bindings = []
        self.stream = cuda.Stream()

        self.input_shapes, self.input_dtypes, self.output_shapes = {}, {}, {}

        rospy.loginfo("--- Initializing Engine Bindings (Inputs/Outputs) ---")
        for binding in self.engine:
            shape = tuple(self.engine.get_tensor_shape(binding))
            dtype = trt.nptype(self.engine.get_tensor_dtype(binding))
            if -1 in shape:
                rospy.logwarn(f"Binding '{binding}' has dynamic dimensions: {shape}")

            # --- 【關鍵修正】---
            # 對於動態尺寸 (-1)，我們需要為其分配一個最大的可能緩衝區。
            # 我們不再手動計算位元組大小，而是採用更安全的模式。
            max_shape = list(shape)
            for i, dim in enumerate(max_shape):
                if dim == -1:
                    # 假設模型的最大輸入/輸出維度是 518。
                    # 如果您的模型有更大的潛在尺寸，請調整此數值。
                    max_shape[i] = 518
            
            # 1. 根據最大元素數量和資料型別分配鎖頁主機記憶體
            num_elements = trt.volume(max_shape)
            host_mem = cuda.pagelocked_empty(num_elements, dtype)
            
            # 2. 使用剛分配好的主機記憶體的位元組大小 (host_mem.nbytes) 來分配設備記憶體
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            # --------------------

            self.bindings.append(int(device_mem))

            tensor_io_mode = self.engine.get_tensor_mode(binding)
            if tensor_io_mode == trt.TensorIOMode.INPUT:
                self.host_inputs.append(host_mem)
                self.device_inputs.append(device_mem)
                self.input_shapes[binding]= shape
                self.input_dtypes[binding] = dtype
            elif tensor_io_mode == trt.TensorIOMode.OUTPUT:
                self.host_outputs.append(host_mem)
                self.device_outputs.append(device_mem)
                self.output_shapes[binding] = shape

        self.primary_input_name = next(name for name in self.engine if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT)
        self.primary_output_name = next(name for name in self.engine if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT)

        rospy.loginfo("--- Bindings initialized ---")
        rospy.loginfo(f"✅ TensorRT engine successfully loaded and initialized from '{self.engine_path}'.")

    def load_engine(self):
        if not os.path.exists(self.engine_path):
            rospy.logerr(f"Engine file not found: {self.engine_path}")
            return None
        rospy.loginfo(f"Loading TensorRT engine from {self.engine_path}...")
        with open(self.engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            rospy.logerr("Failed to deserialize the engine.")
            return None
        rospy.loginfo("Engine loaded successfully.")
        return engine

    def predictions(self, frame_rgb: np.ndarray):
        if not self.context:
            rospy.logerr("TensorRT context is not initialized in predictions.")
            return None
        
        input_name = self.primary_input_name
        output_name = self.primary_output_name
        
        dtype = self.input_dtypes[input_name]

        # 假設模型的標準輸入尺寸為 518x518
        H_in, W_in = 518, 518
        if frame_rgb.shape[0] != H_in or frame_rgb.shape[1] != W_in:
            img = cv2.resize(frame_rgb, (W_in, H_in), interpolation=cv2.INTER_AREA)
        else:
            img = frame_rgb

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
                    rospy.logerr(f"Failed to set input shape to {img.shape} for binding '{input_name}'")
                    return None
                self.context.set_tensor_address(input_name, int(self.device_inputs[0]))
                self.context.set_tensor_address(output_name, int(self.device_outputs[0]))
                if not self.context.execute_async_v3(stream_handle=self.stream.handle):
                    rospy.logerr("execute_async_v3() returned False.")
                    return None
            else: # Fallback for v2 APIs
                self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

            output_shape = self.context.get_tensor_shape(output_name)
            output_size_in_elements = trt.volume(output_shape)
            host_output_buffer = self.host_outputs[0]
            
            cuda.memcpy_dtoh_async(host_output_buffer.ravel()[:output_size_in_elements], self.device_outputs[0], self.stream)
            self.stream.synchronize()

        except Exception as e:
            rospy.logerr(f"TensorRT inference failed (API: {self.execution_api_type}): {e}")
            traceback.print_exc()
            return None
        
        depths = host_output_buffer[:output_size_in_elements].reshape(output_shape)
        depths = depths.squeeze() 

        return depths

# =================================================================================
#  ROS 節點包裝類別 - 增加了拼接影像的發布功能
# =================================================================================
class DepthAnythingROS:
    def __init__(self):
        rospy.init_node('depth_anything_ros_node', anonymous=True)
        rospy.loginfo("Starting Depth Anything ROS Node...")

        self.cuda_ctx = None
        try:
            rospy.loginfo("Initializing CUDA...")
            cuda.init()
            device = cuda.Device(0)
            self.cuda_ctx = device.make_context()
            rospy.loginfo(f"CUDA initialized on device {device.name()}.")
        except Exception as e:
            rospy.logfatal(f"Failed to initialize CUDA: {e}")
            rospy.signal_shutdown("CUDA initialization failed.")
            return

        rospy.on_shutdown(self.cleanup)
        self.cuda_ctx.push()
        try:
            # --- 參數設定 ---
            self._image_topic = rospy.get_param('~image_topic', '/camera/image_raw')
            self._depth_raw_topic = rospy.get_param('~depth_raw_topic', '/depth_anything/depth_raw')
            self._depth_vis_topic = rospy.get_param('~depth_vis_topic', '/depth_anything/depth_color')
            # --- 【新功能】為拼接影像新增 Topic 參數 ---
            self._stitched_image_topic = rospy.get_param('~stitched_image_topic', '/depth_anything/stitched_image')

            encoder = rospy.get_param('~encoder', 'vitb')
            precision = rospy.get_param('~precision', 'fp16')
            root_weights = rospy.get_param('~root_weights', './checkpoints')

            # --- 初始化模型與 Publisher ---
            self.bridge = CvBridge()
            self.model = DepthAnythingTRT(encoder, precision, root_weights)
            if rospy.is_shutdown():
                return

            self.depth_pub = rospy.Publisher(self._depth_raw_topic, Image, queue_size=1)
            self.vis_pub = rospy.Publisher(self._depth_vis_topic, Image, queue_size=1)
            # --- 【新功能】建立新的 Publisher ---
            self.stitched_pub = rospy.Publisher(self._stitched_image_topic, Image, queue_size=1)

            self.image_sub = rospy.Subscriber(self._image_topic, Image, self.image_callback, queue_size=1, buff_size=2**24)

            rospy.loginfo("Depth Anything ROS node is ready.")
            rospy.loginfo(f"Publishing stitched image to '{self._stitched_image_topic}'") # 提示用戶
        finally:
            self.cuda_ctx.pop()


    def image_callback(self, msg: Image):
        self.cuda_ctx.push()
        try:
            rospy.logdebug("Received an image.")
            try:
                # cv_image 是原始的 BGR 格式影像
                cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            except CvBridgeError as e:
                rospy.logerr(f"CvBridge Error: {e}")
                return

            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

            inference_start_time = time.time()
            depth_map = self.model.predictions(rgb_image)
            latency_ms = (time.time() - inference_start_time) * 1000

            if depth_map is not None:
                rospy.loginfo(f"Inference latency: {latency_ms:.2f} ms")
                depth_map = depth_map.astype(np.float32)

                # --- 發布原始深度圖 (這部分不變) ---
                try:
                    depth_msg = self.bridge.cv2_to_imgmsg(depth_map, encoding="32FC1")
                    depth_msg.header = msg.header
                    self.depth_pub.publish(depth_msg)
                except CvBridgeError as e:
                    rospy.logerr(f"CvBridge Error (Raw Depth): {e}")

                # --- 建立視覺化深度圖 ---
                depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                depth_colormap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_MAGMA)

                # --- 發布單獨的視覺化深度圖 (這部分不變) ---
                try:
                    vis_msg = self.bridge.cv2_to_imgmsg(depth_colormap, encoding="bgr8")
                    vis_msg.header = msg.header
                    self.vis_pub.publish(vis_msg)
                except CvBridgeError as e:
                    rospy.logerr(f"CvBridge Error (Visualization): {e}")

                # --- 【新功能】建立並發布拼接影像 ---
                try:
                    # 獲取深度圖的尺寸
                    h, w, _ = depth_colormap.shape
                    # 將原始影像 resize 成與深度圖相同尺寸
                    frame_resized = cv2.resize(cv_image, (w, h))

                    # 將原始影像(調整尺寸後)和深度圖水平拼接
                    result_image = np.hstack((frame_resized, depth_colormap))

                    # 將拼接後的影像轉換為 ROS message 並發布
                    stitched_msg = self.bridge.cv2_to_imgmsg(result_image, "bgr8")
                    stitched_msg.header = msg.header
                    self.stitched_pub.publish(stitched_msg)
                except Exception as e:
                    rospy.logerr(f"Failed to create or publish stitched image: {e}")

        finally:
            self.cuda_ctx.pop()


    def cleanup(self):
        rospy.loginfo("Cleaning up CUDA context...")
        if self.cuda_ctx:
            self.cuda_ctx.detach()
            self.cuda_ctx = None
            rospy.loginfo("CUDA context cleaned up.")

if __name__ == '__main__':
    node = None
    try:
        node = DepthAnythingROS()
        if not rospy.is_shutdown():
            rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS node shutting down.")
    except Exception as e:
        rospy.logfatal(f"An unhandled exception occurred in main: {e}")
        traceback.print_exc()
    finally:
        # 確保即使在初始化失敗時也能嘗試清理
        if node and hasattr(node, 'cleanup') and callable(getattr(node, 'cleanup', None)):
            node.cleanup()