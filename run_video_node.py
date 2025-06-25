#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ROS2 節點：Depth Anything V2 PyTorch 推理節點

1. 初始化 ROS2 節點。
2. 透過 ROS2 參數服務讀取模型設定 (encoder, input_size 等)。
3. 使用 PyTorch 載入 DepthAnythingV2 模型。
4. 訂閱 /camera/image_raw 主題。
5. 將接收到的影像送入模型進行深度預測，並計算推論延遲與 FPS。
6. 發布三個主題：
   a) /ai_depth/image: 原始浮點數深度圖 (32FC1)，供 RVIZ 或其他節點使用。
   b) /ai_depth/image_vis: 可視化的彩色深度圖 (bgr8)，方便人類觀察。
   c) /ai_depth/image_stitched: 原始影像與可視化深度圖拼接後的影像 (bgr8)，並附帶 FPS 資訊。
"""
import os

# 獲取當前工作路徑
current_path = os.getcwd()

# 顯示路徑
print(f"當前的工作路徑是：{current_path}")
import sys
import os
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import torch
import matplotlib
import time # <--- 【新增】導入 time 模組

# ROS 相關導入
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

# 導入 Depth Anything V2 模型
# 請確保 depth_anything_v2 這個 Python 包在您的環境中可以被找到
try:
    from Depth_Anything_V2.depth_anything_v2.dpt import DepthAnythingV2
except ImportError:
    print("錯誤：無法導入 'depth_anything_v2'。請確保已安裝此套件或將其路徑加入 PYTHONPATH。", file=sys.stderr)
    sys.exit(1)


class DepthNodePyTorch(Node):
    def __init__(self):
        """
        節點的建構函式。
        """
        super().__init__('depth_anything_pytorch_ros_node')
        self.get_logger().info("初始化 Depth Anything (PyTorch) ROS2 節點...")

        # 1. 宣告並獲取參數
        self.declare_parameter('encoder', 'vits')
        self.declare_parameter('input_size', 518)
        self.declare_parameter('checkpoint_path', './checkpoints/torch')
        self.declare_parameter('grayscale', False)

        self.encoder = self.get_parameter('encoder').get_parameter_value().string_value
        self.input_size = self.get_parameter('input_size').get_parameter_value().integer_value
        self.checkpoint_path = self.get_parameter('checkpoint_path').get_parameter_value().string_value
        self.grayscale = self.get_parameter('grayscale').get_parameter_value().bool_value

        self.get_logger().info(f"模型參數 - Encoder: '{self.encoder}', Input Size: {self.input_size}, Checkpoints: '{self.checkpoint_path}'")

        # 2. 載入模型
        try:
            self.get_logger().info("正在載入 Depth Anything V2 PyTorch 模型...")
            
            self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.get_logger().info(f"使用設備: {self.DEVICE}")

            model_configs = {
                'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
                'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
            }
            
            self.model = DepthAnythingV2(**model_configs[self.encoder])
            
            model_weight_path = os.path.join(self.checkpoint_path, f'depth_anything_v2_{self.encoder}.pth')
            self.get_logger().info(f"正在從 '{model_weight_path}' 載入權重...")
            
            if not os.path.exists(model_weight_path):
                raise IOError(f"權重檔案不存在: {model_weight_path}")

            self.model.load_state_dict(torch.load(model_weight_path, map_location=self.DEVICE))
            self.model = self.model.to(self.DEVICE).eval()
            
            self.get_logger().info("模型載入成功。")

        except Exception as e:
            self.get_logger().fatal(f"載入模型時發生嚴重錯誤: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

        # 3. 創建 CvBridge 和 Colormap
        self.bridge = CvBridge()
        self.cmap = matplotlib.colormaps.get_cmap('Spectral_r')

        # 4. 創建 ROS Publisher
        self.depth_pub = self.create_publisher(Image, "/ai_depth/image", 1)
        self.vis_pub = self.create_publisher(Image, "/ai_depth/image_vis", 1)
        self.stitched_pub = self.create_publisher(Image, "/ai_depth/image_stitched", 1)

        # 5. 創建 ROS Subscriber
        self.image_sub = self.create_subscription(
            Image, "/camera/image_raw", self.image_callback, 1 # QoS profile
        )
        
        self.get_logger().info("節點已準備就緒，等待 /camera/image_raw 的影像...")
        self.get_logger().info("將會發布以下主題:")
        self.get_logger().info("  - /ai_depth/image (原始深度圖 32FC1)")
        self.get_logger().info("  - /ai_depth/image_vis (可視化深度圖 bgr8)")
        self.get_logger().info("  - /ai_depth/image_stitched (原始影像與深度圖拼接 bgr8)")


    def image_callback(self, ros_image: Image):
        self.get_logger().debug("接收到一幀影像")

        try:
            # 將 ROS Image 訊息轉換為 OpenCV 影像 (BGR 格式)
            cv_image = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge 轉換錯誤: {e}")
            return

        try:
            # --- 【修改】開始計時 ---
            inference_start_time = time.time()
            
            # 執行深度預測 (返回的是 float32 NumPy 陣列)
            depth_map = self.model.infer_image(cv_image, self.input_size)
            
            # --- 【修改】計算延遲與 FPS ---
            latency_ms = (time.time() - inference_start_time) * 1000
            inference_fps = 1000.0 / latency_ms if latency_ms > 0 else float('inf')
            self.get_logger().info(f"Inference FPS: {inference_fps:.2f} | Latency: {latency_ms:.2f} ms")


            if depth_map is None:
                self.get_logger().warn("模型預測返回 None。")
                return

            # --- 發布給 RVIZ 使用的原始深度圖 (32FC1) ---
            depth_msg = self.bridge.cv2_to_imgmsg(depth_map.astype(np.float32), "32FC1")
            depth_msg.header = ros_image.header
            self.depth_pub.publish(depth_msg)

            # --- 發布用於人類觀看的可視化深度圖 (bgr8) ---
            depth_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255.0
            depth_norm = depth_norm.astype(np.uint8)

            if self.grayscale:
                depth_colormap = np.repeat(depth_norm[..., np.newaxis], 3, axis=-1)
            else:
                depth_colormap = (self.cmap(depth_norm)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
            
            vis_msg = self.bridge.cv2_to_imgmsg(depth_colormap, "bgr8")
            vis_msg.header = ros_image.header
            self.vis_pub.publish(vis_msg)

            # --- 拼接原始影像與深度圖並發布 ---
            try:
                # 假設 depth_colormap 和 cv_image 具有相同的 H, W 尺寸
                if cv_image.shape[:2] != depth_colormap.shape[:2]:
                     self.get_logger().warn(f"原始影像 ({cv_image.shape[:2]}) 與深度圖 ({depth_colormap.shape[:2]}) 尺寸不匹配，無法拼接。")
                else:
                    stitched_image = np.hstack((cv_image, depth_colormap))
                    
                    # --- 【修改】在影像上繪製 FPS 文字 ---
                    fps_text = f"Inference FPS: {inference_fps:.2f}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(stitched_image, fps_text, (15, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    
                    stitched_msg = self.bridge.cv2_to_imgmsg(stitched_image, "bgr8")
                    stitched_msg.header = ros_image.header
                    self.stitched_pub.publish(stitched_msg)
            except Exception as e:
                self.get_logger().error(f"拼接或發布拼接影像時發生錯誤: {e}")


        except Exception as e:
            self.get_logger().error(f"在影像回呼函式中發生錯誤: {e}")
            import traceback
            traceback.print_exc()

def main(args=None):
    rclpy.init(args=args)
    node = DepthNodePyTorch()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()