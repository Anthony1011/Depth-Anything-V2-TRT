#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ROS1 節點：Depth Anything V2 PyTorch 推理節點

基於使用者提供的 run_video.py 進行改寫。
1. 初始化 ROS 節點。
2. 從 ROS 參數伺服器讀取模型設定 (encoder, input_size 等)。
3. 使用 PyTorch 載入 DepthAnythingV2 模型。
4. 訂閱 /camera/image_raw 主題。
5. 將接收到的影像送入模型進行深度預測。
6. 發布兩個主題：
   a) /ai_depth/image: 原始浮點數深度圖 (32FC1)，供 RVIZ 或其他節點使用。
   b) /ai_depth/image_vis: 可視化的彩色深度圖 (bgr8)，方便人類觀察。
"""

import sys
import os
import rospy
import cv2
import numpy as np
import torch
import matplotlib

# --- ROS 相關導入 ---
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

# --- 使用者指定的 cv_bridge 路徑 ---
try:
    sys.path.append('/home/user/Desktop/catkin_ws_cvbridge/devel/lib/python3/dist-packages')
    from cv_bridge import CvBridge, CvBridgeError
except ImportError as e:
    rospy.logerr("無法導入 cv_bridge。請確認您的 ROS 環境和 Python 路徑是否已正確設定。")
    rospy.logerr(f"錯誤訊息: {e}")
    sys.exit(1)

# --- 導入 Depth Anything V2 模型 ---
# 請確保 depth_anything_v2 這個 Python 包在您的環境中可以被找到
try:
    from depth_anything_v2.dpt import DepthAnythingV2
except ImportError:
    rospy.logerr("無法導入 'depth_anything_v2'。請確保已安裝此套件或將其路徑加入 PYTHONPATH。")
    sys.exit(1)


class DepthNodePyTorch:
    def __init__(self):
        """
        節點的建構函式。
        """
        # 1. 初始化 ROS 節點
        rospy.init_node('depth_anything_ros_node', anonymous=True)
        rospy.loginfo("初始化 Depth Anything (PyTorch) ROS 節點...")

        # 2. 載入模型
        try:
            rospy.loginfo("正在載入 Depth Anything V2 PyTorch 模型...")
            
            # --- 從 ROS 參數伺服器讀取模型設定 ---
            self.encoder = rospy.get_param('~encoder', 'vits')
            self.input_size = rospy.get_param('~input_size', 518)
            self.checkpoint_path = rospy.get_param('~checkpoint_path', 'checkpoints')
            self.grayscale = rospy.get_param('~grayscale', False)
            
            rospy.loginfo(f"模型參數 - Encoder: '{self.encoder}', Input Size: {self.input_size}, Checkpoints: '{self.checkpoint_path}'")
            
            # --- 模型初始化 ---
            self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
            rospy.loginfo(f"使用設備: {self.DEVICE}")

            model_configs = {
                'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
                'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
            }
            
            self.model = DepthAnythingV2(**model_configs[self.encoder])
            
            # 構建權重檔案的完整路徑
            model_weight_path = os.path.join(self.checkpoint_path, f'depth_anything_v2_{self.encoder}.pth')
            rospy.loginfo(f"正在從 '{model_weight_path}' 載入權重...")
            
            if not os.path.exists(model_weight_path):
                raise IOError(f"權重檔案不存在: {model_weight_path}")

            self.model.load_state_dict(torch.load(model_weight_path, map_location=self.DEVICE))
            self.model = self.model.to(self.DEVICE).eval()
            
            rospy.loginfo("模型載入成功。")

        except Exception as e:
            rospy.logerr(f"載入模型時發生嚴重錯誤: {e}")
            import traceback
            traceback.print_exc()
            rospy.signal_shutdown("模型載入失敗")
            return

        # 3. 創建 CvBridge 和 Colormap
        self.bridge = CvBridge()
        self.cmap = matplotlib.colormaps.get_cmap('Spectral_r')

        # 4. 創建 ROS Publisher
        self.depth_pub = rospy.Publisher("/ai_depth/image", Image, queue_size=1)
        self.vis_pub = rospy.Publisher("/ai_depth/image_vis", Image, queue_size=1)

        # 5. 創建 ROS Subscriber
        self.image_sub = rospy.Subscriber(
            "/camera/image_raw", Image, self.image_callback, queue_size=1, buff_size=2**24
        )
        
        rospy.loginfo("節點已準備就緒，等待 /camera/image_raw 的影像...")

    def image_callback(self, ros_image):
        if rospy.is_shutdown():
            return
        
        rospy.logdebug("接收到一幀影像")

        try:
            # 將 ROS Image 訊息轉換為 OpenCV 影像 (BGR 格式)
            cv_image = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge 轉換錯誤: {e}")
            return

        try:
            # 執行深度預測，模型內部會處理 BGR->RGB 和其他預處理
            # 返回的是一個 float32 的 NumPy 陣列
            depth_map = self.model.infer_image(cv_image, self.input_size)

            if depth_map is None:
                rospy.logwarn("模型預測返回 None。")
                return

            # --- 發布給 RVIZ 使用的原始深度圖 (32FC1) ---
            depth_msg = self.bridge.cv2_to_imgmsg(depth_map.astype(np.float32), "32FC1")
            depth_msg.header = ros_image.header
            self.depth_pub.publish(depth_msg)

            # --- 發布用於人類觀看的可視化深度圖 (bgr8) ---
            # 複製 run_video.py 中的可視化邏輯
            depth_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255.0
            depth_norm = depth_norm.astype(np.uint8)

            if self.grayscale:
                depth_colormap = np.repeat(depth_norm[..., np.newaxis], 3, axis=-1)
            else:
                depth_colormap = (self.cmap(depth_norm)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
            
            vis_msg = self.bridge.cv2_to_imgmsg(depth_colormap, "bgr8")
            vis_msg.header = ros_image.header
            self.vis_pub.publish(vis_msg)

        except Exception as e:
            rospy.logerr(f"在影像回呼函式中發生錯誤: {e}")
            import traceback
            traceback.print_exc()

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    try:
        node = DepthNodePyTorch()
        node.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("節點被手動關閉。")
    except Exception as e:
        rospy.logerr(f"節點啟動時發生未預期的錯誤: {e}")

