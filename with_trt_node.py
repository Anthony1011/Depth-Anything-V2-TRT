#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ROS1 節點：Depth Anything V2 TensorRT 推理節點

這個腳本會做以下幾件事：
1. 初始化一個名為 'depth_anything_ros_node' 的 ROS 節點。
2. 從 ROS 參數伺服器讀取模型設定 (如 encoder, precision)。
3. 從同目錄下的 trt.py 導入 DepthAnythingTRT 模型類，並傳入設定。
4. 訂閱 /camera/image_raw 主題，接收影像串流。
5. 將接收到的 ROS Image 訊息轉換為 OpenCV 格式。
6. 將影像送入 DepthAnythingTRT 模型進行深度預測。
7. 將預測出的深度圖：
   a) 以原始浮點數格式 (32FC1) 發布到 /ai_depth/image，供 RVIZ 或其他節點使用。
   b) 以可視化的彩色圖 (bgr8) 發布到 /ai_depth/image_vis，方便人類觀察。
"""

import sys
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

# --- 使用者指定的 cv_bridge 路徑 ---
# 確保 Python 環境能找到您編譯的 cv_bridge
try:
    sys.path.append('/home/user/Desktop/catkin_ws_cvbridge/devel/lib/python3/dist-packages')
    from cv_bridge import CvBridge, CvBridgeError
except ImportError as e:
    rospy.logerr("無法導入 cv_bridge。請確認您的 ROS 環境和 Python 路徑是否已正確設定。")
    rospy.logerr(f"錯誤訊息: {e}")
    sys.exit(1)

# --- 導入 TensorRT 模型 ---
# 假設 trt.py 和這個腳本在同一個目錄下
try:
    from trt import DepthAnythingTRT
except ImportError:
    rospy.logerr("無法導入 'trt.py'。請確保 trt.py 與此腳本在同一個目錄下，並且相關依賴已安裝。")
    sys.exit(1)


class DepthNode:
    def __init__(self):
        """
        節點的建構函式。
        """
        # 1. 初始化 ROS 節點
        rospy.init_node('depth_anything_ros_node', anonymous=True)
        rospy.loginfo("初始化 Depth Anything ROS 節點...")

        # 2. 載入深度預測模型
        try:
            rospy.loginfo("正在載入 Depth Anything TensorRT 模型...")

            # --- 從 ROS 參數伺服器讀取所有可配置的模型設定 ---
            # 使用 `~` 表示這是節點的私有參數
            encoder = rospy.get_param('~encoder', 'vitb')
            precision = rospy.get_param('~precision', 'fp16')
            root_weights = rospy.get_param('~root_weights', './checkpoints')
            
            # [新增] 讀取其他在 trt.py 中定義的參數，即使它們在ROS節點中不直接使用
            # 這樣可以完全匹配 trt.py 的 argparse，增加靈活性與兼容性
            outdir = rospy.get_param('~outdir', './vis_video_depth_trt')
            video_scale = rospy.get_param('~video_scale', 40)

            rospy.loginfo(f"模型參數 - Encoder: '{encoder}', Precision: '{precision}', Weights: '{root_weights}'")
            rospy.loginfo(f"其他參數(ROS模式下未使用) - Outdir: '{outdir}', Video Scale: {video_scale}")

            # --- 透過修改 sys.argv 將 ROS 參數傳遞給 argparse ---
            # 這是為了與不直接支援 ROS 的 trt.py 腳本進行整合的技巧
            original_argv = list(sys.argv)
            try:
                # 建立一個新的參數列表，第一個是腳本名稱
                sys.argv = [original_argv[0]] 
                sys.argv.append(f'--encoder={encoder}')
                sys.argv.append(f'--precision={precision}')
                sys.argv.append(f'--root_weights={root_weights}')
                sys.argv.append(f'--outdir={outdir}')
                sys.argv.append(f'--video_scale={video_scale}')
                # trt.py 的 argparse 可能需要 video_path，給一個虛擬值即可，因為影像來源是topic
                sys.argv.append('--video_path=dummy_ros_input')
                
                # 這會調用 trt.py 中的 __init__，並透過 argparse 載入 .engine 檔案
                self.model = DepthAnythingTRT()
                rospy.loginfo("模型載入成功。")

            finally:
                # 無論成功或失敗，都必須還原 sys.argv，以免影響 ROS 的正常運作
                sys.argv = original_argv
            # --- 參數設定結束 ---

        except Exception as e:
            rospy.logerr(f"載入 DepthAnythingTRT 模型時發生嚴重錯誤: {e}")
            import traceback
            traceback.print_exc()
            rospy.signal_shutdown("模型載入失敗")
            return

        # 3. 創建 CvBridge 實例
        self.bridge = CvBridge()

        # 4. 創建 ROS Publisher
        self.depth_pub = rospy.Publisher("/ai_depth/image", Image, queue_size=1)
        self.vis_pub = rospy.Publisher("/ai_depth/image_vis", Image, queue_size=1)

        # 5. 創建 ROS Subscriber
        self.image_sub = rospy.Subscriber(
            "/camera/image_raw", Image, self.image_callback, queue_size=1, buff_size=2**24
        )
        
        rospy.loginfo("節點已準備就緒，等待 /camera/image_raw 的影像...")

    def image_callback(self, ros_image):
        """
        訂閱 /camera/image_raw 的回呼函式。
        """
        if rospy.is_shutdown():
            return
        
        rospy.logdebug("接收到一幀影像")

        try:
            cv_image = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge 轉換錯誤: {e}")
            return

        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        try:
            depth_map = self.model.predictions(rgb_image)

            if depth_map is None:
                rospy.logwarn("模型預測返回 None。")
                return

            # --- 發布給 RVIZ 使用的原始深度圖 ---
            depth_msg = self.bridge.cv2_to_imgmsg(depth_map.astype(np.float32), "32FC1")
            depth_msg.header = ros_image.header
            self.depth_pub.publish(depth_msg)

            # --- 發布用於人類觀看的可視化深度圖 ---
            depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            depth_colormap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
            vis_msg = self.bridge.cv2_to_imgmsg(depth_colormap, "bgr8")
            vis_msg.header = ros_image.header
            self.vis_pub.publish(vis_msg)

        except Exception as e:
            rospy.logerr(f"在影像回呼函式中發生錯誤: {e}")
            import traceback
            traceback.print_exc()

    def run(self):
        """
        保持節點運行。
        """
        rospy.spin()


if __name__ == '__main__':
    try:
        node = DepthNode()
        node.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("節點被手動關閉。")
    except Exception as e:
        rospy.logerr(f"節點啟動時發生未預期的錯誤: {e}")

