#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ROS2 節點：讀取影片檔案並將影像發佈到 /camera/image_raw 主題。

這個 ROS2 版本的節點使用一個 Timer 來控制發布頻率，
並且所有 ROS 功能都被封裝在一個 Node class 中。
"""

import rclpy
from rclpy.node import Node
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import sys
# --- 新增 IMPORT ---
from rcl_interfaces.msg import ParameterDescriptor
from rcl_interfaces.msg import ParameterType
import os

# 獲取當前工作路徑
current_path = os.getcwd()

# 顯示路徑
print(f"當前的工作路徑是：{current_path}")

class VideoPublisherNode(Node):
    """
    讀取本地影片檔案並將其作為 sensor_msgs/Image 發布的節點。
    """
    def __init__(self):
        """
        節點的建構函式。
        """
        super().__init__('video_publisher_node')
        self.get_logger().info("初始化影片發布節點 (ROS2)...")

        # --- 1. 修改參數宣告 (已修正) ---
        # 使用 ParameterDescriptor 來宣告參數，更清晰且健壯
        video_path_descriptor = ParameterDescriptor(
            type=ParameterType.PARAMETER_STRING,  # 使用 ParameterType 中的常數
            description='Path to the video file to be processed.'
        )
        publish_rate_descriptor = ParameterDescriptor(
            type=ParameterType.PARAMETER_DOUBLE, # 使用 ParameterType 中的常數
            description='Publish rate in Hz. If not set, it uses the video\'s FPS.'
        )
        
        # 宣告參數時提供預設值。對於 video_path，我們不給預設值，讓程式在未提供時報錯。
        # 對於 publish_rate，給定 0.0 作為一個特殊標記，表示「使用影片的 FPS」。
        # 注意：在Foxy中，更推薦的寫法是將descriptor作為關鍵字參數傳遞
        self.declare_parameter('video_path', descriptor=video_path_descriptor)
        self.declare_parameter('publish_rate', 0.0, descriptor=publish_rate_descriptor)

        # 獲取參數值
        # 由於 'video_path' 沒有預設值，如果未提供，get_parameter 會拋出 ParameterNotDeclaredException
        # 我們用 try-except 來處理這種情況，或者依賴後面的檢查
        try:
            video_path = self.get_parameter('video_path').get_parameter_value().string_value
        except rclpy.exceptions.ParameterNotDeclaredException:
            video_path = "" # 手動設置為空字符串，以便後續邏輯處理

        if not video_path:
            self.get_logger().error("錯誤：未提供影片路徑！請使用 `--ros-args -p video_path:=/your/video/path` 參數設定。")
            # 拋出異常是比 sys.exit() 更好的方式，讓 main 函數中的 finally 能夠執行
            raise SystemExit("Video path parameter is required.")

        # 2. 開啟影片並獲取幀率
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            self.get_logger().error(f"無法開啟影片檔案: {video_path}")
            raise SystemExit(f"Could not open video file: {video_path}")

        video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:
            video_fps = 30.0
            self.get_logger().warn("無法讀取影片幀率，使用預設值 30 Hz")

        # --- 2. 修改獲取 publish_rate 的邏輯 (保持不變) ---
        user_publish_rate = self.get_parameter('publish_rate').get_parameter_value().double_value

        if user_publish_rate > 0.0:
            publish_rate_hz = user_publish_rate
            self.get_logger().info(f"使用使用者設定的發布頻率: {publish_rate_hz} Hz")
        else:
            publish_rate_hz = video_fps
            self.get_logger().info(f"未設定發布頻率，使用影片的 FPS: {publish_rate_hz} Hz")

        self.get_logger().info(f"影片路徑: {video_path}")
        self.get_logger().info(f"最終發布頻率: {publish_rate_hz} Hz")

        # 3. 創建 ROS Publisher 和 CvBridge 實例
        self.image_pub = self.create_publisher(Image, "/camera/image_raw", 10)
        self.bridge = CvBridge()
        
        # 4. 創建一個 Timer 來以固定頻率觸發影像發布
        timer_period = 1.0 / publish_rate_hz
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.get_logger().info("影片發布節點已啟動，開始發布影像...")

    def timer_callback(self):
        """
        Timer 的回呼函式，負責讀取、轉換並發布一幀影像。
        """
        ret, frame = self.cap.read()

        if ret:
            try:
                # 將 OpenCV 影像轉換為 ROS Image 訊息
                ros_image = self.bridge.cv2_to_imgmsg(frame, "bgr8")
                ros_image.header.stamp = self.get_clock().now().to_msg()
                ros_image.header.frame_id = "camera_frame"
                
                # 發布訊息
                self.image_pub.publish(ros_image)

            except CvBridgeError as e:
                self.get_logger().error(f"CvBridge 轉換錯誤: {e}")
        else:
            # 影片結束，重置到開頭以實現循環播放
            self.get_logger().info("影片播放完畢，將從頭開始循環播放。")
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    def destroy_node(self):
        """
        節點關閉時釋放資源。
        """
        self.get_logger().info("節點關閉，釋放影片資源。")
        self.cap.release()
        super().destroy_node()

def main(args=None):
    """
    ROS2 節點的主函式。
    """
    rclpy.init(args=args)
    
    # 使用 try-except-finally 確保即使在初始化失敗時也能正確關閉
    video_publisher_node = None
    try:
        video_publisher_node = VideoPublisherNode()
        rclpy.spin(video_publisher_node)
    except KeyboardInterrupt:
        pass
    except SystemExit as e:
        # 捕獲 sys.exit()，讓程式可以乾淨地退出
        if video_publisher_node:
            video_publisher_node.get_logger().info(f"節點因 SystemExit({e}) 關閉。")
    finally:
        if video_publisher_node:
            video_publisher_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()