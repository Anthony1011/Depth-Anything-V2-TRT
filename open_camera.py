#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# python3 open_camera.py _video_path:="/home/soc123/Desktop/Depth-Anything-V2-TRT/data/video2.mp4"
"""
ROS1 節點（簡化版）：讀取影片檔案並將影像發佈到 /camera/image_raw 主題。

這個腳本移除了 Class 結構，並改用 rospy.Rate 搭配 while 迴圈，
使得程式碼流程更為線性且易於理解。
"""

import sys
import rospy
import cv2
from sensor_msgs.msg import Image

def main():
    """
    節點的主函式。
    """
    # 1. 初始化 ROS 節點
    rospy.init_node('video_publisher_node', anonymous=True)
    rospy.loginfo("初始化影片發布節點 (簡化版)...")

    # --- cv_bridge 路徑處理 ---
    # 這部分對於您的特定環境是必要的，所以予以保留。
    try:
        sys.path.append('/home/soc123/Documents/catkin_ws_cvbridge/devel/lib/python3/dist-packages')
        from cv_bridge import CvBridge, CvBridgeError
    except ImportError as e:
        rospy.logerr(f"無法導入 cv_bridge。請確認 ROS 環境是否已正確設定。錯誤: {e}")
        sys.exit(1)

    # 2. 從 ROS 參數伺服器獲取設定
    video_path = rospy.get_param('~video_path', None)
    if not video_path:
        rospy.logerr("錯誤：未提供影片路徑！請使用 `_video_path:=/your/video/path` 參數設定。")
        return # 直接退出函式

    # 3. 開啟影片並獲取幀率
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        rospy.logerr(f"無法開啟影片檔案: {video_path}")
        return

    # 嘗試從影片讀取幀率，如果失敗則使用預設值 30
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 30
        rospy.logwarn("無法讀取影片幀率，使用預設值 30 Hz")

    publish_rate_hz = rospy.get_param('~publish_rate', video_fps)
    
    rospy.loginfo(f"影片路徑: {video_path}")
    rospy.loginfo(f"發布頻率: {publish_rate_hz} Hz")

    # 4. 創建 ROS Publisher 和 CvBridge 實例
    image_pub = rospy.Publisher("/camera/image_raw", Image, queue_size=10)
    bridge = CvBridge()
    
    # 5. 創建迴圈頻率控制器
    rate = rospy.Rate(publish_rate_hz)

    # 6. 主迴圈：讀取、轉換並發布影像
    rospy.loginfo("影片發布節點已啟動，開始發布影像...")
    while not rospy.is_shutdown():
        ret, frame = cap.read()

        if ret:
            try:
                # 將 OpenCV 影像轉換為 ROS Image 訊息
                ros_image = bridge.cv2_to_imgmsg(frame, "bgr8")
                ros_image.header.stamp = rospy.Time.now()
                ros_image.header.frame_id = "camera_frame"
                
                # 發布訊息
                image_pub.publish(ros_image)

            except CvBridgeError as e:
                rospy.logerr(f"CvBridge 轉換錯誤: {e}")
        else:
            # 影片結束，重置到開頭以實現循環播放
            rospy.loginfo("影片播放完畢，將從頭開始循環播放。")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue # 繼續下一次迴圈

        # 根據設定的頻率延遲
        rate.sleep()

    # 7. 程式結束前釋放資源
    rospy.loginfo("節點關閉，釋放影片資源。")
    cap.release()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        # 當按下 Ctrl+C 時，rospy.is_shutdown() 會變為 True，
        # 迴圈會正常結束，這裡可以不寫任何東西或只留下一句 log。
        rospy.loginfo("影片發布節點被手動關閉。")
    except Exception as e:
        rospy.logerr(f"節點運行時發生未預期的錯誤: {e}")
