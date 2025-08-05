#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# filename: depth_anything_v2_launch.py
# Place this file in the 'launch' directory of your 'Depth_Anything_V2' package.

# ros2 launch Depth_Anything_V2 depth_anything.launch.py depth_model_type:=onnx_hybrid start_rviz:=true encoder:=vits precision:=fp16

# onnx_hybrid, tensorrt
 
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument 
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.conditions import IfCondition
from launch_ros.actions import Node
import pathlib

def generate_launch_description():

    # ================================
    # 設定 yolov8 環境變數給 ROS2 執行使用
    # ================================
    os.environ["CONDA_PREFIX"] = "/home/nvidia/miniconda3/envs/yolov8" 
    os.environ["PATH"] = "/home/nvidia/miniconda3/envs/yolov8/bin:" + os.environ["PATH"]
    os.environ["PYTHONPATH"] = "/home/nvidia/miniconda3/envs/yolov8/lib/python3.10/site-packages:" + os.environ.get("PYTHONPATH", "")

    # 定義新的套件名稱
    pkg_name = 'depth_anything_ros2'
    pkg_source_dir = os.getcwd()
    print("pkg_source_dir 位置:", pkg_source_dir)
    
    # 為了讓 RViz 等仍使用標準路徑的節點正常運作，我們保留 share 路徑的獲取
    pkg_share_dir = get_package_share_directory(pkg_name)
    rviz_config_file = os.path.join(pkg_share_dir, 'rviz', 'view_images.rviz')
    print("套件的分享路徑:", pkg_share_dir)

    # 1. 宣告所有 Launch Arguments
    # =================================================================
    # --- 通用設定 ---
    declare_depth_model_type_arg = DeclareLaunchArgument(
        'depth_model_type', default_value='onnx_hybrid',
        description="選擇深度估計模型類型: 'onnx_hybrid', 'tensorrt'")

    declare_start_rviz_arg = DeclareLaunchArgument(
        'start_rviz', default_value='false',
        description="是否啟動 RViz (true/false)")

    # --- 通用模型參數 (已整合) ---
    declare_encoder_arg = DeclareLaunchArgument(
        'encoder', default_value='vits',
        description="模型的通用編碼器類型 (vits, vitb, vitl, vitg)")
    
    declare_precision_arg = DeclareLaunchArgument(
        'precision', default_value='fp16',
        description="模型的通用精度 (fp16/fp32)")
        
    declare_maxdistance_arg = DeclareLaunchArgument(
        'max_depth', default_value='120.0',
        description="轉換絕對深度最大距離")
    declare_mindistance_arg = DeclareLaunchArgument(
        'min_depth', default_value='0.1',
        description="轉換絕對深度最小距離")
    
    # --- Weights path ---
    declare_root_weights_arg = DeclareLaunchArgument(
        'root_weights', default_value=os.path.join(pkg_source_dir, 'src', pkg_name, 'checkpoints'),
        description="TensorRT 引擎和 ONNX 檔案的根路徑")
    
    # --- input Topic ---
    declare_input_image_topic_arg = DeclareLaunchArgument(
        'input_image_topic', default_value='/zedx_one_0/image',
        description="原始影像的輸入 topic")

    # --- output topic ---
    declare_output_depth_np_topic_arg = DeclareLaunchArgument(
        'output_depth_np_topic', default_value='/zedx_0_depth_np_image',
        description="原始深度圖的輸出 (bgr8)")

    declare_output_depth_colormap_topic_arg = DeclareLaunchArgument(
        'output_depth_colormap_topic', default_value='/zedx_0_depth_colormap_image',
        description="彩色可視化深度圖的輸出 topic (bgr8)")

    declare_output_pred_depth_colormap_topic_arg = DeclareLaunchArgument(
        'output_pred_depth_colormap_topic', default_value='/zedx_0_pred_depth_colormap_image',
        description="灰階深度圖 (32FC1)")

    # --- ONNX Hybrid parameters ---
    declare_onnx_hybrid_width_arg = DeclareLaunchArgument(
        'onnx_hybrid_width', default_value='518',
        description="模型輸入影像的寬度")

    declare_onnx_hybrid_height_arg = DeclareLaunchArgument(
        'onnx_hybrid_height', default_value='518',
        description="模型輸入影像的高度")

    declare_use_trt_arg = DeclareLaunchArgument(
        'use_trt', default_value='false',
        description="是否啟用 TensorRT 加速 (true/false)")

    declare_onnx_hybrid_workspace_arg = DeclareLaunchArgument(
        'onnx_hybrid_workspace', default_value='4',
        description="TensorRT 建構時的工作區記憶體大小 (GB)")

    # # --- RViz 設定 ---
    # declare_rviz_config_arg = DeclareLaunchArgument(
    #     name='rviz_config',
    #     default_value=rviz_config_file,
    #     description='Full path to the RViz config file to use'
    # )

    # 2. 定義要啟動的節點
    # =================================================================
    depth_onnx_node = Node(
        package=pkg_name,
        executable='demo_onnx_node',
        name='depth_onnx_node',
        output='screen',
        condition=IfCondition(
            PythonExpression(["'", LaunchConfiguration("depth_model_type"), "' == 'onnx_hybrid'"])
        ),
        parameters=[{
            'input_image':LaunchConfiguration('input_image_topic'),

            'depth_np':LaunchConfiguration('output_depth_np_topic'),
            'depth_colormap':LaunchConfiguration('output_depth_colormap_topic'),
            'pred_depth_colormap':LaunchConfiguration('output_pred_depth_colormap_topic'),

            'root_weights':LaunchConfiguration('root_weights'),
            'encoder':LaunchConfiguration('encoder'),
            'width':LaunchConfiguration('onnx_hybrid_width'),
            'height':LaunchConfiguration('onnx_hybrid_height'),
            'max_depth':LaunchConfiguration('max_depth'),
            'min_depth':LaunchConfiguration('min_depth'),
            'use_trt':LaunchConfiguration('use_trt'),
            'precision':LaunchConfiguration('precision'),
            'workspace':LaunchConfiguration('onnx_hybrid_workspace'),

            
        }]
    )

    depth_tensorrt_node = Node(
        package=pkg_name,
        executable='trt_node',
        name='depth_tensorrt_node',
        output='screen',
        condition=IfCondition(
            PythonExpression(["'", LaunchConfiguration('depth_model_type'), "' == 'tensorrt'"])
        ),
        parameters=[{
            'input_image':LaunchConfiguration('input_image_topic'),

            'depth_np':LaunchConfiguration('output_depth_np_topic'),
            'depth_colormap':LaunchConfiguration('output_depth_colormap_topic'),
            'pred_depth_colormap':LaunchConfiguration('output_pred_depth_colormap_topic'),

            'root_weights':LaunchConfiguration('root_weights'),
            'encoder': LaunchConfiguration('encoder'), # 使用通用參數
            'precision': LaunchConfiguration('precision'), # 使用通用參數
        }],
    )

    # rviz_node = Node(
    #     package='rviz2',
    #     executable='rviz2',
    #     name='rviz2',
    #     arguments=['-d', LaunchConfiguration('rviz_config')],
    #     condition=IfCondition(LaunchConfiguration('start_rviz'))
    # )

    # 3. 組合並返回 LaunchDescription
    # =================================================================
    return LaunchDescription([
        # 宣告參數
        declare_depth_model_type_arg,
        declare_start_rviz_arg,
        declare_encoder_arg,
        declare_precision_arg,
        declare_maxdistance_arg,
        declare_mindistance_arg,
        declare_input_image_topic_arg,

        declare_output_depth_np_topic_arg,
        declare_output_depth_colormap_topic_arg,
        declare_output_pred_depth_colormap_topic_arg,

        declare_root_weights_arg,

        declare_onnx_hybrid_width_arg,
        declare_onnx_hybrid_height_arg,
        declare_use_trt_arg,
        declare_onnx_hybrid_workspace_arg,
        # declare_rviz_config_arg,

        # 啟動節點
        depth_tensorrt_node,
        depth_onnx_node,
    ])
