#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# filename: depth_anything_v2_launch.py
# Place this file in the 'launch' directory of your 'Depth_Anything_V2' package.

# ros2 launch Depth_Anything_V2 depth_anything.launch.py depth_model_type:=onnx_hybrid start_rviz:=true encoder:=vits precision:=fp16

# onnx_hybrid, tensorrt, pytorch
 
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.conditions import IfCondition
from launch_ros.actions import Node
import pathlib

def generate_launch_description():

    # 定義新的套件名稱
    pkg_name = 'Depth_Anything_V2'
    
    # ------------------- 修改開始 -------------------
    # 註解掉舊的寫法
    # pkg_share_dir = get_package_share_directory(pkg_name)
    
    # 2. 使用 pathlib 獲取套件的 "原始碼" 路徑
    # __file__ 指的是本 launch 檔案的路徑
    # .parent 指的是 launch/ 目錄
    # .parent.parent 指的就是套件的根目錄 (src/Depth_Anything_V2/)
    pkg_source_dir = os.getcwd()
    print("套件的原始碼路徑:", pkg_source_dir) 
    
    # 為了讓 RViz 等仍使用標準路徑的節點正常運作，我們保留 share 路徑的獲取
    pkg_share_dir = get_package_share_directory(pkg_name)
    rviz_config_file = os.path.join(pkg_share_dir, 'rviz', 'view_images.rviz')
    print("套件的分享路徑:", pkg_share_dir)
    # ------------------- 修改結束 -------------------
    # 1. 宣告所有 Launch Arguments
    # =================================================================
    # --- 通用設定 ---
    declare_depth_model_type_arg = DeclareLaunchArgument(
        'depth_model_type', default_value='tensorrt',
        description="選擇深度估計模型類型: 'pytorch', 'tensorrt', 'onnx_hybrid'")

    declare_start_rviz_arg = DeclareLaunchArgument(
        'start_rviz', default_value='true',
        description="是否啟動 RViz (true/false)")

    # --- 通用模型參數 (已整合) ---
    declare_encoder_arg = DeclareLaunchArgument(
        'encoder', default_value='vitb',
        description="模型的通用編碼器類型 (vits, vitb, vitl, vitg)")
    
    declare_precision_arg = DeclareLaunchArgument(
        'precision', default_value='fp16',
        description="模型的通用精度 (fp16/fp32)")

    # --- Topic & 路徑設定 ---
    declare_video_path_arg = DeclareLaunchArgument(
        'video_path', default_value=os.path.join(pkg_share_dir, 'data', 'video2.mp4'),
        description="輸入影片檔案的路徑")
    
    declare_video_output_topic_arg = DeclareLaunchArgument(
        'video_output_topic', default_value='/camera/image_raw',
        description="影片發布節點的輸出 topic")

    declare_input_image_topic_arg = DeclareLaunchArgument(
        'input_image_topic', default_value='/camera/image_raw',
        description="原始影像的輸入 topic")

    declare_output_depth_raw_topic_arg = DeclareLaunchArgument(
        'output_depth_raw_topic', default_value='/depth_anything/depth_raw',
        description="原始深度圖的輸出 topic (32FC1)")

    declare_output_depth_vis_topic_arg = DeclareLaunchArgument(
        'output_depth_vis_topic', default_value='/depth_anything/depth_color',
        description="彩色可視化深度圖的輸出 topic (bgr8)")

    declare_output_stitched_topic_arg = DeclareLaunchArgument(
        'output_stitched_topic', default_value='/depth_anything/stitched_image',
        description="原始影像與深度圖拼接後的輸出 topic (bgr8)")

    # --- PyTorch 專用參數 ---
    declare_pytorch_checkpoint_path_arg = DeclareLaunchArgument(
        'pytorch_checkpoint_path', default_value=os.path.join(pkg_source_dir, pkg_name, pkg_name, 'checkpoints', 'torch'),
        description="PyTorch 模型權重檔案 (.pth) 的資料夾路徑")

    declare_pytorch_input_size_arg = DeclareLaunchArgument(
        'pytorch_input_size', default_value='518',
        description="PyTorch 模型輸入影像的尺寸 (寬高相同)")

    declare_pytorch_grayscale_arg = DeclareLaunchArgument(
        'pytorch_grayscale', default_value='false',
        description="PyTorch 深度圖是否輸出灰階可視化 (true/false)")

    # --- TensorRT 專用參數 ---
    declare_trt_root_weights_arg = DeclareLaunchArgument(
        'trt_root_weights', default_value=os.path.join(pkg_source_dir, pkg_name, pkg_name, 'checkpoints'),
        description="TensorRT 引擎和 ONNX 檔案的根路徑")

    declare_trt_outdir_arg = DeclareLaunchArgument(
        'trt_outdir', default_value=os.path.join(pkg_share_dir, 'vis_video_depth_trt'),
        description="TensorRT 推論結果的輸出資料夾")

    declare_trt_video_scale_arg = DeclareLaunchArgument(
        'trt_video_scale', default_value='40',
        description="TensorRT 影片縮放比例")

    # --- ONNX Hybrid 專用參數 ---
    declare_onnx_hybrid_checkpoints_path_arg = DeclareLaunchArgument(
        'onnx_hybrid_checkpoints_path', default_value=os.path.join(pkg_source_dir, pkg_name, pkg_name, 'checkpoints'),
        description="模型權重、ONNX 和 TensorRT 引擎的路徑")

    declare_onnx_hybrid_width_arg = DeclareLaunchArgument(
        'onnx_hybrid_width', default_value='518',
        description="模型輸入影像的寬度")

    declare_onnx_hybrid_height_arg = DeclareLaunchArgument(
        'onnx_hybrid_height', default_value='518',
        description="模型輸入影像的高度")

    declare_use_trt_arg = DeclareLaunchArgument(
        'use_trt', default_value='true',
        description="是否啟用 TensorRT 加速 (true/false)")

    declare_onnx_hybrid_workspace_arg = DeclareLaunchArgument(
        'onnx_hybrid_workspace', default_value='4',
        description="TensorRT 建構時的工作區記憶體大小 (GB)")

    # --- RViz 設定 ---
    declare_rviz_config_arg = DeclareLaunchArgument(
        name='rviz_config',
        default_value=rviz_config_file,
        description='Full path to the RViz config file to use'
    )

    # 2. 定義要啟動的節點
    # =================================================================
    
    video_publisher_node = Node(
        package=pkg_name,
        executable='open_camera',
        name='video_publisher_node',
        output='screen',
        parameters=[{
            'video_path': LaunchConfiguration('video_path'),
            'output_topic_name': LaunchConfiguration('video_output_topic'),
        }]
    )

    depth_pytorch_node = Node(
        package=pkg_name,
        executable='run_video_node',
        name='depth_pytorch_node',
        output='screen',
        condition=IfCondition(
            PythonExpression(["'", LaunchConfiguration('depth_model_type'), "' == 'pytorch'"])
        ),
        parameters=[{
            'encoder': LaunchConfiguration('encoder'), # 使用通用參數
            'checkpoint_path': LaunchConfiguration('pytorch_checkpoint_path'),
            'input_size': LaunchConfiguration('pytorch_input_size'),
            'grayscale': LaunchConfiguration('pytorch_grayscale'),
        }],
        remappings=[
            ('/camera/image_raw', LaunchConfiguration('video_output_topic')),
            ('/ai_depth/image', LaunchConfiguration('output_depth_raw_topic')),
            ('/ai_depth/image_vis', LaunchConfiguration('output_depth_vis_topic')),
            ('/ai_depth/image_stitched', LaunchConfiguration('output_stitched_topic')),
        ]
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
            'encoder': LaunchConfiguration('encoder'), # 使用通用參數
            'precision': LaunchConfiguration('precision'), # 使用通用參數
            'root_weights': LaunchConfiguration('trt_root_weights'),
            'outdir': LaunchConfiguration('trt_outdir'),
            'video_scale': LaunchConfiguration('trt_video_scale'),
        }],
        remappings=[
            ('/camera/image_raw', LaunchConfiguration('video_output_topic')),
            ('/depth_anything/depth_raw', LaunchConfiguration('output_depth_raw_topic')),
            ('/depth_anything/depth_color', LaunchConfiguration('output_depth_vis_topic')),
            ('/depth_anything/stitched_image', LaunchConfiguration('output_stitched_topic')),
        ]
    )

    depth_onnx_hybrid_node = Node(
        package=pkg_name,
        executable='demo_onnx_node',
        name='depth_onnx_hybrid_node',
        output='screen',
        condition=IfCondition(
            PythonExpression(["'", LaunchConfiguration('depth_model_type'), "' == 'onnx_hybrid'"])
        ),
        parameters=[{
            'input_topic': LaunchConfiguration('input_image_topic'),
            'output_topic': LaunchConfiguration('output_stitched_topic'),
            'checkpoints_path': LaunchConfiguration('onnx_hybrid_checkpoints_path'),
            'encoder': LaunchConfiguration('encoder'), # 使用通用參數
            'width': LaunchConfiguration('onnx_hybrid_width'),
            'height': LaunchConfiguration('onnx_hybrid_height'),
            'use_trt': LaunchConfiguration('use_trt'),
            'precision': LaunchConfiguration('precision'), # 使用通用參數
            'workspace': LaunchConfiguration('onnx_hybrid_workspace'),
        }]
    )
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', LaunchConfiguration('rviz_config')],
        condition=IfCondition(LaunchConfiguration('start_rviz'))
    )

    # 3. 組合並返回 LaunchDescription
    # =================================================================
    return LaunchDescription([
        # 宣告參數
        declare_depth_model_type_arg,
        declare_start_rviz_arg,
        declare_encoder_arg,
        declare_precision_arg,
        declare_video_path_arg,
        declare_video_output_topic_arg,
        declare_input_image_topic_arg,
        declare_output_depth_raw_topic_arg,
        declare_output_depth_vis_topic_arg,
        declare_output_stitched_topic_arg,
        declare_pytorch_checkpoint_path_arg,
        declare_pytorch_input_size_arg,
        declare_pytorch_grayscale_arg,
        declare_trt_root_weights_arg,
        declare_trt_outdir_arg,
        declare_trt_video_scale_arg,
        declare_onnx_hybrid_checkpoints_path_arg,
        declare_onnx_hybrid_width_arg,
        declare_onnx_hybrid_height_arg,
        declare_use_trt_arg,
        declare_onnx_hybrid_workspace_arg,
        declare_rviz_config_arg,

        # 啟動節點
        video_publisher_node,
        depth_pytorch_node,
        depth_tensorrt_node,
        depth_onnx_hybrid_node,
        rviz_node,
    ])