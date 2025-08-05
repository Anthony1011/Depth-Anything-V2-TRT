# 檔案：Depth_Anything_V2/setup.py

import os
from glob import glob
# 重新導入 find_packages，這是更穩健的做法
from setuptools import find_packages, setup

package_name = 'depth_anything_ros2'

setup(
    name=package_name,
    version='0.0.1', # 建議版本號從 0.0.1 開始
    
    # 使用 find_packages() 自動尋找所有 Python 套件 (包含 __init__.py 的資料夾)
    # 這會自動找到 'Depth_Anything_V2' 以及所有子套件，例如 'Depth_Anything_V2.depth_anything_v2'
    # 比手動列表更不容易出錯。
    packages=find_packages(exclude=['test']),
    
    # data_files 負責安裝非 Python 程式碼的資源檔案
    data_files=[
        # 這是標準的 ament 索引，必須要有
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        # 安裝 package.xml
        ('share/' + package_name, ['package.xml']),
        
        # 安裝 launch 資料夾下的所有 .launch.py 檔案
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*.launch.py'))),
        
        # 安裝 rviz 資料夾下的所有 .rviz 檔案
        (os.path.join('share', package_name, 'rviz'), glob(os.path.join('rviz', '*.rviz'))),
        
        # 安裝 data 資料夾下的所有 .mp4 影片檔案
        (os.path.join('share', package_name, 'data'), glob(os.path.join('data', '*.mp4'))),
        
        # 安裝模型權重檔案
        (os.path.join('share', package_name, 'checkpoints', 'torch'), glob(os.path.join('checkpoints', 'torch', '*.pth'))),
        
        (os.path.join('share', package_name, 'checkpoints', 'onnx'), glob(os.path.join('checkpoints', 'onnx', '*.*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    # Meta data
    maintainer='Anthony itri',
    maintainer_email='chento7611@gmail.com',
    description='A ROS2 package for Depth Anything V2 inference.',
    license='Apache-2.0',
    tests_require=['pytest'],
    
    # entry_points 將 Python 腳本註冊為可以在終端機執行的命令
    entry_points={
        'console_scripts': [
            # 格式：'可執行檔名稱 = 模組路徑:主函數'
            'trt_node = depth_anything_ros2.trt_node:main',
            'demo_onnx_node = depth_anything_ros2.demo_onnx_node:main',
        ],
    },
)
