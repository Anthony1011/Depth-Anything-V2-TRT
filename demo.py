import argparse
import numpy as np
import os
import torch
import cv2
from depth_anything_v2.dpt import DepthAnythingV2
import time


class DepthAnything():
    def __init__(self) -> None:

        self.augments()
        self.model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.load_model()

    def load_model(self):

        depth_anything = DepthAnythingV2(**self.model_configs[self.args.encoder])
        depth_anything.load_state_dict(torch.load(f'./checkpoints/depth_anything_v2_{self.args.encoder}.pth', map_location = self.device), strict=True)
        depth_anything = depth_anything.to(self.device).eval()

        return depth_anything

    def predictions(self, frames):

        p_start = time.time()
        depths = self.model.infer_image(frames, input_size=self.args.input_size)
        p_end = time.time()
        print(f"[predictions] elapsed: {(p_end - p_start) * 1000:.2f} ms")

        return depths

    def run(self):

        run_start = time.time()

        src = 0 if self.args.video_path == '0' else self.args.video_path
        print(f"Images Source {self.args.video_path}")
        cap = cv2.VideoCapture(src)
        assert cap.isOpened(), f'Cannot Open {src}'

        while True:
            ret, frame = cap.read()
            if not ret: 
                break

            W, H = frame.shape[:2]
            print(f"Source Images Size{W,H}")

            frame = cv2.resize(frame, (self.args.width, self.args.height), cv2.INTER_AREA)
            # BGR --> RGB
            frame_rgb = frame[:, :, ::-1].copy()

            #--- 單張推論 ---
            depth = self.predictions(frame_rgb)
            print(f"Predictions Shape : {depth.shape}")

            #--- 視覺化 ---
            width = int(frame.shape[1] * self.args.video_scale / 100)
            height = int(frame.shape[0] * self.args.video_scale / 100)
            dim = (width, height)
            # --- normalize depth to [0-1] then convert to [0-225] 
            depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            # depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            # depth = depth.astype(np.uint8)
            # --- grayscale depth ---
            depth_bgr = cv2.cvtColor(depth_norm, cv2.COLOR_GRAY2BGR)
            # --- colornepth depth ---
            # --- colornepth depth ---
            # 不同染色 cv2.COLORMAP_TURBO cv2.COLORMAP_MAGMA COLORMAP_INFERNO cv2.COLORMAP_JET
            """
            Colormap	        效果	               用途建議
            COLORMAP_JET    	藍→綠→黃→紅	最常見      明亮清楚
            COLORMAP_TURBO	    藍→淺藍→黃→紅           更現代化的 Jet(更平滑)
            COLORMAP_MAGMA  	黑→紫→紅→黃	            科學視覺用，暗底
            COLORMAP_INFERNO	黑→紅→黃白             	高對比，亮區清楚
            """
            depth_colormap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_MAGMA)
            # --- resize image to visualization ---
            frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
            depth_bgr = cv2.resize(depth_bgr, dim, interpolation=cv2.INTER_AREA)
            depth_colormap = cv2.resize(depth_colormap, dim, interpolation=cv2.INTER_AREA)

            Result = np.hstack((frame, depth_colormap))
            cv2.imshow("Result", Result)

            run_end = time.time()
            print(f"[run] elapsed: {(run_end - run_start) * 1000:.2f} ms")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                return True  # 返回True表示需要退出循环

        cap.release()
        cv2.destroyAllWindows()

    def augments(self):

        parser = argparse.ArgumentParser(description='Depth Anything V2')
        parser.add_argument('--video_path', type=str, default="./videos/4.mp4",  help='image source video_path or 0 from webcame')
        parser.add_argument('--input_size', type=int, default=518)
        parser.add_argument('--outdir', type=str, default='./vis_video_depth')
        parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl', 'vitg'])
        parser.add_argument('--pred_only', dest='pred_only', action='store_true', help='only display the prediction')
        parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
        parser.add_argument('--width', type=int , default=640 ,help='')
        parser.add_argument('--height', type=int , default=480 ,help='')
        parser.add_argument('--video_scale', type=int , default=100 ,help='percentage to shrink display window (e.g., 50)')


        self.args = parser.parse_args()

if __name__=="__main__":

    DAv2 = DepthAnything()
    DAv2.run()