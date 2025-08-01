import argparse
import numpy as np
import os
import torch
import cv2
from depth_anything_v2.dpt import DepthAnythingV2
import time
import onnxruntime as ort
import tensorrt as trt
import sys
from trt import DepthAnythingTRT 
from export import DA_export

class DepthAnything():
    def __init__(self) -> None:

        self.augments()
        self.dav2_trt = DepthAnythingTRT()

        self.weight_path = os.path.join(self.args.root_weights, f"depth_anything_v2_{self.args.encoder}.pth")
        self.onnx_path = os.path.join(self.args.root_weights, "onnx", f"depth_anything_v2_{self.args.encoder}.onnx")
        self.engine_path = os.path.join(self.args.root_weights, "onnx", f"depth_anything_v2_{self.args.encoder}_{self.args.precision}.engine")

        # self.onnx_path = f"./checkpoints/onnx/depth_anything_v2_{self.args.encoder}.onnx"
        # self.engine_path = f"./checkpoints/onnx/depth_anything_v2_{self.args.encoder}_{self.args.precision}.engine"

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
        depth_anything.load_state_dict(torch.load(self.weight_path, map_location = self.device), strict=True)
        depth_anything = depth_anything.to(self.device).eval()

        if self.args.trt == "True":

            if not os.path.exists(self.onnx_path):
                print("⚡Exporting onnx weight")
                export_model = DA_export()
                export_model.expotr_model(depth_anything, self.onnx_path, self.args.width, self.args.height)
            else:
                print(f"✅ ONNX model found at: {self.onnx_path}")

            if not os.path.exists(self.engine_path):
                print("⚡Exporting engine weight")
                export_model.build_trt_engine(self.onnx_path, self.args.workspace, self.args.precision, self.engine_path)

            depth_anything_onnx = ort.InferenceSession(self.onnx_path, providers=["CUDAExecutionProvider"])
            print(f"✅ Complete Load {self.args.encoder} ONNX Depth Anything Mode")

            return depth_anything_onnx

        else:
            return depth_anything

    def predictions(self, frames):

        p_start = time.time()

        if isinstance(self.model, ort.InferenceSession):
            img = frames.astype(np.float32)/255.0
            img = np.transpose(img, (2, 0, 1))
            img = np.expand_dims(img, axis=0)
            output = self.model.run(None, {"input":img})[0]
            depths = output[0]
        else:
            depths = self.model.infer_image(frames, input_size=self.args.input_size)
        p_end = time.time()
        print(f"[predictions] elapsed: {(p_end - p_start) * 1000:.2f} ms")

        return depths

    def run(self):

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
            run_start = time.time()
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

            latency = (time.time() - run_start)
            print(f"[run] elapsed: {latency * 1000:.2f} ms")
            Result = np.hstack((frame, depth_colormap))
            fps = 1000 / latency
            cv2.putText(Result, f"FPS : {(1/latency):.2f} FPS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.imshow("Result", Result)

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
        parser.add_argument('--width', type=int , default=518 ,help='')
        parser.add_argument('--height', type=int , default=518 ,help='')
        parser.add_argument('--video_scale', type=int , default=100 ,help='percentage to shrink display window (e.g., 50)')

        parser.add_argument('--trt', type=str , default="True", choices=["True","False"] ,help='True is mean export trt Model or not')
        parser.add_argument('--precision', type=str, default='fp32', choices=['fp32', 'fp16'], help='Precision mode for TensorRT engine')
        parser.add_argument('--workspace', type=int , default=4, choices=[1, 2, 4, 6, 8] ,help='')

        parser.add_argument('--root_weights', type=str, default="./checkpoints", help='The root directory of weights')

        self.args = parser.parse_args()

if __name__=="__main__":

    DAv2 = DepthAnything()
    DAv2.run()