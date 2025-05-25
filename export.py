import torch
import os
import sys
import tensorrt as trt
import onnxruntime as ort

class DA_export():
    def __init__(self, model_configs, device, model, onnx_path, precision, encoder) -> None:
        

        self.model_configs = model_configs
        self.device = device
        self.model = model
        self.onnx_path = onnx_path
        self.precision = precision

    def expotr_model(self, model, onnx_path, width, height):

        # Define dummy input data
        dummy_input = torch.ones((3, width, height)).unsqueeze(0).to(self.device)

        # Provide an example input to the model, this is necessary for exporting to ONNX
        example_output = model.forward(dummy_input)

        # Export the PyTorch model to ONNX format
        torch.onnx.export(
            model,
            dummy_input, 
            onnx_path,
            opset_version=11,
            input_names=["input"],
            output_names=["output"],
            verbose=True
            )
        print(f"Model exported to {onnx_path}")

    def build_trt_engine(self, onnx_path, workspace, precision, engine_path):
        """
        從 ONNX 檔案建立 TensorRT 引擎。

        Args:
            onnx_path (str): 輸入的 ONNX 檔案路徑。
            engine_path (str): 要儲存的 TensorRT 引擎檔案路徑。
            precision (str): 期望的精度 ('fp32' 或 'fp16')。
            workspace (int): 最大工作空間大小 (GiB)。
        """
        
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(network_flags)
        parser = trt.OnnxParser(network, logger)

        if not os.path.exists(onnx_path):
            print(f" 錯誤 找不到 {onnx_path} 檔案")
            sys.exit(1)

        print(f"正在從 {onnx_path} 解析 ONNX 模型...")
        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                print("錯誤: 解析 ONNX 檔案失敗。")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                sys.exit(1)

        print("ONNX 模型解析完成。")

        # 檢查模型輸入輸出
        print(f"模型輸入數量: {network.num_inputs}")
        for i in range(network.num_inputs):
            tensor = network.get_input(i)
            print(f"  Input {i}: name={tensor.name}, dtype={tensor.dtype}, shape={tensor.shape}")
        print(f"模型輸出數量: {network.num_outputs}")
        for i in range(network.num_outputs):
            tensor = network.get_output(i)
            print(f"  Output {i}: name={tensor.name}, dtype={tensor.dtype}, shape={tensor.shape}")

        config = builder.create_builder_config()
        workspace_bytes = workspace * (1 << 30)
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
        print(f"設定最大工作空間限制: {workspace} GiB ({workspace_bytes} Bytes)")

        if precision == "fp16":
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                print("⚡ Building TensorRT engine in FP16 mode")
            else:
                print("❌ Your platform does not support FP16, falling back to FP32")
        elif precision == "fp32":
            print("⚡ Building TensorRT engine in FP32 mode")
        else:
            print(f"錯誤: 不支援的精度 '{precision}'。請使用 'fp32' 或 'fp16'。")

        # From GOOGLE gemini
        serialized_engine = builder.build_serialized_network(network, config)  
        if serialized_engine is None:
            print("錯誤: 建構序列化引擎失敗。")
            sys.exit(1)
        print("TensorRT 引擎建構完成。")

        engine_dir = os.path.dirname(engine_path)
        if engine_dir and not os.path.exists(engine_dir):
            os.makedirs(engine_dir)
            print(f"已建立目錄: {engine_dir}")

        with open(engine_path, 'wb') as f:
            f.write(serialized_engine)
        print(f"引擎已成功儲存至: {engine_path}")
