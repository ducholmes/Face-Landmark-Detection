import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import torch
import torch.onnx
from src.models.components.resnet import FaceLandmarkResNet
from src.models.facial_landmark_regression_module import FacialLandmarkRegressionModule

torch.serialization.add_safe_globals([FaceLandmarkResNet])

model = FacialLandmarkRegressionModule(FaceLandmarkResNet(num_landmarks=98))

checkpoint_path = './logs/train/last.ckpt'

checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
state_dict = checkpoint['state_dict']
new_state_dict = state_dict
new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

model.load_state_dict(new_state_dict)
model.eval()

dummy_input = torch.randn(1, 3, 256, 256)

onnx_file_path = "./experiment/resnet18.onnx"
torch.onnx.export(
    model, 
    dummy_input, 
    onnx_file_path, 
    export_params=True,       
    opset_version=12,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

print(f"Đã chuyển đổi thành công sang: {onnx_file_path}")