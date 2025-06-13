import torch
import sys
sys.path.insert(0, '/workspace')

# 测试配置加载
from openstl.utils import create_parser
from openstl.api import BaseExperiment

# 基本参数
args = create_parser().parse_args([
    '-d', 'mmnist',
    '--lr', '1e-3',
    '-c', 'configs/mmnist/simvp/SimVP_gSTA.py',
    '--ex_name', 'test_run',
    '--epochs', '2',  # 只训练2个epoch测试
    '--batch_size', '4'  # 小batch size
])

print("✅ Config loaded successfully!")
print(f"Dataset: {args.dataname}")
print(f"Model: {args.method}")
print(f"Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")

# 创建实验对象测试
try:
    exp = BaseExperiment(args)
    print("✅ Experiment created successfully!")
except Exception as e:
    print(f"❌ Error: {e}")
