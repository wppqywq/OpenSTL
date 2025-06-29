# 最小复现案例

from openstl.api import BaseExperiment
from openstl.utils import create_parser, default_parser

# 创建最简单的配置
parser = create_parser()
args = parser.parse_args([
    '-d', 'coco_search',
    '-m', 'SimVP',
    '--ex_name', 'minimal_test',
    '--epoch', '1',
    '--batch_size', '2',
    '--device', 'mps'
])

# 手动设置关键参数
args.in_shape = (5, 1, 32, 32)
args.hid_S = 32
args.hid_T = 128
args.N_S = 2
args.N_T = 4
args.model_type = 'gSTA'
args.pre_seq_length = 5
args.aft_seq_length = 5
args.drop_path = 0.1  # 这个可能是关键！
args.total_length = 10 

# 添加默认值
defaults = default_parser()
for k, v in defaults.items():
    if not hasattr(args, k):
        setattr(args, k, v)

print("尝试创建 BaseExperiment...")
try:
    exp = BaseExperiment(args)
    print("成功创建!")
    
    # 检查模型
    print("\n检查模型的 BatchNorm 层:")
    for name, module in exp.method.model.named_modules():
        if 'norm' in name.lower():
            if hasattr(module, 'num_features'):
                print(f"{name}: {module.num_features} features")
    
except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()
