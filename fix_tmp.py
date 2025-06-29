# run_simvp_coco18.py
from __future__ import annotations
import torch
from torch.optim import Adam
from openstl.api   import BaseExperiment
from openstl.utils import create_parser

# ---------- 1. CLI args -------------------------------------------------
parser = create_parser()
args = parser.parse_args([
    "-d", "coco_search",
    "-m", "SimVP",
    "--ex_name", "minimal_run",
    "--epoch", "1",
    "--batch_size", "2",
    "--device", "mps"
])
args.in_shape        = (5, 1, 32, 32)
args.pre_seq_length  = 5
args.aft_seq_length  = 5
args.total_length    = 10
args.model_type      = "gSTA"
args.drop_path       = 0.05
args.num_workers     = 0          # macOS spawn-safe

# ---------- 2. Subclass：禁 dry-run + 截帧 ------------------------------
class MyExperiment(BaseExperiment):
    def _load_callbacks(self, *_, **__):
        return [], "./runs/debug"

    def _feed_model(self, batch):
        frames, target = batch[:2]
        x_in  = frames[:, :self.args.pre_seq_length].to(self.device)
        return self.method(x_in), target.to(self.device)

# ---------- 3. Build experiment ----------------------------------------
exp = MyExperiment(args)
orig_fwd = exp.method.forward           # backup

def fwd_5frames(self, x, *a, **kw):
    # x shape: [B, T, C, H, W]  —— 只留前 5 帧
    x = x[:, :args.pre_seq_length]
    return orig_fwd(x, *a, **kw)

import types
exp.method.forward = types.MethodType(fwd_5frames, exp.method)

# ----- 3-A: 给方法对象塞一个极简 configure_optimizers ---------------
def _adam_only(self):
    opt = Adam(self.parameters(), lr=1e-3)
    return opt  # Lightning 接受只返 optimizer

# 绑定到实例
import types
exp.method.configure_optimizers = types.MethodType(_adam_only, exp.method)

# ---------- 4. Smoke test ----------------------------------------------
frames, _ = next(iter(exp.data.train_loader))
print("Loader batch shape:", frames.shape)  # -> torch.Size([2,5,1,32,32])

exp.train()   # 1 epoch

print("✓ End-to-end run finished —— no BN, no Adam, no spawn issues")
