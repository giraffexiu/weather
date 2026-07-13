"""端到端集成测试：用少量数据快速验证 DataLoader → 模型"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader
from dataset_loader import (
    WeatherSequenceDataset,
    config as dl_config,
)
from hourly_train.model import create_model, count_parameters

print("="*60)
print("端到端集成测试（小样本）")
print("="*60)

device = torch.device('cpu')

# 使用小样本数据集（禁用缓存和城市分组以加速）
ds = WeatherSequenceDataset(
    data_path=dl_config.TRAIN_DATA_PATH,
    seq_length=6,           # 短序列加速
    pred_horizon=1,
    use_cache=False,
    group_by_city=False,    # 不按城市分组，加速创建
)
print(f"数据集大小: {len(ds):,}")

loader = DataLoader(ds, batch_size=16, shuffle=True)
sample_batch = next(iter(loader))
print(f"\nBatch keys: {list(sample_batch.keys())}")
for k, v in sample_batch.items():
    if isinstance(v, torch.Tensor):
        print(f"  {k}: {v.shape}, dtype={v.dtype}")

# 模型
model = create_model(device=device)
print(f"\n参数量: {count_parameters(model):,}")

model.eval()
with torch.no_grad():
    pred = model(sample_batch)

print(f"\n输出 shape: {pred.shape}")
print(f"预测范围: [{pred.min().item():.4f}, {pred.max().item():.4f}]")

target = sample_batch['target']
if target.dim() == 1:
    target = target.unsqueeze(1)
mse = torch.nn.functional.mse_loss(pred, target)
print(f"MSE: {mse.item():.6f}")
print("\n集成测试通过!")
