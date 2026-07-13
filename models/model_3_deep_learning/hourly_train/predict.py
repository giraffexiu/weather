"""
预测接口 — 加载模型对小时数据进行一步预测
可通过命令行或 Python API 调用
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_HERE = Path(__file__).parent
_PROJ_ROOT = _HERE.parent.parent.parent  # weather/
sys.path.insert(0, str(_PROJ_ROOT / "data" / "data_engineer" / "hourly_data"))
sys.path.insert(0, str(_HERE.parent))

from dataset_loader import get_dataloader, config as dl_config
from hourly_train.model import create_model
from hourly_train.config import CHECKPOINT_PATH


class Predictor:
    def __init__(self, checkpoint_path: Path = None, device: str = None):
        self.checkpoint_path = checkpoint_path or CHECKPOINT_PATH
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model = self._load()

    def _load(self):
        model = create_model(device=self.device)
        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        print(f"已加载模型 (epoch={ckpt.get('epoch','?')}, val_loss={ckpt.get('val_loss',0):.6f})")
        return model

    @torch.no_grad()
    def predict(self, csv_path: str, batch_size: int = 1024) -> pd.DataFrame:
        """对 CSV 文件做预测，返回带预测列的 DataFrame"""
        loader = get_dataloader(split='test', data_path=csv_path, batch_size=batch_size,
                                shuffle=False, num_workers=0)
        preds = []
        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            preds.append(self.model(batch).cpu().numpy())
        preds = np.concatenate(preds, axis=0)  # (N, num_targets)

        df = pd.read_csv(csv_path)
        shift = len(df) - len(preds)
        # 预测对应 CSV 中 [shift:] 的行，前面行填充 NaN
        for i, col in enumerate(dl_config.TARGET_COLUMNS):
            df[f'pred_{col}'] = np.nan
            df.loc[shift:, f'pred_{col}'] = preds[:, i]
        return df

    def save(self, csv_path: str, output_path: str, batch_size: int = 1024):
        df = self.predict(csv_path, batch_size)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"预测结果已保存: {output_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Wide & Deep 预测")
    parser.add_argument('--input', type=str, required=True, help='输入 CSV 路径')
    parser.add_argument('--output', type=str, required=True, help='输出 CSV 路径')
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT_PATH)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    pred = Predictor(checkpoint_path=args.checkpoint, device=args.device)
    pred.save(args.input, args.output, args.batch_size)
