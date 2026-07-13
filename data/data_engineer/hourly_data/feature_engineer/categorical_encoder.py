"""
类别特征编码模块
功能：
1. 为类别特征（city, country）建立索引映射
2. 将类别转换为整数ID（为后续 nn.Embedding 准备）
3. 保存和加载映射字典
"""
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Optional
from pathlib import Path


class CategoricalEncoder:
    """类别特征编码器"""
    
    def __init__(self, categorical_columns: List[str]):
        """
        初始化类别编码器
        
        Args:
            categorical_columns: 需要编码的类别列名列表
        """
        self.categorical_columns = categorical_columns
        self.mappings: Dict[str, Dict[str, int]] = {}
        self.reverse_mappings: Dict[str, Dict[int, str]] = {}
        
    def fit(self, df: pd.DataFrame) -> 'CategoricalEncoder':
        """
        拟合：为每个类别列建立索引映射
        
        Args:
            df: 训练数据框
            
        Returns:
            self
        """
        for col in self.categorical_columns:
            if col not in df.columns:
                raise ValueError(f"列 '{col}' 不存在于数据框中")
            
            # 获取唯一值并排序（保证可复现性）
            unique_values = sorted(df[col].unique())
            
            # 建立映射：类别 -> ID（key 转为原生 Python 类型，确保 JSON 可序列化）
            self.mappings[col] = {
                int(val) if isinstance(val, (np.integer,)) else str(val) if isinstance(val, (np.str_,)) else val: idx
                for idx, val in enumerate(unique_values)
            }
            
            # 建立反向映射：ID -> 类别
            self.reverse_mappings[col] = {idx: val for val, idx in self.mappings[col].items()}
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        转换：将类别列转换为整数ID
        
        Args:
            df: 输入数据框
            
        Returns:
            转换后的数据框（添加 *_id 列）
        """
        if not self.mappings:
            raise ValueError("编码器未拟合，请先调用 fit() 方法")
        
        df = df.copy()
        
        for col in self.categorical_columns:
            if col not in df.columns:
                raise ValueError(f"列 '{col}' 不存在于数据框中")
            
            # 创建新列名
            id_col = f"{col}_id"
            
            # 映射类别到ID，处理未见过的类别
            df[id_col] = df[col].map(self.mappings[col])
            
            # 检查是否有未映射的值（测试集中出现训练集没有的类别）
            if df[id_col].isna().any():
                unmapped = df[df[id_col].isna()][col].unique()
                print(f"警告: 列 '{col}' 中存在未见过的类别: {unmapped}")
                print(f"这些值将被映射为 -1")
                df[id_col].fillna(-1, inplace=True)
            
            # 转换为整数类型
            df[id_col] = df[id_col].astype(int)
        
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        拟合并转换
        
        Args:
            df: 输入数据框
            
        Returns:
            转换后的数据框
        """
        return self.fit(df).transform(df)
    
    def save_mappings(self, save_dir: Path) -> None:
        """
        保存映射字典到JSON文件
        
        Args:
            save_dir: 保存目录
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for col in self.categorical_columns:
            mapping_path = save_dir / f"{col}_mapping.json"
            with open(mapping_path, 'w', encoding='utf-8') as f:
                json.dump(self.mappings[col], f, ensure_ascii=False, indent=2)
            print(f"已保存映射: {mapping_path}")
    
    def load_mappings(self, save_dir: Path) -> 'CategoricalEncoder':
        """
        从JSON文件加载映射字典
        
        Args:
            save_dir: 保存目录
            
        Returns:
            self
        """
        save_dir = Path(save_dir)
        
        for col in self.categorical_columns:
            mapping_path = save_dir / f"{col}_mapping.json"
            if not mapping_path.exists():
                raise FileNotFoundError(f"映射文件不存在: {mapping_path}")
            
            with open(mapping_path, 'r', encoding='utf-8') as f:
                self.mappings[col] = json.load(f)
            
            # 重建反向映射
            self.reverse_mappings[col] = {idx: val for val, idx in self.mappings[col].items()}
            
            print(f"已加载映射: {mapping_path}")
        
        return self
    
    def get_num_classes(self, column: str) -> int:
        """
        获取某列的类别数量（用于确定 Embedding 维度）
        
        Args:
            column: 列名
            
        Returns:
            类别数量
        """
        if column not in self.mappings:
            raise ValueError(f"列 '{column}' 未被编码")
        return len(self.mappings[column])
    
    def decode(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        反向转换：将ID转换回原始类别
        
        Args:
            df: 包含 *_id 列的数据框
            
        Returns:
            解码后的数据框
        """
        df = df.copy()
        
        for col in self.categorical_columns:
            id_col = f"{col}_id"
            if id_col not in df.columns:
                continue
            
            # 反向映射
            df[f"{col}_decoded"] = df[id_col].map(self.reverse_mappings[col])
        
        return df
    
    def get_info(self) -> Dict:
        """
        获取编码器信息
        
        Returns:
            包含各列类别数量的字典
        """
        info = {}
        for col in self.categorical_columns:
            if col in self.mappings:
                info[col] = {
                    'num_classes': len(self.mappings[col]),
                    'classes': list(self.mappings[col].keys())
                }
        return info


if __name__ == "__main__":
    # 测试代码
    test_df = pd.DataFrame({
        'city': ['Paris', 'London', 'Berlin', 'Paris', 'London'],
        'country': ['France', 'UK', 'Germany', 'France', 'UK']
    })
    
    encoder = CategoricalEncoder(['city', 'country'])
    
    print("原始数据:")
    print(test_df)
    
    # 拟合并转换
    result = encoder.fit_transform(test_df)
    print("\n编码后的数据:")
    print(result)
    
    # 查看编码信息
    print("\n编码器信息:")
    print(encoder.get_info())
    
    # 测试解码
    decoded = encoder.decode(result)
    print("\n解码验证:")
    print(decoded[['city', 'city_id', 'city_decoded']])
