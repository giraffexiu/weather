"""
数值特征标准化模块
功能：
1. 对数值特征进行标准化（StandardScaler 或 MinMaxScaler）
2. 保存和加载标准化器
3. 确保训练集和测试集使用相同的标准化参数
"""
import pandas as pd
import numpy as np
import pickle
from typing import List, Literal
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class NumericalScaler:
    """数值特征标准化器"""
    
    def __init__(
        self, 
        numerical_columns: List[str],
        method: Literal['standard', 'minmax'] = 'standard'
    ):
        """
        初始化数值标准化器
        
        Args:
            numerical_columns: 需要标准化的数值列名列表
            method: 标准化方法，'standard' 或 'minmax'
        """
        self.numerical_columns = numerical_columns
        self.method = method
        
        # 选择标准化器
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"不支持的标准化方法: {method}，请选择 'standard' 或 'minmax'")
        
        self.is_fitted = False
        
    def fit(self, df: pd.DataFrame) -> 'NumericalScaler':
        """
        拟合：计算标准化参数（仅在训练集上调用）
        
        Args:
            df: 训练数据框
            
        Returns:
            self
        """
        # 检查列是否存在
        missing_cols = set(self.numerical_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"以下列不存在于数据框中: {missing_cols}")
        
        # 拟合标准化器
        self.scaler.fit(df[self.numerical_columns])
        self.is_fitted = True
        
        return self
    
    def transform(self, df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
        """
        转换：应用标准化（在训练集和测试集上都调用）
        
        Args:
            df: 输入数据框
            inplace: 是否原地修改
            
        Returns:
            标准化后的数据框
        """
        if not self.is_fitted:
            raise ValueError("标准化器未拟合，请先调用 fit() 方法")
        
        if not inplace:
            df = df.copy()
        
        # 检查列是否存在
        missing_cols = set(self.numerical_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"以下列不存在于数据框中: {missing_cols}")
        
        # 应用标准化
        df[self.numerical_columns] = self.scaler.transform(df[self.numerical_columns])
        
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        拟合并转换（仅在训练集上调用）
        
        Args:
            df: 训练数据框
            
        Returns:
            标准化后的数据框
        """
        return self.fit(df).transform(df)
    
    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        反向转换：将标准化后的数据还原为原始尺度
        
        Args:
            df: 标准化后的数据框
            
        Returns:
            还原后的数据框
        """
        if not self.is_fitted:
            raise ValueError("标准化器未拟合，无法进行反向转换")
        
        df = df.copy()
        df[self.numerical_columns] = self.scaler.inverse_transform(df[self.numerical_columns])
        
        return df
    
    def save(self, save_path: Path) -> None:
        """
        保存标准化器到文件
        
        Args:
            save_path: 保存路径（.pkl 文件）
        """
        if not self.is_fitted:
            raise ValueError("标准化器未拟合，无法保存")
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存整个对象
        with open(save_path, 'wb') as f:
            pickle.dump(self, f)
        
        print(f"标准化器已保存至: {save_path}")
    
    @classmethod
    def load(cls, load_path: Path) -> 'NumericalScaler':
        """
        从文件加载标准化器
        
        Args:
            load_path: 加载路径（.pkl 文件）
            
        Returns:
            加载的标准化器实例
        """
        load_path = Path(load_path)
        if not load_path.exists():
            raise FileNotFoundError(f"标准化器文件不存在: {load_path}")
        
        with open(load_path, 'rb') as f:
            scaler = pickle.load(f)
        
        print(f"标准化器已从 {load_path} 加载")
        return scaler
    
    def get_statistics(self) -> pd.DataFrame:
        """
        获取标准化统计信息
        
        Returns:
            包含均值、标准差或最小最大值的数据框
        """
        if not self.is_fitted:
            raise ValueError("标准化器未拟合")
        
        if self.method == 'standard':
            stats = pd.DataFrame({
                'feature': self.numerical_columns,
                'mean': self.scaler.mean_,
                'std': self.scaler.scale_
            })
        else:  # minmax
            stats = pd.DataFrame({
                'feature': self.numerical_columns,
                'min': self.scaler.data_min_,
                'max': self.scaler.data_max_
            })
        
        return stats
    
    def print_statistics(self) -> None:
        """打印标准化统计信息"""
        if not self.is_fitted:
            print("标准化器未拟合")
            return
        
        print(f"\n{'='*60}")
        print(f"标准化方法: {self.method.upper()}")
        print(f"{'='*60}")
        
        stats = self.get_statistics()
        print(stats.to_string(index=False))
        print(f"{'='*60}\n")


if __name__ == "__main__":
    # 测试代码
    import numpy as np
    
    # 创建测试数据
    np.random.seed(42)
    train_df = pd.DataFrame({
        'temp': np.random.normal(20, 5, 100),
        'humidity': np.random.normal(60, 10, 100),
        'pressure': np.random.normal(1013, 5, 100)
    })
    
    test_df = pd.DataFrame({
        'temp': np.random.normal(22, 5, 20),
        'humidity': np.random.normal(65, 10, 20),
        'pressure': np.random.normal(1015, 5, 20)
    })
    
    # 初始化标准化器
    scaler = NumericalScaler(['temp', 'humidity', 'pressure'], method='standard')
    
    # 拟合训练集
    train_scaled = scaler.fit_transform(train_df)
    
    print("训练集标准化前:")
    print(train_df.describe())
    
    print("\n训练集标准化后:")
    print(train_scaled.describe())
    
    # 转换测试集（使用训练集的参数）
    test_scaled = scaler.transform(test_df)
    
    print("\n测试集标准化后:")
    print(test_scaled.describe())
    
    # 打印统计信息
    scaler.print_statistics()
