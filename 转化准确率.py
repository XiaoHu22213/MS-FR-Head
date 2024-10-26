import numpy as np
import pandas as pd
import argparse

def convert_npy_to_csv(input_file, output_file):
    # 加载 .npy 文件
    data = np.load(input_file)

    # 创建一个 DataFrame
    df = pd.DataFrame(data)

    # 重命名列
    df.columns = [f'Class_{i}' for i in range(df.shape[1])]

    # 添加样本 ID 列
    df.insert(0, 'Sample_ID', range(len(df)))

    # 将所有浮点数列保留两位小数
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].round(2)

    # 保存为 CSV 文件
    df.to_csv(output_file, index=False, float_format='%.2f')
    print(f"文件已保存为 {output_file}")

    # 打印一些基本统计信息
    print("\n基本统计信息:")
    print(f"样本数: {len(df)}")
    print(f"类别数: {df.shape[1] - 1}")  # 减去 Sample_ID 列
    print("\n前5行数据:")
    print(df.head().to_string(float_format='{:.2f}'.format))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='将 .npy 文件转换为 .csv 文件')
    parser.add_argument('--input_file', type=str, default='pred.npy', help='输入的 .npy 文件路径')
    parser.add_argument('--output_file', type=str, default='pred.csv', help='输出的 .csv 文件路径')

    args = parser.parse_args()

    convert_npy_to_csv(args.input_file, args.output_file)

