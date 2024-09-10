import pandas as pd
import numpy as np
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from tqdm import tqdm
import argparse
import os

def process_step1(input_file, output_file):
    # 读取 Excel 文件，并按 'POS-2' 排序
    Input = pd.read_excel(input_file, sheet_name=0)
    Input = Input.sort_values(by='POS-2').reset_index(drop=True)

    # 一次性计算所有符号的数量
    symbol_columns = Input.columns[7:]  # 假设符号从第8列开始
    for symbol in ['A', 'H', 'B', '-']:
        Input[f'{symbol}_Count'] = (Input[symbol_columns] == symbol).sum(axis=1)

    # 定义并计算 f(A) 和 f(H)
    Input['f(A)'] = Input.apply(lambda row: (2*row['A_Count']+row['H_Count']) / (2*(row['A_Count']+row['H_Count']+row['B_Count'])) if (row['A_Count']+row['H_Count']+row['B_Count']) != 0 else np.nan, axis=1)
    Input['f(H)'] = Input.apply(lambda row: row['H_Count'] / (row['A_Count']+row['H_Count']+row['B_Count']) if (row['A_Count']+row['H_Count']+row['B_Count']) != 0 else np.nan, axis=1)

    # 创建一个新的 write_only Workbook
    wb = Workbook(write_only=True)
    ws = wb.create_sheet("step1")

    # 将DataFrame转换为行数据，并一次性添加到工作表，使用tqdm显示进度
    rows = list(dataframe_to_rows(Input, index=False, header=True))
    for r in tqdm(rows, desc="Writing rows"):
        ws.append(r)

    # 保存到新文件
    wb.save(output_file)

def main():
    parser = argparse.ArgumentParser(description='Process genetic data in Excel files.')
    parser.add_argument('-i', '--input', required=True, help='Input Excel file')
    parser.add_argument('-o', '--output', help='Output Excel file')
    args = parser.parse_args()

    input_file = args.input
    output_file = args.output

    if not output_file:
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_for_step1{ext}"

    process_step1(input_file, output_file)

if __name__ == "__main__":
    main()
