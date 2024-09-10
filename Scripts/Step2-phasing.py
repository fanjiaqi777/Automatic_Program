import argparse
import pandas as pd
from openpyxl import load_workbook
import numpy as np
from tqdm import tqdm  # 加入进度条库
import os

# 添加命令行参数
parser = argparse.ArgumentParser(description='Process an Excel file.')
parser.add_argument('-i', '--input', required=True, help='Path to the input Excel file')
parser.add_argument('-d', '--distance', type=int, default=100000, help='Distance for bounds calculation')
parser.add_argument('-e', '--error_threshold', type=int, default=10, help='Threshold for errors')
parser.add_argument('-o', '--output', help='Path to the output Excel file (optional). If not provided, input file will be overwritten.')

args = parser.parse_args()

error_threshold = args.error_threshold

# 获取输入文件路径
file_path = args.input

# 如果用户指定了 -o，则使用指定的输出文件，否则覆盖原始文件
if args.output:
    output_file_path = args.output
else:
    output_file_path = file_path

# 读取默认的第一个表格
df = pd.read_excel(file_path, sheet_name=0)

# 按第一列升序排序
df_sorted = df.sort_values(by=df.columns[0], ascending=True)

# 计算第二列中每个染色体标记的频率
chromosome_counts = df_sorted.iloc[:, 1].value_counts()
most_common_chromosome = chromosome_counts.idxmax()

# 过滤掉不属于最常见染色体标记的行
df_not_most_common = df_sorted[df_sorted.iloc[:, 1] != most_common_chromosome]

# 保留属于最常见染色体标记的行
df_sorted = df_sorted[df_sorted.iloc[:, 1] == most_common_chromosome]

# 分析第六列前100行中 'A' 和 'H' 的频率
letter_counts = df_sorted.iloc[:100, 5].value_counts()

# 默认升序排序当 'A' 是多数
sort_ascending = True

if letter_counts.get('H', 0) > letter_counts.get('A', 0):
    sort_ascending = False

# 执行排序
df_sorted = df_sorted.sort_values(by=[df_sorted.columns[5], df_sorted.columns[0]],
                                  ascending=[sort_ascending, True])

# 找到不再递增的起始行，并且第六列的值发生了变化
start_row = None
for i in range(1, len(df_sorted)):
    if df_sorted.iloc[i, 0] < df_sorted.iloc[i-1, 0]:
        prev_value = df_sorted.iloc[i-1, 5]
        curr_value = df_sorted.iloc[i, 5]
        if (prev_value == 'A' and curr_value == 'H') or (prev_value == 'H' and curr_value == 'A'):
            start_row = i
            break

# 优化字母替换操作
if start_row is not None:
    df_sorted.iloc[start_row:, 5:] = df_sorted.iloc[start_row:, 5:].replace({'A': 'K', 'H': 'A'})
    df_sorted.iloc[start_row:, 5:] = df_sorted.iloc[start_row:, 5:].replace({'K': 'H'})

# 重新排序
df_sorted = df_sorted.sort_values(by=df_sorted.columns[0])

# 删除旧的计数列和功能列（如果存在）
for col in ['A_Count', 'B_Count', 'H_Count', '-_Count', 'f(A)', 'f(H)']:
    if col in df_sorted.columns:
        df_sorted.drop(col, axis=1, inplace=True)

# 重新计算计数列
def count_letters_from_sixth_column(row, letter):
    return (row.iloc[5:] == letter).sum()

df_sorted['A_Count'] = df_sorted.apply(lambda row: count_letters_from_sixth_column(row, 'A'), axis=1)
df_sorted['B_Count'] = df_sorted.apply(lambda row: count_letters_from_sixth_column(row, 'B'), axis=1)
df_sorted['H_Count'] = df_sorted.apply(lambda row: count_letters_from_sixth_column(row, 'H'), axis=1)
df_sorted['-_Count'] = df_sorted.apply(lambda row: count_letters_from_sixth_column(row, '-'), axis=1)

df_sorted['f(A)'] = df_sorted.apply(
    lambda row: (2 * row['A_Count'] + row['H_Count']) / (2 * (row['A_Count'] + row['B_Count'] + row['H_Count']))
    if (row['A_Count'] + row['B_Count'] + row['H_Count']) > 0 else np.nan, axis=1)
df_sorted['f(H)'] = df_sorted.apply(
    lambda row: row['H_Count'] / (row['A_Count'] + row['B_Count'] + row['H_Count'])
    if (row['A_Count'] + row['B_Count'] + row['H_Count']) > 0 else np.nan, axis=1)

# 保存到新的表格 "Parent1-1" 和 "DSC.1"
with pd.ExcelWriter(output_file_path, engine='openpyxl', mode='w') as writer:
    df_sorted.to_excel(writer, sheet_name="Parent1-1", index=False)
    df_not_most_common.to_excel(writer, sheet_name="DSC.1", index=False)

# 继续后续处理...
sheet_name = "Parent1-1"
df = pd.read_excel(output_file_path, sheet_name=sheet_name)

# 计算两个行之间的差异
def compute_difference(row1, row2):
    return sum(a != b for a, b in zip(row1, row2))

# 翻转字母
def flip_letters(s):
    return s.replace({'A': 'H', 'H': 'A'})

optimized_rows = [df.iloc[0]]

for i in tqdm(range(1, len(df)), desc="Optimizing rows"):
    current_row = df.iloc[i]
    prev_row = optimized_rows[-1]
    
    diff_with_current = compute_difference(prev_row[5:], current_row[5:])
    flipped_current_row = flip_letters(current_row[5:])
    diff_with_flipped = compute_difference(prev_row[5:], flipped_current_row)
    
    if diff_with_flipped < diff_with_current:
        optimized_rows.append(current_row.apply(lambda x: 'H' if x == 'A' else 'A' if x == 'H' else x))
    else:
        optimized_rows.append(current_row)

optimized_df = pd.DataFrame(optimized_rows).reset_index(drop=True)

# 保存 "Parent1-2"
with pd.ExcelWriter(output_file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    optimized_df.to_excel(writer, sheet_name="Parent1-2", index=False)


# 假设 df 是 DataFrame，并且每个单元格只包含字母 A 或 H
def similarity(row1, row2):
    return sum(a == b for a, b in zip(row1, row2))

def flip_row(row):
    return ['H' if cell == 'A' else 'A' if cell == 'H' else cell for cell in row]

optimized_rows = [df.iloc[0, 5:].tolist()]

for i in tqdm(range(1, len(df)), desc="Flipping rows"):
    current_row = df.iloc[i, 5:].tolist()
    prev_optimized_row = optimized_rows[-1]
    
    sim_without_flip = similarity(prev_optimized_row, current_row)
    flipped_row = flip_row(current_row)
    sim_with_flip = similarity(prev_optimized_row, flipped_row)
    
    if sim_with_flip > sim_without_flip:
        optimized_rows.append(flipped_row)
    else:
        optimized_rows.append(current_row)

for i, row in enumerate(optimized_rows, start=0):
    df.iloc[i, 5:] = row

# 保存 "Parent1-3"
with pd.ExcelWriter(output_file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    df.to_excel(writer, sheet_name="Parent1-3", index=False)


# 继续后续处理...
sheet_name = "Parent1-3"
df = pd.read_excel(output_file_path, sheet_name=sheet_name)

total_cells = df.iloc[0, 5:].apply(lambda x: x in ['A', 'B', 'H', '-']).sum()

final_df = df

def find_bounds_for_all_rows(df, distance):
    bounds = {}
    for index, row in df.iterrows():
        pos = row['POS-2']
        upper_bound = df[df['POS-2'] <= pos - distance].index.max() if not df[df['POS-2'] <= pos - distance].empty else None
        lower_bound = df[df['POS-2'] >= pos + distance].index.min() if not df[df['POS-2'] >= pos + distance].empty else None
        bounds[index] = (upper_bound, lower_bound)
    return bounds


def validate_rows_with_bounds(df, bounds):
    error_counts = []
    for index, row in tqdm(df.iterrows(), desc="Validating rows", total=len(df)):
        error_count = 0
        upper_bound_index, lower_bound_index = bounds.get(index, (None, None))
        if upper_bound_index is None or lower_bound_index is None:
            error_counts.append(error_count)
            continue
        for col in range(5, df.shape[1] - 4):
            cell = row.iloc[col]
            if cell == '-':
                continue
            upper_cell = df.iloc[upper_bound_index, col] if upper_bound_index else None
            lower_cell = df.iloc[lower_bound_index, col] if lower_bound_index else None
            if upper_cell == '-' or lower_cell == '-':
                continue
            if cell != upper_cell and cell != lower_cell:
                error_count += 1
        error_counts.append(error_count)
    return error_counts

bounds = find_bounds_for_all_rows(final_df,args.distance)
error_counts = validate_rows_with_bounds(final_df, bounds)
final_df['Error_Count'] = error_counts

# 避免错误的函数
def find_bounds_for_all_rows_avoiding_errors(df, bounds, error_threshold=10):
    new_bounds = {}
    for index in bounds:
        upper_bound_index, lower_bound_index = bounds[index]
        while upper_bound_index is not None and df.at[upper_bound_index, 'Error_Count'] > error_threshold:
            upper_bound_index -= 1
            if upper_bound_index < 0:
                upper_bound_index = None
                break
        while lower_bound_index is not None and df.at[lower_bound_index, 'Error_Count'] > error_threshold:
            lower_bound_index += 1
            if lower_bound_index >= len(df):
                lower_bound_index = None
                break
        new_bounds[index] = (upper_bound_index, lower_bound_index)
    return new_bounds

def reevaluate_error_counts(df, bounds):
    error_counts = []
    for index, row in tqdm(df.iterrows(), desc="Reevaluating error counts", total=len(df)):
        upper_bound_index, lower_bound_index = bounds.get(index, (None, None))
        error_count = 0
        if upper_bound_index is None or lower_bound_index is None:
            error_counts.append(error_count)
            continue
        for col in range(5, df.shape[1] - 4):
            cell = row.iloc[col]
            if cell == '-' or df.iloc[upper_bound_index, col] == '-' or df.iloc[lower_bound_index, col] == '-':
                continue
            if cell != df.iloc[upper_bound_index, col] and cell != df.iloc[lower_bound_index, col]:
                error_count += 1
        error_counts.append(error_count)
    return error_counts

high_error_indices = final_df[final_df['Error_Count'] > args.error_threshold].index  # 这里也需要改动
bounds = find_bounds_for_all_rows(final_df, args.distance)
new_bounds = find_bounds_for_all_rows_avoiding_errors(final_df, bounds, error_threshold=args.error_threshold)  # 传递参数
new_error_counts = reevaluate_error_counts(final_df, new_bounds)

for index, error_count in zip(final_df.index, new_error_counts):
    final_df.at[index, 'Error_Count'] = error_count

# 保存 "Parent1-5" 数据
with pd.ExcelWriter(output_file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    final_df.to_excel(writer, sheet_name="Parent1-5", index=False)


# 继续后续处理...
sheet_name = "Parent1-3"
df = pd.read_excel(output_file_path, sheet_name=sheet_name)
total_cells = df.iloc[0, 5:].apply(lambda x: x in ['A', 'B', 'H', '-']).sum()
final_df = df


bounds = find_bounds_for_all_rows(final_df, args.distance)
error_counts = validate_rows_with_bounds(final_df, bounds)
final_df['Error_Count2'] = error_counts

# 避免错误
def find_bounds_for_all_rows_avoiding_errors(df, bounds, error_threshold=10):
    new_bounds = {}
    for index in bounds:
        upper_bound_index, lower_bound_index = bounds[index]
        while upper_bound_index is not None and df.at[upper_bound_index, 'Error_Count2'] > error_threshold:
            upper_bound_index -= 1
            if upper_bound_index < 0:
                upper_bound_index = None
                break
        while lower_bound_index is not None and df.at[lower_bound_index, 'Error_Count2'] > error_threshold:
            lower_bound_index += 1
            if lower_bound_index >= len(df):
                lower_bound_index = None
                break
        new_bounds[index] = (upper_bound_index, lower_bound_index)
    return new_bounds

new_bounds = find_bounds_for_all_rows_avoiding_errors(final_df, bounds)
new_error_counts = reevaluate_error_counts(final_df, new_bounds)

for index, error_count in zip(final_df.index, new_error_counts):
    final_df.at[index, 'Error_Count2'] = error_count

# 保存 "Parent1-6" 数据
with pd.ExcelWriter(output_file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    final_df.to_excel(writer, sheet_name="Parent1-6", index=False)


# 最后筛选
df_parent1_5 = pd.read_excel(output_file_path, sheet_name='Parent1-5')
df_parent1_6 = pd.read_excel(output_file_path, sheet_name='Parent1-6')

df_combined = pd.concat([df_parent1_5, df_parent1_6[['Error_Count2']]], axis=1)

# 使用 Pandas 的过滤方式代替循环删除
df_combined['Keep'] = (df_combined['Error_Count'] <= args.error_threshold) | (df_combined['Error_Count2'] <= args.error_threshold)
df_final = df_combined[df_combined['Keep']].drop(columns=['Error_Count2', 'Keep'])
df_deleted = df_combined[~df_combined['Keep']]

# 保存 "Parent1-7" 和 "DSC1.1"
with pd.ExcelWriter(output_file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    df_final.to_excel(writer, sheet_name='Parent1-7', index=False)
    df_deleted.to_excel(writer, sheet_name='DSC1.1', index=False)
