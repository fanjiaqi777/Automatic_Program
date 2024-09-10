import os
import pandas as pd
from scipy import stats
from tqdm import tqdm
import argparse

# 定义输入和输出路径
def get_paths(base_dir):
    paths = {
        'filtered_vcf_path': os.path.join(base_dir, 'RUN-step-pre-1.vcf'),
        'reordered_vcf_path': os.path.join(base_dir, 'RUN-step-pre-2.vcf'),
        'extracted_vcf_path': os.path.join(base_dir, 'RUN-step-pre-3.vcf'),
        'split_output_dir': os.path.join(base_dir, 'step1-sep'),
        'processed_output_dir': os.path.join(base_dir, 'step1-processed')
    }
    os.makedirs(paths['split_output_dir'], exist_ok=True)
    os.makedirs(paths['processed_output_dir'], exist_ok=True)
    return paths

# Step 1: 过滤VCF文件
def filter_vcf(vcf, maternal, paternal, min_dp, missing_rate, P_threshold, output_path):
    def parse_parent(maternal, paternal, line):
        samples = line.split("\t")
        index_m = samples.index(maternal)
        index_f = samples.index(paternal)

        index_min = min(index_m, index_f)
        index_max = max(index_m, index_f)

        samples.pop(index_max)
        samples.pop(index_min)

        samples.insert(9, maternal)
        samples.insert(9, paternal)

        sampleline = "\t".join(samples)
        return index_m, index_f, index_min, index_max, sampleline

    def format_dp(sam):
        try:
            dp = sam.split(":")[2]
        except IndexError:
            dp = 0
        dp = 0 if dp == "." else int(dp)
        if dp < min_dp:
            sam = "./.:0,0:0:.:.:.:0,0,0"
        gt = sam.split(":")[0]
        return sam, gt

    def segregation(gt_m, gt_f):
        gamete_m = gt_m.split("/")
        gamete_f = gt_f.split("/")
        ratio = {}
        for m in gamete_m:
            for f in gamete_f:
                zygote = "{}/{}".format(m, f)
                if zygote == "1/0":
                    zygote = "0/1"
                ratio[zygote] = ratio.get(zygote, 0) + 0.25
        return ratio

    def check_offspring(gt_m, gt_f, off):
        ratio = segregation(gt_m, gt_f)
        if not ratio.get(off.split(":")[0]):
            off = "./.:0,0:0:.:.:.:0,0,0"
        off_gt = off.split(":")[0]
        return off, off_gt

    def calc_exp(gt_m, gt_f, off_num):
        ratio = segregation(gt_m, gt_f)
        off_num = int(off_num)
        exp_00 = ratio.get("0/0", 0) * off_num
        exp_01 = ratio.get("0/1", 0) * off_num
        exp_11 = ratio.get("1/1", 0) * off_num
        return exp_00, exp_01, exp_11

    output = open(output_path, "w")
    with open(vcf) as fd:
        lines = fd.readlines()
        index_m = index_f = index_min = index_max = 0
        for line in tqdm(lines, desc="Processing VCF"):
            line = line.strip()
            if line.startswith("#"):
                if line.startswith("#CHROM"):
                    index_m, index_f, index_min, index_max, sampleline = parse_parent(maternal, paternal, line)
                    output.write(sampleline + "\n")
                else:
                    output.write(line + "\n")
                continue
            line_data = line.split("\t")
            tmp = line_data[:9]
            mother = line_data[index_m]
            father = line_data[index_f]
            mother, gt_m = format_dp(mother)
            father, gt_f = format_dp(father)
            if gt_m == "./." or gt_f == "./." or (gt_m == "0/0" and gt_f == "0/0") or (gt_m == "1/1" and gt_f == "1/1"):
                continue
            count = {}
            offsprings = line_data[9:index_min] + line_data[index_min + 1:index_max] + line_data[index_max + 1:]
            off_num = len(offsprings)
            for off in offsprings:
                off, off_gt = format_dp(off)
                off, off_gt = check_offspring(gt_m, gt_f, off)
                tmp.append(off)
                count[off_gt] = count.get(off_gt, 0) + 1
            num_miss = count.get("./.", 0)
            if num_miss / off_num > missing_rate:
                continue
            effect_off_num = off_num - num_miss
            exp_00, exp_01, exp_11 = calc_exp(gt_m, gt_f, effect_off_num)
            obs_tmp = [count.get(gt, 0) for gt in ["0/0", "0/1", "1/1"]]
            exp_tmp = [exp_00, exp_01, exp_11]
            non_zero = [i for i in range(3) if exp_tmp[i] != 0]
            exp = [exp_tmp[i] for i in non_zero]
            obs = [obs_tmp[i] for i in non_zero]
            chi, pvalue = stats.chisquare(obs, f_exp=exp)
            if pvalue > P_threshold:
                tmp.insert(9, mother)
                tmp.insert(9, father)
                newline = "\t".join(tmp)
                output.write(newline + "\n")
    output.close()
    print("Processing complete. Output saved to:", output_path)

# Step 2: 母本列移到前面
def reorder_columns(vcf_path, maternal):
    with open(vcf_path) as file:
        lines = file.readlines()
    header_lines = [line for line in lines if line.startswith('#')]
    data_lines = [line for line in lines if not line.startswith('#')]
    header = header_lines[-1].strip().split('\t')
    data = [line.strip().split('\t') for line in data_lines]
    df = pd.DataFrame(data, columns=header)
    maternal_index = header.index(maternal)
    columns = header[:9] + [header[maternal_index]] + header[9:maternal_index] + header[maternal_index+1:]
    df = df[columns]
    header_lines[-1] = '\t'.join(columns) + '\n'
    with open(vcf_path, 'w') as file:
        for line in header_lines:
            file.write(line)
        for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Writing final VCF"):
            file.write('\t'.join(row) + '\n')
    print("Reordering complete. Output saved to the same file:", vcf_path)

# Step 3: 提取基因型和有用的列
def extract_and_replace_genotypes(vcf_file, output_file):
    with open(vcf_file, 'r') as infile, open(output_file, 'w') as outfile:
        header = []
        samples = []
        metadata_lines = []
        lines = infile.readlines()
        for line in tqdm(lines, desc="Extracting and Replacing Genotypes"):
            if line.startswith('##'):
                metadata_lines.append(line)
            elif line.startswith('#'):
                header = line.strip().split('\t')
                samples = header[9:]
                if metadata_lines:
                    outfile.write(metadata_lines[-1])
                outfile.write('#CHROM\tPOS\tREF\tALT\t' + '\t'.join(samples) + '\n')
            else:
                columns = line.strip().split('\t')
                chrom = columns[0]
                pos = columns[1]
                ref = columns[3]
                alt = columns[4]
                genotypes = columns[9:]
                new_genotypes = []
                for genotype in genotypes:
                    gt = genotype.split(':')[0]
                    if gt in ['0/0', '0|0']:
                        new_genotypes.append(ref + ref)
                    elif gt in ['1/1', '1|1']:
                        new_genotypes.append(alt + alt)
                    elif gt in ['0/1', '0|1']:
                        new_genotypes.append(ref + alt)
                    elif gt in ['1/0', '1|0']:
                        new_genotypes.append(alt + ref)
                    else:
                        new_genotypes.append('-')
                new_line = '\t'.join([chrom, pos, ref, alt] + new_genotypes) + '\n'
                outfile.write(new_line)
    print("Extraction and replacement complete. Output saved to:", output_file)

# Step 4: 按染色体和支架分开
def split_vcf(file_path, output_dir, chrom_filter=None):
    data = []
    header = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('##'):
                continue  # 跳过注释行
            if line.startswith('#'):
                header = line.strip().split('\t')
            else:
                data.append(line.strip().split('\t'))

    # 将数据转换为DataFrame
    df = pd.DataFrame(data, columns=header)

    # 获取所有的染色体信息
    chromosomes = df['#CHROM'].unique()

    if chrom_filter:
        chromosomes = [chrom_filter] if chrom_filter in chromosomes else []

    # 分离出Scaffold和其他染色体信息
    scaffold_df = df[df['#CHROM'].str.startswith('Scaffold')]

    if chrom_filter and chrom_filter.startswith('Scaffold'):
        other_chromosomes = []
    else:
        other_chromosomes = [chrom for chrom in chromosomes if not chrom.startswith('Scaffold')]

    # 将Scaffold数据写入单独的Excel文件
    if not chrom_filter or chrom_filter.startswith('Scaffold'):
        scaffold_output_file = os.path.join(output_dir, 'Scaffold.xlsx')
        scaffold_df.to_excel(scaffold_output_file, index=False)
        print(f"Scaffold数据已成功写入 {scaffold_output_file}")

    # 将其他每个染色体的数据写入不同的Excel文件
    for chrom in other_chromosomes:
        chrom_df = df[df['#CHROM'] == chrom]
        chrom_output_file = os.path.join(output_dir, f'{chrom}.xlsx')
        chrom_df.to_excel(chrom_output_file, index=False)
        print(f"{chrom}数据已成功写入 {chrom_output_file}")

# Step 5: 处理每个分开的Excel文件
pd.options.mode.copy_on_write = True
def clean_and_replace(df, ref_col='ref', alt_col='alt'):
    df = df.replace("..", "-")
    def replace_values(row):
        ref = row[ref_col]
        alt = row[alt_col]
        for col in row.index[5:]:
            if pd.isnull(row[col]) or row[col] == '-':
                continue
            cell_value = row[col].replace("/", "")
            if cell_value == ref + ref:
                row[col] = 'A'
            elif cell_value == alt + alt:
                row[col] = 'B'
            else:
                row[col] = 'H'
        return row

    for idx in tqdm(df.index, desc="Cleaning and Replacing Data"):
        df.loc[idx] = replace_values(df.loc[idx])
    
    return df

def process_excel_files(input_dir, output_dir, chrom_filter=None):
    # 获取所有Excel文件
    input_files = [f for f in os.listdir(input_dir) if f.endswith('.xlsx')]

    if chrom_filter:
        input_files = [f for f in input_files if f.startswith(chrom_filter) or f.startswith('Scaffold')]

    for file in input_files:
        input_file_path = os.path.join(input_dir, file)
        output_file_path = os.path.join(output_dir, file.replace('.xlsx', '_processed.xlsx'))

        # 读取Excel文件
        df = pd.read_excel(input_file_path)

        # 添加新列 'POS-1'，其值为第一列和POS列的组合
        df.insert(1, 'POS-1', df['#CHROM'] + "_" + df['POS'].astype(str))

        # 修改列名
        new_columns = ['CHR', 'POS-1', 'POS-2', 'ref', 'alt', 'Female', 'Male'] + \
                      [f'Individual{i}' for i in range(1, len(df.columns) - 6)]
        df.columns = new_columns

        # 应用清理和替换函数到 DataFrame
        df_processed = clean_and_replace(df)

        # 保存处理后的Excel文件
        df_processed.to_excel(output_file_path, index=False)

        print(f"数据已成功写入 {output_file_path}")

# Step 6: 合并处理后的 Scaffold 文件
def get_scaffold_files(directory):
    files = [f for f in os.listdir(directory) if f.startswith('scaffold') and f.endswith('_processed.xlsx')]
    files.sort(key=lambda x: int(x.split('_')[1]))
    return files

def merge_scaffold_files(directory, output_file):
    files = get_scaffold_files(directory)
    merged_df = pd.DataFrame()
    first_file = True
    
    for file in files:
        file_path = os.path.join(directory, file)
        df = pd.read_excel(file_path)
        if first_file:
            merged_df = df
            first_file = False
        else:
            merged_df = pd.concat([merged_df, df.iloc[1:]], ignore_index=True)
    
    merged_df.to_excel(output_file, index=False)
    print(f"所有文件已成功合并并写入 {output_file}")

# 使用argparse解析命令行参数
def main():
    parser = argparse.ArgumentParser(description="VCF Processing Pipeline")
    parser.add_argument("-v", dest="vcf", required=True, help="input VCF file")
    parser.add_argument("-f", dest="paternal", type=str, required=True, help="paternal ID")
    parser.add_argument("-m", dest="maternal", type=str, required=True, help="maternal ID")
    parser.add_argument("-d", dest="min_dp", type=int, default=5, help="permitted minimum DP [5]")
    parser.add_argument("-r", dest="missing_rate", type=float, default=0.25, help="max-missing rate [0.25]")
    parser.add_argument("-p", dest="P_threshold", type=float, default=0.001, help="p-value threshold [0.001]")
    parser.add_argument("-o", dest="output_dir", type=str, default="step0", help="output directory [step0]")
    parser.add_argument("-x", dest="chrom_filter", type=str, help="Chromosome to filter")

    args = parser.parse_args()

    # 获取路径
    paths = get_paths(args.output_dir)

    # Step 1: 过滤VCF文件
    filter_vcf(args.vcf, args.maternal, args.paternal, args.min_dp, args.missing_rate, args.P_threshold, paths['filtered_vcf_path'])

    # Step 2: 母本列移到前面
    reorder_columns(paths['filtered_vcf_path'], args.maternal)
    if os.path.exists(paths['filtered_vcf_path']):
        os.rename(paths['filtered_vcf_path'], paths['reordered_vcf_path'])
    else:
        print(f"Error: The file {paths['filtered_vcf_path']} does not exist after reordering.")

    # Step 3: 提取基因型和有用的列
    extract_and_replace_genotypes(paths['reordered_vcf_path'], paths['extracted_vcf_path'])

    # Step 4: 按染色体和支架分开
    split_vcf(paths['extracted_vcf_path'], paths['split_output_dir'], args.chrom_filter)

    # Step 5: 处理每个分开的Excel文件
    process_excel_files(paths['split_output_dir'], paths['processed_output_dir'], args.chrom_filter)

    # Step 6: 合并处理后的 Scaffold 文件
    merge_scaffold_files(paths['processed_output_dir'], os.path.join(paths['processed_output_dir'], 'merged_scaffold_files.xlsx'))

    print("所有步骤完成，最终文件已生成。")

if __name__ == "__main__":
    main()
