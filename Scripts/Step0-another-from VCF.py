import os
import argparse
import pandas as pd
from tqdm import tqdm

# 第一步：提取基因型和有用的列
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

# 第二步：筛选父本和母本都不是"-"的行
def filter_vcf(input_file, output_file):
    vcf_data = pd.read_csv(input_file, sep='\t', comment='#', header=None)
    with open(input_file) as file:
        for line in file:
            if line.startswith("#CHROM"):
                header = line.strip().split("\t")
                break
    vcf_data.columns = header

    # 筛选数据
    filtered_vcf_data = vcf_data[(vcf_data[header[4]] != '-') & (vcf_data[header[5]] != '-')]

    filtered_vcf_data.to_csv(output_file, sep='\t', index=False)
    print(f'筛选后的VCF文件已保存为 {output_file}')

# 第三步：按染色体和支架分开
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
    
    df = pd.DataFrame(data, columns=header)

    chromosomes = df['#CHROM'].unique()
    if chrom_filter:
        chromosomes = [chrom_filter] if chrom_filter in chromosomes else []
    scaffold_df = df[df['#CHROM'].str.startswith('Scaffold')]
    if chrom_filter and chrom_filter.startswith('Scaffold'):
        other_chromosomes = []
    else:
        other_chromosomes = [chrom for chrom in chromosomes if not chrom.startswith('Scaffold')]
    if not chrom_filter or chrom_filter.startswith('Scaffold'):
        scaffold_output_file = os.path.join(output_dir, 'Scaffold.xlsx')
        scaffold_df.to_excel(scaffold_output_file, index=False)
        print(f"Scaffold数据已成功写入 {scaffold_output_file}")
    for chrom in tqdm(other_chromosomes, desc="Writing chromosomes to Excel"):
        chrom_df = df[df['#CHROM'] == chrom]
        chrom_output_file = os.path.join(output_dir, f'{chrom}.xlsx')
        chrom_df.to_excel(chrom_output_file, index=False)
        print(f"{chrom}数据已成功写入 {chrom_output_file}")

# 第四步：处理每个分开的Excel文件
# 设置 Copy-on-Write 模式
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
    input_files = [f for f in os.listdir(input_dir) if f.endswith('.xlsx')]
    if chrom_filter:
        input_files = [f for f in input_files if f.startswith(chrom_filter) or f.startswith('Scaffold')]
    for file in tqdm(input_files, desc="Processing Excel files"):
        input_file_path = os.path.join(input_dir, file)
        output_file_path = os.path.join(output_dir, file.replace('.xlsx', '_processed.xlsx'))
        df = pd.read_excel(input_file_path)
        df.insert(1, 'POS-1', df['#CHROM'] + "_" + df['POS'].astype(str))
        new_columns = ['CHR', 'POS-1', 'POS-2', 'ref', 'alt', 'Female', 'Male'] + \
                      [f'Individual{i}' for i in range(1, len(df.columns) - 6)]
        df.columns = new_columns
        df_processed = clean_and_replace(df)
        df_processed.to_excel(output_file_path, index=False)
        print(f"数据已成功写入 {output_file_path}")

# 第五步：合并Scaffold的处理文件
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

# 主函数调用顺序
def main(args):
    vcf_file = args.input_file
    extracted_vcf = os.path.join(args.output_dir, 'extracted_genotypes.vcf')
    filtered_vcf = os.path.join(args.output_dir, 'filtered_genotypes.vcf')
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)
    
    extract_and_replace_genotypes(vcf_file, extracted_vcf)
    filter_vcf(extracted_vcf, filtered_vcf)
    split_vcf(filtered_vcf, output_dir, args.chrom_filter)
    process_excel_files(output_dir, output_dir, args.chrom_filter)
    
    # 调用合并函数
    merge_scaffold_files(output_dir, os.path.join(output_dir, 'merged_scaffold_files.xlsx'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VCF Processing Script")
    parser.add_argument("-i", dest="input_file", type=str, required=True, help="Input VCF file")
    parser.add_argument("-o", dest="output_dir", type=str, default="step0", help="Output directory [default: step0]")
    parser.add_argument("-x", dest="chrom_filter", type=str, default=None, help="Specific chromosome to extract, e.g., Pp01 [default: all]")

    args = parser.parse_args()
    print("\nThis program performs the following steps:")
    print("1. Extract and replace genotypes from the VCF file.")
    print("2. Filter rows where both parents are not '-'.")
    print("3. Split the filtered VCF file into separate Excel files based on chromosomes.")
    print("4. Process each Excel file to clean and replace values.")
    print("5. Merge processed scaffold files into a single Excel file.\n")

    main(args)
