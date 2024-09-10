import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import argparse
import os

# Apply downgrade function
def apply_downgrade(df, column_name, start_index, current_value, downgrade_value, range_distance):
    modified_count = 0
    last_modified_index = start_index

    for i in range(start_index, len(df)):
        if df.loc[i, column_name].upper() == current_value:
            if current_value == 'H':
                df.at[i, column_name] = 'a'
            elif current_value == 'A':
                df.at[i, column_name] = 'h'
            modified_count += 1
            last_modified_index = i
        else:
            break
    
    return modified_count, last_modified_index

# First stage logic
def validate_individual_with_complete_logic(column_name, df, range_distance):
    df_local = df.copy()
    modifications = []
    current_start_pos = None
    modified_cells_count = 0

    for i in tqdm(range(1, len(df_local)), desc=f"Processing {column_name}"):
        current_value = df_local.loc[i, column_name].upper()

        if current_value == '-':
            continue

        previous_value = None
        previous_pos = None
        for j in range(i - 1, -1, -1):
            if df_local.loc[j, column_name] != '-':
                previous_value = df_local.loc[j, column_name].upper()
                previous_pos = df_local.loc[j, 'POS-2']
                break

        if previous_value is None:
            continue

        start_pos = df_local.loc[i, 'POS-2']
        end_pos = start_pos + range_distance

        in_range_mask = (df_local['POS-2'] > start_pos) & (df_local['POS-2'] <= end_pos)
        in_range_values = df_local.loc[in_range_mask, column_name].str.upper()
        in_range_values = in_range_values[in_range_values != '-']

        h_count = in_range_values.value_counts().get('H', 0)
        a_count = in_range_values.value_counts().get('A', 0)
        total_count = h_count + a_count

        next_valid_index = df_local.loc[(df_local['POS-2'] > end_pos) & (df_local[column_name] != '-')].index.min()
        if next_valid_index and next_valid_index < len(df_local):
            next_value = df_local.loc[next_valid_index, column_name].upper()
            next_pos = df_local.loc[next_valid_index, 'POS-2']

            new_range_start = next_pos
            new_range_end = new_range_start + range_distance
            new_range_mask = (df_local['POS-2'] > new_range_start) & (df_local['POS-2'] <= new_range_end)
            new_range_values = df_local.loc[new_range_mask, column_name].str.upper()
            new_range_values = new_range_values[new_range_values != '-']

            new_h_count = new_range_values.value_counts().get('H', 0)
            new_a_count = new_range_values.value_counts().get('A', 0)

            if new_a_count >= new_h_count:
                next_value = 'A'
            else:
                next_value = 'H'

            if previous_value == 'A' and current_value == 'H':
                if total_count > 0 and (h_count >= a_count) and next_value == 'A':
                    modified_cells_count, last_modified_index = apply_downgrade(df_local, column_name, i, 'H', 'a', range_distance)
                    if modified_cells_count > 0:
                        current_start_pos = df_local.loc[i, 'POS-2']
                        if (df_local.loc[last_modified_index, 'POS-2'] - current_start_pos) > range_distance:
                            modified_cells_count, last_modified_index = apply_downgrade(df_local, column_name, i, 'a', 'H', range_distance)
                        else:
                            modifications.append({
                                'Individual': column_name,
                                'POS-2': f"{current_start_pos}-{df_local.loc[last_modified_index, 'POS-2']}",
                                'Modified Cells Count': modified_cells_count
                            })
                            df_local[column_name] = df_local[column_name].copy()
                            df_local = validate_individual_with_extended_logic_stage1(df_local, column_name, i, last_modified_index, range_distance)
                    else:
                        df_local = validate_individual_with_extended_logic_stage1(df_local, column_name, i, last_modified_index, range_distance)

            elif previous_value == 'H' and current_value == 'A':
                if total_count > 0 and (a_count >= h_count) and next_value == 'H':
                    modified_cells_count, last_modified_index = apply_downgrade(df_local, column_name, i, 'A', 'h', range_distance)
                    if modified_cells_count > 0:
                        current_start_pos = df_local.loc[i, 'POS-2']
                        if (df_local.loc[last_modified_index, 'POS-2'] - current_start_pos) > range_distance:
                            modified_cells_count, last_modified_index = apply_downgrade(df_local, column_name, i, 'h', 'A', range_distance)
                        else:
                            modifications.append({
                                'Individual': column_name,
                                'POS-2': f"{current_start_pos}-{df_local.loc[last_modified_index, 'POS-2']}",
                                'Modified Cells Count': modified_cells_count
                            })
                            df_local[column_name] = df_local[column_name].copy()
                            df_local = validate_individual_with_extended_logic_stage1(df_local, column_name, i, last_modified_index, range_distance)
                    else:
                        df_local = validate_individual_with_extended_logic_stage1(df_local, column_name, i, last_modified_index, range_distance)
                else:
                    df_local = validate_individual_with_extended_logic_stage1(df_local, column_name, i, next_valid_index, range_distance)
    
    return df_local[column_name], modifications

# Initialize modifications_summary_2 list to track downgrades in stage 2
modifications_summary_2 = []

# Third stage logic (from the first script, renamed to avoid confusion)
def validate_individual_with_extended_logic_stage1(df, column_name, start_index, end_index, range_distance):
    for i in tqdm(range(start_index, end_index + 1), desc=f"Processing {column_name}"):
        current_value = df.loc[i, column_name].upper()

        if current_value == '-':
            continue

        previous_value = None
        previous_pos = None
        for j in range(i - 1, -1, -1):
            if df.loc[j, column_name] != '-':
                previous_value = df.loc[j, column_name].upper()
                previous_pos = df.loc[j, 'POS-2']
                break

        if previous_value is None:
            continue

        if previous_value == 'H' and current_value == 'A':
            start_pos = df.loc[i, 'POS-2']
            end_pos = start_pos + range_distance
            in_range_mask = (df['POS-2'] > start_pos) & (df['POS-2'] <= end_pos)
            in_range_values = df.loc[in_range_mask, column_name].str.upper()
            in_range_values = in_range_values[in_range_values != '-']

            if in_range_values.empty:
                next_valid_index = df.loc[i+1:, column_name].ne('-').idxmax()
                in_range_values = pd.Series([df.loc[next_valid_index, column_name].upper()])

                if previous_pos is not None and (start_pos - previous_pos) > range_distance:
                    df.at[i, column_name] = '-'
                    continue

            a_count = in_range_values.value_counts().get('A', 0)
            total_count = in_range_values.isin(['A', 'H']).sum()
            if total_count == 0 or (a_count / total_count) < 0.5:
                df.at[i, column_name] = 'h'
            else:
                next_valid_index = i + 1
                while next_valid_index < len(df) and df.loc[next_valid_index, column_name] == '-':
                    next_valid_index += 1

                if next_valid_index < len(df):
                    if df.loc[next_valid_index, column_name].upper() == 'H':
                        df.at[i, column_name] = 'h'

        elif previous_value == 'A' and current_value == 'H':
            start_pos = df.loc[i, 'POS-2']
            end_pos = start_pos + range_distance
            in_range_mask = (df['POS-2'] > start_pos) & (df['POS-2'] <= end_pos)
            in_range_values = df.loc[in_range_mask, column_name].str.upper()
            in_range_values = in_range_values[in_range_values != '-']

            if in_range_values.empty:
                next_valid_index = df.loc[i+1:, column_name].ne('-').idxmax()
                in_range_values = pd.Series([df.loc[next_valid_index, column_name].upper()])

                if previous_pos is not None and (start_pos - previous_pos) > range_distance:
                    df.at[i, column_name] = '-'
                    continue

            h_count = in_range_values.value_counts().get('H', 0)
            total_count = in_range_values.isin(['A', 'H']).sum()
            if total_count == 0 or (h_count / total_count) < 0.5:
                df.at[i, column_name] = 'a'
    return df

# Second script's extended logic, with original function name
def validate_individual_with_extended_logic(df, column_name, range_distance):
    for i in tqdm(range(1, len(df)), desc=f"Processing {column_name}"):
        current_value = df.loc[i, column_name].upper()

        if current_value == '-':
            continue

        previous_value = None
        previous_pos = None
        for j in range(i - 1, -1, -1):
            if df.loc[j, column_name] != '-':
                previous_value = df.loc[j, column_name].upper()
                previous_pos = df.loc[j, 'POS-2']
                break

        if previous_value is None:
            continue

        if previous_value == 'H' and current_value == 'A':
            start_pos = df.loc[i, 'POS-2']
            end_pos = start_pos + range_distance
            in_range_mask = (df['POS-2'] > start_pos) & (df['POS-2'] <= end_pos)
            in_range_values = df.loc[in_range_mask, column_name].str.upper()
            in_range_values = in_range_values[in_range_values != '-']

            if in_range_values.empty:
                next_valid_index = df.loc[i+1:, column_name].ne('-').idxmax()
                in_range_values = pd.Series([df.loc[next_valid_index, column_name].upper()])

                if previous_pos is not None and (start_pos - previous_pos) > range_distance:
                    df.at[i, column_name] = '-'
                    continue
            
            a_count = in_range_values.value_counts().get('A', 0)
            total_count = in_range_values.isin(['A', 'H']).sum()
            if total_count == 0 or (a_count / total_count) < 0.5:
                df.at[i, column_name] = 'h'
                for k in range(i + 1, len(df)):
                    if df.loc[k, column_name].upper() == 'A':
                        df.at[k, column_name] = 'h'
                    else:
                        break
            else:
                next_valid_index = i + 1
                while next_valid_index < len(df) and df.loc[next_valid_index, column_name] == '-':
                    next_valid_index += 1

                if next_valid_index < len(df):
                    if df.loc[next_valid_index, column_name].upper() == 'H':
                        df.at[i, column_name] = 'h'

        elif previous_value == 'A' and current_value == 'H':
            start_pos = df.loc[i, 'POS-2']
            end_pos = start_pos + range_distance
            in_range_mask = (df['POS-2'] > start_pos) & (df['POS-2'] <= end_pos)
            in_range_values = df.loc[in_range_mask, column_name].str.upper()
            in_range_values = in_range_values[in_range_values != '-']

            if in_range_values.empty:
                next_valid_index = df.loc[i+1:, column_name].ne('-').idxmax()
                in_range_values = pd.Series([df.loc[next_valid_index, column_name].upper()])

                if previous_pos is not None and (start_pos - previous_pos) > range_distance:
                    df.at[i, column_name] = '-'
                    continue

            h_count = in_range_values.value_counts().get('H', 0)
            total_count = in_range_values.isin(['A', 'H']).sum()
            if total_count == 0 or (h_count / total_count) < 0.5:
                df.at[i, column_name] = 'a'
                for k in range(i + 1, len(df)):
                    if df.loc[k, column_name].upper() == 'H':
                        df.at[k, column_name] = 'a'
                    else:
                        break

    return df

# Function to generate modifications_summary_2
def generate_modifications_summary_2(df):
    modifications_summary_2 = []

    individual_columns = [col for col in df.columns if isinstance(col, str) and col.startswith('Individual')]

    for col in individual_columns:
        start_index = None
        current_letter = None
        count = 0

        for i in range(len(df)):
            value = df.loc[i, col]
            pos = df.loc[i, 'POS-2']

            if value.islower():
                if start_index is None:
                    start_index = pos
                    current_letter = value
                count += 1
            else:
                if start_index is not None:
                    end_index = df.loc[i-1, 'POS-2']
                    pos_range = f"{start_index}-{end_index}" if start_index != end_index else f"{start_index}"
                    pos_diff = end_index - start_index
                    modifications_summary_2.append({
                        'Individual': col,
                        'POS-2': pos_range,
                        'Modified Cells Count': count,
                        'Lowercase Letter Type': current_letter,
                        'Position Difference': pos_diff
                    })
                    start_index = None
                    current_letter = None
                    count = 0

        # Handle the case where the last characters in the column are lowercase
        if start_index is not None:
            end_index = df.loc[i, 'POS-2']
            pos_range = f"{start_index}-{end_index}" if start_index != end_index else f"{start_index}"
            pos_diff = end_index - start_index
            modifications_summary_2.append({
                'Individual': col,
                'POS-2': pos_range,
                'Modified Cells Count': count,
                'Lowercase Letter Type': current_letter,
                'Position Difference': pos_diff
            })

    return pd.DataFrame(modifications_summary_2)

# Process the data in parallel for both stages
def process_data_parallel(file_path, range_distance, n_jobs):
    df = pd.read_excel(file_path)
    individual_columns = [col for col in df.columns if isinstance(col, str) and col.startswith('Individual')]

    # Stage 1: Parallel processing using first stage logic
    stage1_results = Parallel(n_jobs=n_jobs)(
        delayed(validate_individual_with_complete_logic)(col, df, range_distance) for col in tqdm(individual_columns, desc="Processing Stage 1 in parallel")
    )

    modifications_summary = []
    
    # Collect and merge results from Stage 1
    for i, (processed_column, modifications) in enumerate(stage1_results):
        df[individual_columns[i]] = processed_column
        if modifications:
            modifications_summary.extend(modifications)

    # Stage 2: Parallel processing using second stage logic
    stage2_results = Parallel(n_jobs=n_jobs)(
        delayed(validate_individual_with_extended_logic)(df, col, range_distance) for col in tqdm(individual_columns, desc="Processing Stage 2 in parallel")
    )

    # Collect and merge results from Stage 2
    for i, processed_df in enumerate(stage2_results):
        df[individual_columns[i]] = processed_df[individual_columns[i]]

    # Generate modifications_summary_2
    modifications_summary_2 = generate_modifications_summary_2(df)

    return df, pd.DataFrame(modifications_summary), modifications_summary_2

# Main function to handle argument parsing
def main():
    parser = argparse.ArgumentParser(description="Process Excel data with parallel computation.")
    parser.add_argument('-i', '--input', required=True, help="Path to the input Excel file.")
    parser.add_argument('-o', '--output', help="Path to the output Excel file. Default: <input_file>_processed.xlsx")
    parser.add_argument('-d', '--distance', type=int, default=100000, help="Range distance parameter. Default: 100000")
    parser.add_argument('-n', '--n_jobs', type=int, help="Number of CPU cores to use. Default: all available cores.")
    
    args = parser.parse_args()

    input_file = args.input
    output_file = args.output or f"{os.path.splitext(input_file)[0]}_imputed_processed.xlsx"
    range_distance = args.distance
    n_jobs = args.n_jobs or -1  # Default to using all available cores

    # Run the data processing
    df_processed, modifications_df, modifications_df_2 = process_data_parallel(input_file, range_distance, n_jobs)

    # Save the final results to a new Excel file
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df_processed.to_excel(writer, index=False, sheet_name='Processed Data')
        modifications_df.to_excel(writer, index=False, sheet_name='Modifications Summary')
        modifications_df_2.to_excel(writer, index=False, sheet_name='Modifications Summary 2')

    print(f"数据处理完成，结果已保存到 {output_file}")

if __name__ == "__main__":
    main()

