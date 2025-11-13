import pandas as pd
import re
from pathlib import Path


def match_csv_rows_with_png_files(directory_path):
    """
    Match CSV rows with PNG files where CSV row's (id - 1) equals PNG's gen_digits
    """
    directory = Path(directory_path)

    # Find CSV and PNG files
    csv_files = list(directory.rglob("*gensummary.csv"))
    png_files = list(directory.rglob("*gen.png"))

    all_results = []

    for csv_file in csv_files:
        csv_filename = csv_file.name
        csv_prefix = re.sub(r'_gensummary\.csv$', '', csv_filename)

        # Find matching PNG files
        matching_pngs = [
            png for png in png_files
            if png.name.startswith(csv_prefix)
        ]

        # Read the CSV file
        try:
            csv_df = pd.read_csv(csv_file)

            # Create idd column (id - 1)
            if 'id' in csv_df.columns:
                csv_df['idd'] = csv_df['id'] - 1
            else:
                print(f"Warning: No 'id' column found in {csv_filename}")
                continue

            # Extract CSV components
            csv_components = extract_csv_components(csv_filename)

            # Process each matching PNG
            for png_file in matching_pngs:
                png_filename = png_file.name

                # Extract digits before '_gen.png'
                gen_match = re.search(r'(\d+)_gen\.png$', png_filename)
                if gen_match:
                    gen_digits = int(gen_match.group(1))

                    # Find CSV row where id - 1 == gen_digits
                    # This means we need id == gen_digits + 1
                    target_id = gen_digits + 1
                    matching_rows = csv_df[csv_df['id'] == target_id]

                    if not matching_rows.empty:
                        # Get the first matching row
                        csv_row = matching_rows.iloc[0]
                        csv_row_index = matching_rows.index[0]

                        # Verify the match
                        if csv_row['idd'] == gen_digits:
                            # Create combined result
                            result_row = {
                                'csv_filename': csv_filename,
                                'png_filename': png_filename,
                                'csv_row_index': csv_row_index,
                                'csv_id': int(csv_row['id']),
                                'csv_idd': int(csv_row['idd']),
                                'gen_digits': gen_digits,
                                'match_verified': True,
                                'prefix_match': csv_prefix,
                                **csv_components
                            }

                            # Add all CSV columns with 'csv_' prefix
                            for col in csv_df.columns:
                                result_row[f'csv_{col}'] = csv_row[col]

                            all_results.append(result_row)
                        else:
                            # Match verification failed
                            result_row = {
                                'csv_filename': csv_filename,
                                'png_filename': png_filename,
                                'csv_row_index': csv_row_index,
                                'csv_id': int(csv_row['id']),
                                'csv_idd': int(csv_row['idd']),
                                'gen_digits': gen_digits,
                                'match_verified': False,
                                'error': f'Match verification failed: idd={csv_row["idd"]} != gen_digits={gen_digits}',
                                **csv_components
                            }
                            all_results.append(result_row)
                    else:
                        # No row found with matching id
                        result_row = {
                            'csv_filename': csv_filename,
                            'png_filename': png_filename,
                            'csv_row_index': None,
                            'csv_id': None,
                            'csv_idd': None,
                            'gen_digits': gen_digits,
                            'match_verified': False,
                            'error': f'No CSV row found with id={target_id} (needed for gen_digits={gen_digits})',
                            'prefix_match': csv_prefix,
                            **csv_components
                        }
                        all_results.append(result_row)

        except Exception as e:
            print(f"Error reading CSV {csv_filename}: {e}")

    return pd.DataFrame(all_results)


def extract_csv_components(filename):
    """Extract components from CSV filename"""
    pattern = r'patch_(\d+).*?step_(\d+).*?batch(\d+)'
    match = re.search(pattern, filename)

    if match:
        patch_num, step_num, batch_num = match.groups()
        return {
            'patch_full': f'patch_{patch_num}',
            'patch_number': int(patch_num),
            'step_full': f'step_{step_num}',
            'step_number': int(step_num),
            'batch_full': f'batch{batch_num}',
            'batch_number': int(batch_num)
        }
    return {
        'patch_full': None, 'patch_number': None,
        'step_full': None, 'step_number': None,
        'batch_full': None, 'batch_number': None
    }


def extractPair(threshold=0.01,tfl='/devb/sar2opt_diff/sar2opt_diff_sar/fltpairs_sar2.csv'):
    rpath = "/devb/sar2opt_diff/sar2opt_diff_sar"
    df = match_csv_rows_with_png_files(rpath)
    df_filtered = df[df['csv_ssim'] >= threshold]
    # Display results
    print(f"Total matches found: {len(df_filtered)}")
    print("\nSample results:")
    print(df_filtered[['csv_filename', 'png_filename', 'csv_id', 'csv_idd', 'gen_digits', 'match_verified']].head())
    df_filtered.to_csv(tfl, index=False)

    # Check for any failed matches
    failed_matches = df_filtered[df_filtered['match_verified'] == False]
    if not failed_matches.empty:
        print(f"\nFailed matches: {len(failed_matches)}")
        print(failed_matches[['csv_filename', 'png_filename', 'error']].head())



if __name__ == "__main__":
    extractPair()
