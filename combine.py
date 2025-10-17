import os
import pandas as pd

def combine_csv_files(folder_path, output_file="combined_data.csv"):
    """
    Combine all CSV files in a given folder into one file.
    Automatically skips NSRDB-style metadata rows and starts reading from 'Year,Month,Day,...' row.
    """
    all_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(".csv")
    ]

    if not all_files:
        print("No CSV files found in the folder.")
        return

    combined_df = []

    for file in all_files:
        print(f"Reading file: {os.path.basename(file)}")

        try:
            # Find where the actual data starts (line containing 'Year')
            with open(file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            data_start = next(
                (i for i, line in enumerate(lines) if line.strip().startswith("Year,")), 0
            )

            # Read CSV from the detected header line
            df = pd.read_csv(file, skiprows=data_start)
            df["Source_File"] = os.path.basename(file)  # optional: track file origin
            combined_df.append(df)

        except Exception as e:
            print(f"Error reading {file}: {e}")

    # Combine all DataFrames
    final_df = pd.concat(combined_df, ignore_index=True)
    print(f"Combined {len(all_files)} files with {len(final_df)} total rows.")

    # Save result
    if output_file.lower().endswith(".xlsx"):
        final_df.to_excel(output_file, index=False)
    else:
        final_df.to_csv(output_file, index=False)

    print(f"Saved combined file as: {output_file}")
    return final_df


# Example usage:
# combine_csv_files(r"D:\Data\NSRDB_Files", "merged_nsrdb.csv")


# Example usage:
combine_csv_files(r"D:\Research Paper\Solar FYP\Project 1\14 year Solar Data", "merged_nsrdb.csv")
