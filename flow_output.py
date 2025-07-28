import pandas as pd
from pathlib import Path

# Input files
input_files = [
    "File1.xlsx",
    "File2.xlsx",
    "File3.xlsx",
    "File4.xlsx",
    "File5.xlsx",
    "File6.xlsx",
    "File7.xlsx",
    "File8.xlsx",
    "File9.xlsx",
    "File10.xlsx"
]

metrics = ['mScar +', 'Mean', 'GFP -']
all_data = []

for file in input_files:
    df = pd.read_excel(file)

    # Check required column
    if 'Sample name' not in df.columns:
        print(f"Skipping {file}: 'Sample name' column not found.")
        continue

    # Prepare new dataframe to hold averages
    averaged_df = pd.DataFrame()
    averaged_df['Sample name'] = df['Sample name']

    for metric in metrics:
        # Find replicate columns (e.g., "mScar + (rep 1)", "mScar + (rep 2)", etc.)
        rep_cols = [col for col in df.columns if col.startswith(metric)]
        if not rep_cols:
            print(f"No replicate columns found for '{metric}' in {file}")
            continue

        # Row-wise average of the replicate columns
        averaged_df[metric] = df[rep_cols].mean(axis=1, skipna=True)

    # Add filename identifier
    averaged_df['Source File'] = Path(file).stem

    all_data.append(averaged_df)

# Combine into one output table
combined = pd.concat(all_data, ignore_index=True)

# Save to Excel
combined.to_excel("Per_File_Averaged_Replicates_Output.xlsx", index=False)
