import pandas as pd
import re

# Load .xls file (requires 'xlrd' package)
# Install with: pip install xlrd
df = pd.read_excel("Input_File.xls", engine="xlrd")

# Extract well identifiers (e.g., B2, C3, etc.)
df['Well'] = df['Name'].str.extract(r'\b([A-H]\d{1,2})\b')

# Store blocks of data
blocks = []

i = 0
while i < len(df):
    row = df.iloc[i]
    name = str(row['Name'])

    if name.endswith("mScarlett pos"):
        block = {}
        block['Well'] = row['Well']
        block['mScar +'] = row['Statistic']
        block['mScar cells'] = row['#Cells']

        # Try to find Mean in next few rows
        found_mean = False
        for j in range(i + 1, min(i + 4, len(df))):
            mean_row = df.iloc[j]
            mean_str = str(mean_row['Name'])
            match = re.search(r"Mean\s*:\s*YL1-A\s*=\s*([\d\.]+)", mean_str)
            if match:
                block['Mean'] = float(match.group(1))
                found_mean = True
                break

        if not found_mean:
            block['Mean'] = None  # Leave blank if not found

        # Search for corresponding GFP neg row
        for k in range(i + 1, min(i + 10, len(df))):
            gfp_row = df.iloc[k]
            gfp_name = str(gfp_row['Name'])
            if gfp_name.endswith("GFP neg"):
                block['GFP -'] = gfp_row['Statistic']
                block['GFP - cells'] = gfp_row['#Cells']
                break

        blocks.append(block)
        i = k  # Skip ahead
    else:
        i += 1

# Convert to DataFrame
full_result = pd.DataFrame(blocks)
result = full_result[full_result['mScar +'] >= 40].copy()

removed = full_result[full_result['mScar +'] < 40]
removed.to_excel("Output_File_1.xlsx", index = False)

# Sort wells down columns
result['Row'] = result['Well'].str.extract(r'([A-H])')
result['Col'] = result['Well'].str.extract(r'(\d{1,2})').astype(int)
result['OriginalIndex'] = result.index  # Preserve input order

# Re-sort based on desired pattern
result = result.sort_values(by=['Col', 'Row', 'OriginalIndex']).drop(columns=['Row', 'Col', 'OriginalIndex'])

# Save final output
result.to_excel("Output_File_2.xlsx", index=False)

# === Second output: Wide format with all metrics and replicates (using filtered result) ===

# Add replicate count per well
result['Replicate'] = result.groupby('Well').cumcount() + 1

# Pivot to wide format
pivoted = result.pivot(index='Well', columns='Replicate')

# Flatten multi-index columns
pivoted.columns = [f"{metric} (rep {rep})" for metric, rep in pivoted.columns]
pivoted.reset_index(inplace=True)

# Sort wells B2, C2, ..., B3, C3, ...
pivoted['Row'] = pivoted['Well'].str.extract(r'([A-H])')
pivoted['Col'] = pivoted['Well'].str.extract(r'(\d{1,2})').astype(int)
pivoted = pivoted.sort_values(by=['Col', 'Row']).drop(columns=['Row', 'Col'])

# Save wide output
pivoted.to_excel("Output_File_3.xlsx", index=False)
