import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
import numpy as np
import os
from itertools import combinations

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

def plot_gfp_rep_comparison(df, filename_prefix, r2_records):
    from sklearn.linear_model import LinearRegression
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    import plotly.graph_objs as go
    from plotly.subplots import make_subplots
    import numpy as np

    reps = ['GFP - (rep 1)', 'GFP - (rep 2)', 'GFP - (rep 3)']
    if 'Sample name' not in df.columns or not all(col in df.columns for col in reps):
        print(f"Skipping {filename_prefix}: required columns missing")
        return

    sample_names = df['Sample name'].astype(str).fillna("Unknown")

    # Define pairs where Y is lower-numbered rep
    rep_pairs = [
        ('GFP - (rep 2)', 'GFP - (rep 1)'),
        ('GFP - (rep 3)', 'GFP - (rep 2)'),
        ('GFP - (rep 3)', 'GFP - (rep 1)')
    ]
    labels = ['Rep 2 vs Rep 1', 'Rep 3 vs Rep 2', 'Rep 3 vs Rep 1']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    r2_values = []

    for ax, (x_col, y_col), title in zip(axes, rep_pairs, labels):
        x = df[x_col]
        y = df[y_col]
        valid = x.notna() & y.notna()
        x = x[valid]
        y = y[valid]
        names = sample_names[valid]

        model = LinearRegression()
        model.fit(x.values.reshape(-1, 1), y.values)
        y_pred = model.predict(x.values.reshape(-1, 1))
        r2 = model.score(x.values.reshape(-1, 1), y.values)
        m = model.coef_[0]
        b = model.intercept_

        r2_values.append(r2)
        r2_records.append({
            "File Name": filename_prefix,
            "Comparison": title,
            "R²": round(r2, 4),
            "Slope (m)": round(m, 4),
            "Intercept (b)": round(b, 4)
        })

        ax.scatter(x, y, alpha=0.7)
        ax.plot(x, y_pred, color='red', linewidth=1.5)
        ax.set_title(f"{title}\ny = {m:.2f}x + {b:.2f}, R² = {r2:.3f}")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    avg_r2 = np.mean(r2_values)
    plt.suptitle(f"{filename_prefix} | Avg R² = {avg_r2:.4f}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    png_path = f"{filename_prefix}_scatter.png"
    plt.savefig(png_path, dpi=300)
    plt.close()
  
    # === Interactive HTML ===
    plotly_fig = make_subplots(rows=1, cols=3, subplot_titles=labels)

    for i, (x_col, y_col) in enumerate(rep_pairs):
        x = df[x_col]
        y = df[y_col]
        valid = x.notna() & y.notna()
        x = x[valid]
        y = y[valid]
        names = sample_names[valid]

        model = LinearRegression()
        model.fit(x.values.reshape(-1, 1), y.values)
        y_pred = model.predict(x.values.reshape(-1, 1))
        r2 = model.score(x.values.reshape(-1, 1), y.values)
        m = model.coef_[0]
        b = model.intercept_

        trace = go.Scatter(
            x=x, y=y,
            mode='markers',
            text=names,
            hovertemplate="Sample: %{text}<br>X: %{x}<br>Y: %{y}<extra></extra>",
            marker=dict(size=6, color='blue'),
            name=labels[i]
        )

        # Best-fit line trace
        line_trace = go.Scatter(
            x=x,
            y=y_pred,
            mode='lines',
            line=dict(color='red'),
            name='Fit',
            showlegend=False
        )

        plotly_fig.add_trace(trace, row=1, col=i + 1)
        plotly_fig.add_trace(line_trace, row=1, col=i + 1)

        # Add annotation with equation and R²
        annotation_text = f"y = {m:.2f}x + {b:.2f}<br>R² = {r2:.3f}"
        plotly_fig.add_annotation(
            text=annotation_text,
            xref="x domain", yref="y domain",
            x=0.99, y=0.01,
            showarrow=False,
            font=dict(size=10),
            align='right',
            row=1, col=i + 1
        )

    plotly_fig.update_layout(
        height=400, width=1200,
        title_text=f"GFP - Replicate Comparison: {filename_prefix}",
        showlegend=False
    )

    html_path = f"{filename_prefix}_interactive.html"
    plotly_fig.write_html(html_path)

metrics = ['mScar +', 'Mean', 'GFP -']
all_data = []
r2_records = []

for file in input_files:
    df = pd.read_excel(file)

    filename_prefix = Path(file).stem
    plot_gfp_rep_comparison(df, filename_prefix, r2_records)

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
    averaged_df['Source File'] = filename_prefix

    all_data.append(averaged_df)

# Combine into one output table
combined = pd.concat(all_data, ignore_index=True)

# Save to Excel
combined.to_excel("Per_File_Averaged_Replicates_Output.xlsx", index=False)

# Save R² summary for all files
r2_df = pd.DataFrame(r2_records)
r2_df.to_csv("R2_Summary_All_Files.csv", index=False)

# === Cross-file comparison settings ===
cross_r2_records = []

def compare_sources_and_plot(df, group_label):
    group_sources = df['Source File'].unique()

    for source1, source2 in combinations(group_sources, 2):
        df1 = df[df['Source File'] == source1][['Sample name', 'GFP -']].rename(columns={'GFP -': f'{source1}'})
        df2 = df[df['Source File'] == source2][['Sample name', 'GFP -']].rename(columns={'GFP -': f'{source2}'})

        # Inner join on Sample name
        merged = pd.merge(df1, df2, on='Sample name')
        if merged.empty:
            continue

        x = merged[f'{source1}']
        y = merged[f'{source2}']
        names = merged['Sample name']

        # Linear regression
        model = LinearRegression()
        model.fit(x.values.reshape(-1, 1), y.values)
        y_pred = model.predict(x.values.reshape(-1, 1))
        r2 = model.score(x.values.reshape(-1, 1), y.values)
        m = model.coef_[0]
        b = model.intercept_

        # Save R² summary
        cross_r2_records.append({
            "Group": group_label,
            "File A": source1,
            "File B": source2,
            "R²": round(r2, 4),
            "Slope (m)": round(m, 4),
            "Intercept (b)": round(b, 4)
        })

        # === Static plot ===
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(x, y, alpha=0.7)
        ax.plot(x, y_pred, color='red', linewidth=1.5)
        ax.set_xlabel(f'{source1} (GFP -)')
        ax.set_ylabel(f'{source2} (GFP -)')
        ax.set_title(f'{source1} vs {source2}\ny = {m:.2f}x + {b:.2f}, R² = {r2:.3f}')
        plt.tight_layout()
        png_name = f'Compare_{source1}_vs_{source2}.png'.replace(" ", "_")
        plt.savefig(png_name, dpi=300)
        plt.close()

        # === Interactive plot ===
        fig_html = go.Figure()

        fig_html.add_trace(go.Scatter(
            x=x, y=y,
            mode='markers',
            text=names,
            hovertemplate="Sample: %{text}<br>X: %{x}<br>Y: %{y}<extra></extra>",
            marker=dict(size=6, color='blue'),
            name='Samples'
        ))

        fig_html.add_trace(go.Scatter(
            x=x, y=y_pred,
            mode='lines',
            line=dict(color='red'),
            name='Fit'
        ))

        fig_html.add_annotation(
            text=f"y = {m:.2f}x + {b:.2f}<br>R² = {r2:.3f}",
            xref="paper", yref="paper",
            x=0.99, y=0.01,
            showarrow=False,
            font=dict(size=10),
            align='right'
        )

        fig_html.update_layout(
            title=f'GFP -: {source1} vs {source2}',
            xaxis_title=f'{source1} (GFP -)',
            yaxis_title=f'{source2} (GFP -)',
            width=600, height=600
        )

        html_name = f'Compare_{source1}_vs_{source2}.html'.replace(" ", "_")
        fig_html.write_html(html_name)

# === Separate and compare 'group1' and 'group2' groups ===
df_group1 = combined[combined['Source File'].str.contains("group1")]
df_group2 = combined[combined['Source File'].str.contains("group2")]

compare_sources_and_plot(df_group1, 'group 1')
compare_sources_and_plot(df_group2, 'group 2')

# === Save combined R² summary ===
r2_cross_df = pd.DataFrame(cross_r2_records)
r2_cross_df.to_csv("CrossFile_GFP_Comparisons_R2_Summary.csv", index=False)
