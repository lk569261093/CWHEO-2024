import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ==================== New Imports ====================
import matplotlib.ticker as ticker  # For customizing ticks
# ======================================================

# Define the Excel filename
excel_filename = '含盐量与有机质样本.xlsx'

# Check if the Excel file exists
if not os.path.isfile(excel_filename):
    print(f"Cannot find Excel file '{excel_filename}'")
    exit()

# Read the Excel file
# Assuming the Excel file has a sheet with two columns: '有机质(g/kg)' and '含盐量(g/kg)'
df = pd.read_excel(excel_filename)

# Check if required columns exist
required_columns = ['有机质(g/kg)', '含盐量(g/kg)']
for col in required_columns:
    if col not in df.columns:
        print(f"Column '{col}' not found in the Excel file.")
        exit()

# Extract the data, dropping any NaN values
organic_matter = df['有机质(g/kg)'].dropna()
salinity = df['含盐量(g/kg)'].dropna()

# ==================== New Functionality Starts ====================

# Define a uniform color for all histograms
hist_color = 'skyblue'  # You can change this color as desired

# Function to compute and print statistics
def compute_statistics(data, label):
    data_min = data.min()
    data_max = data.max()
    data_mean = data.mean()
    data_median = data.median()
    print(f"{label} Statistics:")
    print(f"Maximum Value: {data_max:.2f} g/kg")
    print(f"Minimum Value: {data_min:.2f} g/kg")
    print(f"Average Value: {data_mean:.2f} g/kg")
    print(f"Median Value: {data_median:.2f} g/kg\n")
    return data_min, data_max, data_mean, data_median

# Compute statistics for Organic Matter
organic_min, organic_max, organic_mean, organic_median = compute_statistics(organic_matter, "Organic Matter (g/kg)")

# Compute statistics for Salinity
salinity_min, salinity_max, salinity_mean, salinity_median = compute_statistics(salinity, "Salinity (g/kg)")

# Function to create and save histogram
def create_and_save_histogram(data, label, xlabel, output_filename, stats):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the histogram
    n_bins = 50  # Adjust the number of bins as needed
    ax.hist(data, bins=n_bins, color=hist_color, edgecolor='black')
    
    # Set title and labels
    ax.set_title(f'Histogram of {label} Concentration', fontsize=20, fontname='Times New Roman')
    ax.set_xlabel(xlabel, fontsize=16, fontname='Times New Roman')
    ax.set_ylabel('Frequency', fontsize=16, fontname='Times New Roman')
    
    # Customize tick parameters
    ax.tick_params(axis='both', which='major', labelsize=14)
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontname('Times New Roman')
    
    # Set consistent x-axis range with padding
    data_min, data_max, _, _ = stats
    data_range = data_max - data_min
    padding = data_range * 0.05  # 5% padding on each side
    x_min = max(data_min - padding, 0)  # Ensure x_min is not negative
    x_max = data_max + padding
    ax.set_xlim(x_min, x_max)
    
    # Add statistical information as a text box
    data_min, data_max, data_mean, data_median = stats
    text_str = (
        f"Max: {data_max:.2f} g/kg\n"
        f"Min: {data_min:.2f} g/kg\n"
        f"Mean: {data_mean:.2f} g/kg\n"
        f"Median: {data_median:.2f} g/kg"
    )
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.98, 0.95, text_str, transform=ax.transAxes,
            fontsize=14, verticalalignment='top', horizontalalignment='right', bbox=props)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the histogram as a high-resolution JPG file
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    
    # Show the histogram
    plt.show()

# Create and save Organic Matter histogram
create_and_save_histogram(
    data=organic_matter,
    label="Organic Matter (g/kg)",
    xlabel='Organic Matter (g/kg)',
    output_filename='Organic_Matter_Concentration_Histogram.jpg',
    stats=(organic_min, organic_max, organic_mean, organic_median)
)

# Create and save Salinity histogram
create_and_save_histogram(
    data=salinity,
    label="Salinity (g/kg)",
    xlabel='Salinity (g/kg)',
    output_filename='Salinity_Concentration_Histogram.jpg',
    stats=(salinity_min, salinity_max, salinity_mean, salinity_median)
)

# ==================== New Functionality Ends ====================
