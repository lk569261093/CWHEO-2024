import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ==================== New Imports ====================
import matplotlib.ticker as ticker  # For customizing ticks
# ======================================================

# Define the Excel filename
excel_filename = 'Zn与SOM样本.xlsx'

# Check if the Excel file exists
if not os.path.isfile(excel_filename):
    print(f"Cannot find Excel file '{excel_filename}'")
    exit()

# Read the Excel file
# Assuming the Excel file has a sheet with two columns: 'Zn(mg/kg)' and 'SOM(%)'
df = pd.read_excel(excel_filename)

# Check if required columns exist
required_columns = ['Zn(mg/kg)', 'SOM(%)']
for col in required_columns:
    if col not in df.columns:
        print(f"Column '{col}' not found in the Excel file.")
        exit()

# Extract the data, dropping any NaN values
zinc = df['Zn(mg/kg)'].dropna()
som = df['SOM(%)'].dropna()

# ==================== New Functionality Starts ====================

# Function to compute and print statistics
def compute_statistics(data, label, unit):
    data_min = data.min()
    data_max = data.max()
    data_mean = data.mean()
    data_median = data.median()
    print(f"{label} Statistics:")
    print(f"Maximum Value: {data_max:.2f} {unit}")
    print(f"Minimum Value: {data_min:.2f} {unit}")
    print(f"Average Value: {data_mean:.2f} {unit}")
    print(f"Median Value: {data_median:.2f} {unit}\n")
    return data_min, data_max, data_mean, data_median

# Compute statistics for Zn
zinc_min, zinc_max, zinc_mean, zinc_median = compute_statistics(zinc, "Zn Concentration", "mg/kg")

# Compute statistics for SOM
som_min, som_max, som_mean, som_median = compute_statistics(som, "SOM Concentration", "%")

# Function to create and save histogram
def create_and_save_histogram(data, label, xlabel, unit, output_filename, stats, color='skyblue', x_limits=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the histogram
    n_bins = 50  # Adjust the number of bins as needed
    ax.hist(data, bins=n_bins, color=color, edgecolor='black')
    
    # Set title and labels
    ax.set_title(f'Histogram of {label}', fontsize=20, fontname='Times New Roman')
    ax.set_xlabel(xlabel, fontsize=16, fontname='Times New Roman')
    ax.set_ylabel('Frequency', fontsize=16, fontname='Times New Roman')
    
    # Set x-axis limits if provided
    if x_limits is not None:
        ax.set_xlim(x_limits)
    
    # Customize tick parameters
    ax.tick_params(axis='both', which='major', labelsize=14)
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontname('Times New Roman')
    
    # Add statistical information as a text box
    data_min, data_max, data_mean, data_median = stats
    text_str = (
        f"Max: {data_max:.2f} {unit}\n"
        f"Min: {data_min:.2f} {unit}\n"
        f"Mean: {data_mean:.2f} {unit}\n"
        f"Median: {data_median:.2f} {unit}"
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

# Define a consistent color for both histograms
hist_color = 'skyblue'  # You can choose any color you prefer

# Create and save Zn histogram
create_and_save_histogram(
    data=zinc,
    label="Zn(mg/kg) Concentration",
    xlabel='Zn (mg/kg)',
    unit='mg/kg',
    output_filename='Zn_Concentration_Histogram.jpg',
    stats=(zinc_min, zinc_max, zinc_mean, zinc_median),
    color=hist_color  # Use the consistent color
)

# Create and save SOM histogram with x-axis limits from 0 to 25
create_and_save_histogram(
    data=som,
    label="SOM(%) Concentration",
    xlabel='SOM (%)',
    unit='%',
    output_filename='SOM_Concentration_Histogram.jpg',
    stats=(som_min, som_max, som_mean, som_median),
    color=hist_color,  # Use the consistent color
    x_limits=(0, 25)    # Set x-axis range from 0 to 25
)

# ==================== New Functionality Ends ====================
