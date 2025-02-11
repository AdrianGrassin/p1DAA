import pandas as pd
import matplotlib.pyplot as plt
import os

# Read the CSV file
csv_path = r'c:\Users\grass\Desktop\P1\MatrixProd\bin\Debug\net8.0\benchmark_results_detailed.csv'
output_path = r'c:\Users\grass\Desktop\P1\MatrixProd\docs\grafica.png'

# Read CSV with custom delimiter handling
data = []
with open(csv_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

header = lines[0].strip().split(',')

for line in lines[1:]:
    tokens = line.strip().split(',')
    row = {
        'Size': int(tokens[0]),
        'Method': tokens[1],
        'Compute Time (ms)': float(tokens[2].replace(',', '.')),
        'Total Time (ms)': float(tokens[4].replace(',', '.')),
        'Memory Usage (MB)': float(tokens[6].replace(',', '.')),
        'GFlops': float(tokens[8].replace(',', '.'))
    }
    data.append(row)

df = pd.DataFrame(data)

# Set style to a built-in style
plt.style.use('bmh')

# Plot setup
colors = ['#2ecc71', '#e74c3c', '#3498db', '#f1c40f']
markers = ['o', 's', '^', 'D']

# Create two subplots with increased size
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Plot 1: Compute Time
for method, color, marker in zip(df['Method'].unique(), colors, markers):
    method_data = df[df['Method'] == method]
    ax1.plot(method_data['Size'], 
            method_data['Compute Time (ms)'],
            marker=marker,
            label=method,
            color=color,
            linewidth=2,
            markersize=6)

ax1.set_xlabel('Tamaño de Matriz', fontsize=12)
ax1.set_ylabel('Tiempo de Computación (ms)', fontsize=12)
ax1.set_title('Tiempo de Computación vs Tamaño', fontsize=14)
ax1.set_yscale('log')
ax1.grid(True, which="both", ls="-", alpha=0.2)
ax1.legend(fontsize=10, loc='upper left')
ax1.tick_params(labelsize=10)

# Plot 2: GFlops
for method, color, marker in zip(df['Method'].unique(), colors, markers):
    method_data = df[df['Method'] == method]
    ax2.plot(method_data['Size'], 
            method_data['GFlops'],
            marker=marker,
            label=method,
            color=color,
            linewidth=2,
            markersize=6)

ax2.set_xlabel('Tamaño de Matriz', fontsize=12)
ax2.set_ylabel('Rendimiento (GFlops)', fontsize=12)
ax2.set_title('Rendimiento vs Tamaño', fontsize=14)
ax2.grid(True, which="both", ls="-", alpha=0.2)
ax2.legend(fontsize=10, loc='upper left')
ax2.tick_params(labelsize=10)

# Adjust layout with more space
plt.tight_layout(pad=2.0)

# Save the plot with higher resolution
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()