import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np
import math

# Main folder path containing subfolders
main_folder_path = r"C:\Users\ashee\Downloads\upwork\dani\2025_05_03\5015_Mai_2025\ill"

# Configuration - Set these values directly
IRRADIANCE = 100.0  # Set your irradiance level here (in mW/cm²)
SELECTED_PIXEL = "jvalues"  # Set the pixel/column you want to plot

def get_experiment_name(folder_name):
    """Generate experiment name based on folder name and timestamp"""
    timestamp = pd.to_datetime('now').strftime('%Y%m%d_%H%M%S')
    return f"Experiment_{folder_name}_{timestamp}"

def get_subfolders_and_files(main_path):
    """Get all subfolders and their CSV/Excel files"""
    subfolders_data = {}
    
    if not os.path.exists(main_path):
        print(f"Main folder not found: {main_path}")
        return subfolders_data
    
    try:
        # Get all items in the main folder
        items = os.listdir(main_path)
        
        for item in items:
            item_path = os.path.join(main_path, item)
            
            # Check if it's a directory
            if os.path.isdir(item_path):
                print(f"Found subfolder: {item}")
                
                # Get all files in this subfolder
                excel_extensions = ['.xlsx', '.xls', '.xlsm', '.xlsb', '.csv']
                excel_files = []
                excel_file_paths = []
                
                try:
                    subfolder_files = os.listdir(item_path)
                    for file in subfolder_files:
                        if any(file.lower().endswith(ext) for ext in excel_extensions):
                            excel_files.append(file)
                            excel_file_paths.append(os.path.join(item_path, file))
                    
                    if excel_file_paths:
                        subfolders_data[item] = {
                            'path': item_path,
                            'files': excel_files,
                            'file_paths': excel_file_paths
                        }
                        print(f"  └── Found {len(excel_files)} data files")
                    else:
                        print(f"  └── No data files found")
                        
                except Exception as e:
                    print(f"  └── Error accessing subfolder {item}: {str(e)}")
                    
    except Exception as e:
        print(f"Error accessing main folder: {str(e)}")
    
    return subfolders_data

def calculate_subplot_layout(num_files):
    """Calculate optimal subplot layout based on number of files"""
    if num_files <= 1:
        return 1, 1
    elif num_files <= 2:
        return 1, 2
    elif num_files <= 4:
        return 2, 2
    elif num_files <= 6:
        return 2, 3
    elif num_files <= 9:
        return 3, 3
    elif num_files <= 12:
        return 3, 4
    elif num_files <= 16:
        return 4, 4
    elif num_files <= 20:
        return 4, 5
    else:
        # For very large numbers, calculate square-ish layout
        cols = math.ceil(math.sqrt(num_files))
        rows = math.ceil(num_files / cols)
        return rows, cols

def get_color_palette(num_colors):
    """Generate a color palette for any number of plots"""
    base_colors = ['darkblue', 'red', 'green', 'orange', 'purple', 'brown', 
                   'pink', 'gray', 'olive', 'cyan', 'magenta', 'yellow',
                   'navy', 'teal', 'lime', 'maroon', 'aqua', 'fuchsia']
    
    if num_colors <= len(base_colors):
        return base_colors[:num_colors]
    else:
        # Generate additional colors using matplotlib colormap
        import matplotlib.cm as cm
        additional_colors = cm.tab20(np.linspace(0, 1, num_colors - len(base_colors)))
        extended_colors = base_colors + [tuple(color[:3]) for color in additional_colors]
        return extended_colors[:num_colors]

def get_linestyle_palette(num_styles):
    """Generate line styles for any number of plots"""
    base_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
    return (base_styles * ((num_styles // len(base_styles)) + 1))[:num_styles]

def process_single_file(df, irradiance, selected_pixel, file_path):
    """Process a single file and return plot data and parameters"""
    # Assuming first column is voltage and selected column is current density
    voltage = df.iloc[:, 0]  # First column as voltage
    current_density = df[selected_pixel]  # Selected column as current density
    pin = irradiance  # Illumination in mW/cm²
    
    # Filter out any NaN or invalid values
    mask = ~(pd.isna(voltage) | pd.isna(current_density))
    voltage = voltage[mask]
    current_density = current_density[mask]
    
    if len(voltage) < 3:
        return None, None, None, None, None, None, None, None
    
    try:
        # Interpolation for smooth curve
        f = interp1d(voltage, current_density, kind='cubic')
        v_hr = np.linspace(voltage.min(), voltage.max(), 1000)
        j_hr = f(v_hr)

        # Key parameters
        jsc = float(f(0)) if 0 >= voltage.min() and 0 <= voltage.max() else current_density.iloc[0]  # Short circuit current
        
        # For Voc calculation, find where current crosses zero
        try:
            voc_func = interp1d(current_density, voltage, kind='linear', fill_value='extrapolate')
            voc = float(voc_func(0))  # Open circuit voltage
        except:
            voc = voltage.iloc[-1]  # Fallback to last voltage value

        power = v_hr * j_hr
        pmax_index = np.argmax(np.abs(power))  # Use absolute value for power
        v_mpp = v_hr[pmax_index]
        j_mpp = j_hr[pmax_index]
        
        if voc != 0 and jsc != 0:
            ff = abs(v_mpp * j_mpp) / abs(voc * jsc)  # Fill factor
            pce = abs(v_mpp * j_mpp / pin) * 100  # Power conversion efficiency
        else:
            ff = 0
            pce = 0

        return v_hr, j_hr, voc, jsc, ff, pce, voltage, current_density
        
    except Exception as e:
        print(f"Error during processing {os.path.basename(file_path)}: {str(e)}")
        return voltage, current_density, 0, 0, 0, 0, voltage, current_density

def plot_individual_curve(v_data, j_data, voc, jsc, ff, pce, selected_pixel, file_index, file_path, experiment_name, save_dir):
    """Plot and save individual IV curve"""
    plt.figure(figsize=(10, 8))
    plt.plot(v_data, j_data, label=f"{selected_pixel}", color="darkblue", linewidth=2)
    
    # Extract filename for title
    filename = os.path.basename(file_path).replace('.csv', '')
    plt.title(f"IV Curve – {selected_pixel}\n{filename}", fontsize=14, fontweight='bold')
    plt.xlabel("Voltage (V)", fontsize=14)
    plt.ylabel("Current Density", fontsize=14)
    plt.grid(True, alpha=0.3)

    # Annotate parameters on plot
    annotation = (
        f"Voc = {voc:.2f} V\n"
        f"Jsc = {jsc:.2f}\n"
        f"FF = {ff*100:.2f} %\n"
        f"PCE = {pce:.2f} %"
    )
    plt.text(0.05, 0.95, annotation, transform=plt.gca().transAxes,
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", edgecolor='gray'))

    plt.legend(fontsize=12)
    plt.tight_layout()
    
    # Save individual plot with filename
    filename_clean = filename.replace('.xlsx', '').replace('.xls', '').replace('.xlsm', '').replace('.xlsb', '')
    individual_filename = f"{filename_clean}_{selected_pixel}_IV_curve.png"
    individual_path = os.path.join(save_dir, "individual_plots", individual_filename)
    
    # Create individual plots directory if it doesn't exist
    individual_dir = os.path.join(save_dir, "individual_plots")
    if not os.path.exists(individual_dir):
        os.makedirs(individual_dir)
    
    plt.savefig(individual_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure without showing it
    
    print(f"Individual plot saved: {individual_filename}")
    return individual_path

def create_combined_plot(all_data, experiment_name, save_dir, folder_name):
    """Create and save a single plot with all curves overlaid"""
    valid_data = [(data, path) for data, path in all_data if data is not None]
    num_valid = len(valid_data)
    
    if num_valid == 0:
        print("No valid data to create combined plot")
        return None
    
    plt.figure(figsize=(14, 10))
    
    # Get colors and line styles for all curves
    colors = get_color_palette(num_valid)
    linestyles = get_linestyle_palette(num_valid)
    
    for i, (file_data, file_path) in enumerate(valid_data):
        v_data, j_data, voc, jsc, ff, pce = file_data
        filename = os.path.basename(file_path).replace('.csv', '').replace('.xlsx', '').replace('.xls', '').replace('.xlsm', '').replace('.xlsb', '')
        
        # Truncate filename if too long for legend
        display_name = filename if len(filename) <= 20 else filename[:17] + "..."
        
        # Plot each curve with different color and style
        plt.plot(v_data, j_data, 
                color=colors[i], 
                linestyle=linestyles[i],
                linewidth=2, 
                label=f"{display_name} (PCE={pce:.1f}%)")
    
    # Formatting
    plt.title(f'Combined IV Curves - {SELECTED_PIXEL} (Folder: {folder_name})\n({num_valid} files)', fontsize=16, fontweight='bold')
    plt.xlabel("Voltage (V)", fontsize=14)
    plt.ylabel("Current Density", fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Adjust legend based on number of items
    if num_valid <= 10:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    else:
        # For many files, use smaller font and multiple columns
        ncols = 2 if num_valid <= 20 else 3
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=ncols)
    
    plt.tight_layout()
    
    # Save combined plot
    combined_filename = f"Combined_{folder_name}_{SELECTED_PIXEL}_IV_curves.png"
    combined_path = os.path.join(save_dir, combined_filename)
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Combined plot saved: {combined_filename}")
    return combined_path

def create_subplots(all_data, experiment_name, save_dir, folder_name):
    """Create and save subplots of all curves"""
    valid_data = [(data, path) for data, path in all_data if data is not None]
    num_valid = len(valid_data)
    
    if num_valid == 0:
        print("No valid data to create subplots")
        return None
    
    # Calculate optimal layout
    rows, cols = calculate_subplot_layout(num_valid)
    
    # Calculate figure size based on number of subplots
    fig_width = min(6 * cols, 24)  # Max width of 24 inches
    fig_height = min(4 * rows, 16)  # Max height of 16 inches
    
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
    
    # Handle case of single subplot
    if num_valid == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else axes
    
    # Get colors for all plots
    colors = get_color_palette(num_valid)
    
    for i, (file_data, file_path) in enumerate(valid_data):
        v_data, j_data, voc, jsc, ff, pce = file_data
        filename = os.path.basename(file_path).replace('.csv', '').replace('.xlsx', '').replace('.xls', '').replace('.xlsm', '').replace('.xlsb', '')
        
        # Truncate filename for subplot title
        display_name = filename if len(filename) <= 25 else filename[:22] + "..."
        
        # Plot on subplot
        axes[i].plot(v_data, j_data, color=colors[i], linewidth=2)
        axes[i].set_title(f"{display_name}", fontsize=max(8, 12 - len(valid_data)//5), fontweight='bold')
        axes[i].set_xlabel("Voltage (V)", fontsize=max(8, 10 - len(valid_data)//10))
        axes[i].set_ylabel("Current Density", fontsize=max(8, 10 - len(valid_data)//10))
        axes[i].grid(True, alpha=0.3)
        
        # Add parameters as text
        annotation = (
            f"Voc={voc:.2f}V\n"
            f"Jsc={jsc:.2f}\n"
            f"FF={ff*100:.1f}%\n"
            f"PCE={pce:.1f}%"
        )
        font_size = max(6, 8 - len(valid_data)//8)
        axes[i].text(0.02, 0.98, annotation, transform=axes[i].transAxes,
                    fontsize=font_size, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", edgecolor='gray', alpha=0.8))
    
    # Hide empty subplots
    total_subplots = rows * cols
    for i in range(num_valid, total_subplots):
        if i < len(axes):
            axes[i].set_visible(False)
    
    # Set main title
    title_font_size = max(12, 16 - len(valid_data)//5)
    fig.suptitle(f'IV Curves - {SELECTED_PIXEL} (Folder: {folder_name}) - {num_valid} files', 
                 fontsize=title_font_size, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)  # Make room for main title
    
    # Save subplots
    subplot_filename = f"Subplots_{folder_name}_{SELECTED_PIXEL}_IV_curves.png"
    subplot_path = os.path.join(save_dir, subplot_filename)
    plt.savefig(subplot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Subplots saved: {subplot_filename}")
    return subplot_path

def process_folder_files(folder_name, folder_data):
    """Process all files in a specific folder"""
    print(f"\n{'='*80}")
    print(f"PROCESSING FOLDER: {folder_name}")
    print(f"{'='*80}")
    
    file_paths = folder_data['file_paths']
    num_files = len(file_paths)
    
    # Create experiment name and directory for this folder
    experiment_name = get_experiment_name(folder_name)
    save_dir = os.path.join(os.getcwd(), "Results", experiment_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    print(f"Processing {num_files} files from folder: {folder_name}")
    print(f"Results will be saved in: {save_dir}")
    
    # Calculate and display subplot layout
    rows, cols = calculate_subplot_layout(num_files)
    print(f"Subplot layout will be: {rows} rows × {cols} columns")
    print("-" * 60)
    
    all_data = []  # Store all processed data for subplots
    individual_paths = []  # Store paths of individual plots
    
    # Loop through all files in this folder
    for file_index, file_path in enumerate(file_paths):
        print(f"\nProcessing file {file_index + 1}/{num_files}: {os.path.basename(file_path)}")
        
        # Load the data
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
            print(f"Successfully loaded: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"Error: Failed to load file {file_index + 1}: {str(e)}")
            all_data.append((None, file_path))
            continue

        # Check that the file is loaded successfully
        if df.empty:
            print(f"Error: File {file_index + 1} is empty. Skipping...")
            all_data.append((None, file_path))
            continue

        # Check if selected pixel exists
        if SELECTED_PIXEL not in df.columns:
            print(f"Error: Column '{SELECTED_PIXEL}' not found in file {file_index + 1}.")
            print(f"Available columns: {list(df.columns)}")
            all_data.append((None, file_path))
            continue

        # Process the file
        v_hr, j_hr, voc, jsc, ff, pce, voltage, current_density = process_single_file(df, IRRADIANCE, SELECTED_PIXEL, file_path)
        
        if v_hr is not None:
            # Store data for subplots
            all_data.append(((v_hr, j_hr, voc, jsc, ff, pce), file_path))
            
            # Create and save individual plot
            individual_path = plot_individual_curve(v_hr, j_hr, voc, jsc, ff, pce, SELECTED_PIXEL, file_index, file_path, experiment_name, save_dir)
            individual_paths.append(individual_path)
            
            print(f"File {file_index + 1} processed successfully!")
        else:
            print(f"File {file_index + 1} could not be processed!")
            all_data.append((None, file_path))
        
        print("-" * 40)
    
    # Create subplots with all data
    valid_files = len([d for d in all_data if d[0] is not None])
    print(f"\nCreating subplots with {valid_files} valid files...")
    subplot_path = create_subplots(all_data, experiment_name, save_dir, folder_name)
    
    # Create combined plot with all curves in one plot
    print(f"\nCreating combined plot with all curves...")
    combined_path = create_combined_plot(all_data, experiment_name, save_dir, folder_name)
    
    # Summary for this folder
    print(f"\n{'='*60}")
    print(f"FOLDER '{folder_name}' COMPLETED: {experiment_name}")
    print(f"{'='*60}")
    print(f"Save directory: {save_dir}")
    if subplot_path:
        print(f"Subplots saved: {os.path.basename(subplot_path)}")
    if combined_path:
        print(f"Combined plot saved: {os.path.basename(combined_path)}")
    print(f"Individual plots: {len(individual_paths)} files in 'individual_plots' folder")
    print(f"Files processed: {valid_files}/{num_files}")
    
    return {
        'folder_name': folder_name,
        'experiment_name': experiment_name,
        'save_dir': save_dir,
        'files_processed': valid_files,
        'total_files': num_files,
        'subplot_path': subplot_path,
        'combined_path': combined_path,
        'individual_paths': individual_paths
    }

def main():
    """Main function to process all subfolders"""
    print(f"{'='*80}")
    print(f"MULTI-FOLDER IV CURVE ANALYSIS")
    print(f"{'='*80}")
    print(f"Main folder: {main_folder_path}")
    print(f"Selected pixel: {SELECTED_PIXEL}")
    print(f"Irradiance: {IRRADIANCE} mW/cm²")
    
    # Get all subfolders and their files
    subfolders_data = get_subfolders_and_files(main_folder_path)
    
    if not subfolders_data:
        print("No subfolders with data files found!")
        return
    
    print(f"\nFound {len(subfolders_data)} subfolders with data files:")
    for folder_name, data in subfolders_data.items():
        print(f"  └── {folder_name}: {len(data['files'])} files")
    
    # Create main results directory
    results_dir = os.path.join(os.getcwd(), "Results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Process each subfolder
    all_results = []
    for folder_name, folder_data in subfolders_data.items():
        try:
            result = process_folder_files(folder_name, folder_data)
            all_results.append(result)
        except Exception as e:
            print(f"Error processing folder {folder_name}: {str(e)}")
            continue
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"ALL FOLDERS COMPLETED!")
    print(f"{'='*80}")
    print(f"Total folders processed: {len(all_results)}")
    
    total_files_processed = sum(r['files_processed'] for r in all_results)
    total_files_found = sum(r['total_files'] for r in all_results)
    
    print(f"Total files processed: {total_files_processed}/{total_files_found}")
    print(f"Results saved in: {results_dir}")
    
    print(f"\nFolder Summary:")
    for result in all_results:
        print(f"  └── {result['folder_name']}: {result['files_processed']}/{result['total_files']} files → {result['experiment_name']}")

# Main execution
if __name__ == "__main__":
    main()