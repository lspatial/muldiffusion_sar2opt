import pandas as pd
import glob
import os

# Set your target path here
target_path = "/devb/sar2opt_diff/sar2opt_map_selected"  # Change this to your desired path
# Alternative examples:
# target_path = "D:\\your_folder"
# target_path = "C:\\Users\\YourName\\Documents"

# Method 1: Using glob (recommended)
def collect_png_files_glob(path):
    """Find all *_gen.png files using glob"""
    # Use ** for recursive search in subdirectories
    pattern = os.path.join(path, "**", "*_gen.png")
    png_files = glob.glob(pattern, recursive=True)
    return png_files

# Method 2: Using os.walk (alternative)
def collect_png_files_walk(path):
    """Find all *_gen.png files using os.walk"""
    png_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith("_gen.png"):
                png_files.append(os.path.join(root, file))
    return png_files

# Main execution
try:
    print(f"Searching for *_gen.png files in: {target_path}")
    
    # Get all PNG files ending with _gen.png
    png_files = collect_png_files_glob(target_path)
    
    # Alternative: use the os.walk method instead
    # png_files = collect_png_files_walk(target_path)
    
    print(f"Found {len(png_files)} files")
    
    if png_files:
        # Create DataFrame
        df = pd.DataFrame({
            'png_filename': png_files
        })
        
        # Optional: You can also add just the filename without full path
        # df['filename_only'] = df['png_filename'].apply(os.path.basename)
        
        # Save to CSV
        output_file = "/devb/sar2opt_diff/sar2opt_map_selected/png_files_summary.csv"
        df.to_csv(output_file, index=False)
        
        print(f"DataFrame saved to: {output_file}")
        print(f"DataFrame shape: {df.shape}")
        print("\nFirst few rows:")
        print(df.head())
        
    else:
        print("No files ending with '_gen.png' found in the specified path.")
        # Create empty DataFrame and save anyway
        df = pd.DataFrame({'png_filename': []})
        df.to_csv("png_files_summary.csv", index=False)
        print("Empty CSV file created.")

except Exception as e:
    print(f"Error: {e}")
