import os
import re
import numpy as np
from pathlib import Path
from typing import List, Union, Tuple, Optional
import pandas as pd 

from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import re
from datetime import datetime

class FileExtractor:
    """Extract files from subfolder structure based on filename patterns."""
    
    def __init__(self, root_path: str = "/devb/WHU-OPT-SAR/jointpatch224full_txt"):
        self.root_path = Path(root_path)
        
    def parse_filename(self, filename: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Parse filename to extract subfolder and file name.
        
        Args:
            filename: e.g., "step_98400train_batch96_pair_NH50E006007_patch_0183_3_gen"
        
        Returns:
            Tuple of (subfolder_name, file_name) or (None, None) if pattern not found
        """
        # Pattern to match pair_XXXXX and patch_XXXX
        pair_pattern = r'pair_([^_]+)'
        patch_pattern = r'patch_(\d+)'
        
        pair_match = re.search(pair_pattern, filename)
        patch_match = re.search(patch_pattern, filename)
        
        if pair_match and patch_match:
            subfolder = f"pair_{pair_match.group(1)}"
            file_name = f"patch_{patch_match.group(1)}.npy"
            return subfolder, file_name
        
        return None, None
    
    def get_file_path(self, filename: str) -> Optional[Path]:
        """
        Get full file path from filename pattern.
        
        Args:
            filename: Pattern filename
        
        Returns:
            Full path to the file or None if not found
        """
        subfolder, file_name = self.parse_filename(filename)
        
        if subfolder and file_name:
            full_path = self.root_path / subfolder / file_name
            return full_path
        
        return None
    
    def load_file(self, filename: str) -> Optional[np.ndarray]:
        """
        Load numpy file based on filename pattern.
        
        Args:
            filename: Pattern filename
        
        Returns:
            Numpy array or None if file not found
        """
        file_path = self.get_file_path(filename)
        
        if file_path and file_path.exists():
            return np.load(file_path)
        else:
            print(f"File not found: {file_path}")
            return None
    
    def load_batch(self, filenames: List[str], 
                   verbose: bool = True) -> List[Tuple[str, Optional[np.ndarray]]]:
        """
        Load multiple files in batch.
        
        Args:
            filenames: List of pattern filenames
            verbose: Print progress
        
        Returns:
            List of tuples (filename, data) where data is numpy array or None
        """
        results = []
        
        for i, filename in enumerate(filenames):
            if verbose and (i + 1) % 10 == 0:
                print(f"Processing {i + 1}/{len(filenames)} files...")
            
            data = self.load_file(filename)
            results.append((filename, data))
        
        if verbose:
            successful = sum(1 for _, data in results if data is not None)
            print(f"Loaded {successful}/{len(filenames)} files successfully")
        
        return results
    
    def batch_exists(self, filenames: List[str]) -> List[Tuple[str, bool, Optional[Path]]]:
        """
        Check if multiple files exist without loading them.
        
        Args:
            filenames: List of pattern filenames
        
        Returns:
            List of tuples (filename, exists, path)
        """
        results = []
        
        for filename in filenames:
            file_path = self.get_file_path(filename)
            exists = file_path.exists() if file_path else False
            results.append((filename, exists, file_path))
        
        return results
      
    def batch_files(self, filenames: List[str]) -> List[Optional[Path]]:
        """
        Check if multiple files exist without loading them.
        
        Args:
            filenames: List of pattern filenames
        
        Returns:
            List of file paths (None if file doesn't exist)
        """
        filepath = []
        
        for filename in filenames:
            file_path = self.get_file_path(filename)
            if not file_path.exists():
                file_path = None 
            filepath.append(file_path)
        
        return filepath  
      
      
    def batch_files2dt(self, filenames: List[str],dts: List[datetime]) -> List[Optional[Path]]:
        """
        Check if multiple files exist without loading them.
        
        Args:
            filenames: List of pattern filenames
        
        Returns:
            List of file paths (None if file doesn't exist)
        """
        filenames_sel = []
        filepath = [] 
        filedt = [] 
        
        for i,filename in enumerate(filenames):
            file_path = self.get_file_path(filename)
            if not file_path.exists():
                file_path = None
                continue 
            filenames_sel.append(filename)
            filepath.append(file_path)
            filedt.append(dts[i])
        return filenames_sel,filepath,filedt 
      
    
    # ========== NEW METHODS FOR RETRIEVING FILES WITH SUFFIX ==========
    
    def get_file_creation_time(self, file_path: Path) -> datetime:
        """
        Get the creation/modification time of a file.
        
        Args:
            file_path: Path object of the file
        
        Returns:
            datetime object representing file creation time
        """
        # Use st_mtime (modification time) which is more reliable across platforms
        timestamp = file_path.stat().st_mtime
        return datetime.fromtimestamp(timestamp)
    
    def get_all_files_with_suffix(self, folder_path: str, suffix: str = "_gen.png", 
                                   recursive: bool = False, 
                                   sort_by_time: bool = False) -> List[Path]:
        """
        Retrieve all files ending with a specific suffix in a folder.
        
        Args:
            folder_path: Path to the folder to search
            suffix: File ending to search for (e.g., "_gen.png")
            recursive: If True, search in subfolders as well
            sort_by_time: If True, sort by creation time (most recent first)
        
        Returns:
            List of Path objects for matching files
        """
        folder = Path(folder_path)
        
        if not folder.exists():
            print(f"Folder not found: {folder}")
            return []
        
        if recursive:
            # Search recursively in all subfolders
            pattern = f"**/*{suffix}"
            files = list(folder.glob(pattern))
        else:
            # Search only in the specified folder
            pattern = f"*{suffix}"
            files = list(folder.glob(pattern))
        
        if sort_by_time:
            # Sort by modification time, most recent first (descending order)
            files = sorted(files, key=lambda f: f.stat().st_mtime, reverse=True)
        else:
            files = sorted(files)
        
        return files
    
    def get_all_files_with_datetime(self, folder_path: str, suffix: str = "_gen.png", 
                                     recursive: bool = False) -> List[Tuple[Path, datetime]]:
        """
        Retrieve all files with their creation datetime, sorted by time (most recent first).
        
        Args:
            folder_path: Path to the folder to search
            suffix: File ending to search for (e.g., "_gen.png")
            recursive: If True, search in subfolders as well
        
        Returns:
            List of tuples (file_path, datetime) sorted by datetime (most recent first)
        """
        folder = Path(folder_path)
        
        if not folder.exists():
            print(f"Folder not found: {folder}")
            return []
        
        if recursive:
            pattern = f"**/*{suffix}"
            files = list(folder.glob(pattern))
        else:
            pattern = f"*{suffix}"
            files = list(folder.glob(pattern))
        
        # Create list of (file, datetime) tuples
        files_with_datetime = [
            (f, self.get_file_creation_time(f)) for f in files
        ]
        
        # Sort by datetime, most recent first (descending order)
        files_with_datetime.sort(key=lambda x: x[1], reverse=True)
        
        return files_with_datetime
    
    def get_all_gen_png_files(self, folder_path: str, recursive: bool = False, 
                              sort_by_time: bool = False) -> List[str]:
        """
        Convenience method to get all _gen.png files and return as string paths.
        
        Args:
            folder_path: Path to the folder to search
            recursive: If True, search in subfolders as well
            sort_by_time: If True, sort by creation time (most recent first)
        
        Returns:
            List of file paths as strings
        """
        files = self.get_all_files_with_suffix(folder_path, "_gen.png", recursive, sort_by_time)
        return [str(f) for f in files]
    
    def get_gen_png_files_with_datetime(self, folder_path: str, 
                                        recursive: bool = False) -> List[Tuple[str, datetime]]:
        """
        Get all _gen.png files with their creation datetime (most recent first).
        
        Args:
            folder_path: Path to the folder to search
            recursive: If True, search in subfolders as well
        
        Returns:
            List of tuples (file_path_string, datetime) sorted by datetime (most recent first)
        """
        files_with_datetime = self.get_all_files_with_datetime(folder_path, "_gen.png", recursive)
        return [(str(f), dt) for f, dt in files_with_datetime]
    
    def get_filenames_only(self, folder_path: str, suffix: str = "_gen.png", 
                           recursive: bool = False, 
                           sort_by_time: bool = False) -> List[str]:
        """
        Get only the filenames (without full path) ending with suffix.
        
        Args:
            folder_path: Path to the folder to search
            suffix: File ending to search for
            recursive: If True, search in subfolders as well
            sort_by_time: If True, sort by creation time (most recent first)
        
        Returns:
            List of filenames only (no path)
        """
        files = self.get_all_files_with_suffix(folder_path, suffix, recursive, sort_by_time)
        return [f.name for f in files]
    
    def get_filenames_with_datetime(self, folder_path: str, suffix: str = "_gen.png", 
                                    recursive: bool = False) -> List[Tuple[str, datetime]]:
        """
        Get filenames (without full path) with their creation datetime (most recent first).
        
        Args:
            folder_path: Path to the folder to search
            suffix: File ending to search for
            recursive: If True, search in subfolders as well
        
        Returns:
            List of tuples (filename, datetime) sorted by datetime (most recent first)
        """
        files_with_datetime = self.get_all_files_with_datetime(folder_path, suffix, recursive)
        return [(f.name, dt) for f, dt in files_with_datetime]
      
      
def extractSelFls(count=100):
    extractor = FileExtractor()
    
    # Get _gen.png files with datetime
    rootpath = '/devb/sar2opt_diff_txt/sar2opt_diff_sarmap_Oct2_test1_continue'
    gen_files = extractor.get_gen_png_files_with_datetime(rootpath)
    allgenfiles, allgenfiles_dt = [], []
    for file_path, dt in gen_files:
#        print(f"{file_path} was created at {dt}")
        allgenfiles.append(file_path)
        allgenfiles_dt.append(dt)
            
    filenames_sel,filepath,filedt = extractor.batch_files2dt(allgenfiles,allgenfiles_dt)
    print(f"all {len(filenames_sel)} files generated!!!")
    sdata = pd.DataFrame({'filename':filenames_sel,'fpath':filepath,'gendtime':filedt},index=[i for i in range(1,len(filenames_sel)+1)])
    print(sdata.dtypes)
    # Sort by datetime (most recent first) and reset index to start from 1
    sdata = sdata.sort_values('gendtime', ascending=False).reset_index(drop=True) 
    sdata[:count].to_csv('/devb/sar2opt_diff_txt/test/selectedTestFl_dt.csv',index=False)
    

def selectFls(count=100):
    extractor = FileExtractor() 
    tpath='/devb/sar2opt_diff_txt/sar2opt_diff_sarmap_Oct2_test1_continue'
    allgenfiles = extractor.get_filenames_only(tpath)
    pathfiles = extractor.batch_files(allgenfiles)
    allgenfiles=[allgenfiles[i] for i, fpath in enumerate(pathfiles) if fpath is not None]
    pathfiles = [fpath for fpath in pathfiles if fpath is not None]
    print(f'{len(allgenfiles)} vs. {len(pathfiles)}')
    sdata = pd.DataFrame({'filename':allgenfiles[:count],'fpath':pathfiles[:count]},index=[i for i in range(1,count+1)])
    sdata.to_csv('/devb/sar2opt_diff_txt/test/selectedTestFl.csv',index=False)


# Example usage
#if __name__ == "__main__":
#    extractSelFls(count=100) 

