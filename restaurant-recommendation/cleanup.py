#!/usr/bin/env python3
"""
Cleanup script for the restaurant recommendation system
Removes temporary files, cache files, and other unwanted files
"""

import os
import shutil
import glob
import sys
from pathlib import Path

def cleanup_directory(directory, patterns, description):
    """Clean up files matching patterns in a directory"""
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist, skipping...")
        return
    
    removed_files = []
    removed_dirs = []
    
    for pattern in patterns:
        # Find files matching pattern
        files = glob.glob(os.path.join(directory, pattern), recursive=True)
        for file_path in files:
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    removed_files.append(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                    removed_dirs.append(file_path)
            except Exception as e:
                print(f"Error removing {file_path}: {e}")
    
    if removed_files or removed_dirs:
        print(f"[OK] {description}:")
        for file_path in removed_files:
            print(f"   Removed file: {file_path}")
        for dir_path in removed_dirs:
            print(f"   Removed directory: {dir_path}")
    else:
        print(f"[INFO] {description}: No files to remove")

def main():
    """Main cleanup function"""
    print("Starting cleanup of restaurant recommendation system...")
    print("=" * 60)
    
    # Define cleanup patterns
    cleanup_patterns = {
        # Python cache files
        "__pycache__": {
            "patterns": ["**/__pycache__", "**/*.pyc", "**/*.pyo"],
            "description": "Python cache files"
        },
        
        # Log files
        "logs": {
            "patterns": ["*.log", "*.log.*", "logs/*.log"],
            "description": "Log files"
        },
        
        # Temporary files
        "temp": {
            "patterns": ["*.tmp", "*.temp", "*.swp", "*.swo", "*~", ".DS_Store"],
            "description": "Temporary files"
        },
        
        # IDE files
        "ide": {
            "patterns": [".vscode", ".idea", "*.sublime-*", ".pytest_cache"],
            "description": "IDE and editor files"
        },
        
        # OS files
        "os": {
            "patterns": [".DS_Store", "Thumbs.db", "desktop.ini"],
            "description": "OS-specific files"
        },
        
        # Backup files
        "backup": {
            "patterns": ["*.bak", "*.backup", "*.old", "*.orig"],
            "description": "Backup files"
        },
        
        # Test artifacts
        "test": {
            "patterns": [".coverage", "htmlcov", ".pytest_cache", "test-results"],
            "description": "Test artifacts"
        }
    }
    
    # Clean up each category
    for category, config in cleanup_patterns.items():
        cleanup_directory(".", config["patterns"], config["description"])
    
    # Clean up specific directories
    directories_to_clean = [
        ("data", ["*.tmp", "*.temp", "*.log"]),
        ("models", ["*.tmp", "*.temp", "*.log"]),
        ("static", ["*.tmp", "*.temp", "*.log"]),
        ("templates", ["*.tmp", "*.temp", "*.log"]),
        ("utils", ["*.tmp", "*.temp", "*.log"])
    ]
    
    for directory, patterns in directories_to_clean:
        cleanup_directory(directory, patterns, f"Files in {directory}/")
    
    # Clean up empty directories
    print("\nChecking for empty directories...")
    empty_dirs = []
    for root, dirs, files in os.walk(".", topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            if not os.listdir(dir_path):
                empty_dirs.append(dir_path)
    
    for empty_dir in empty_dirs:
        try:
            os.rmdir(empty_dir)
            print(f"   Removed empty directory: {empty_dir}")
        except Exception as e:
            print(f"   Error removing empty directory {empty_dir}: {e}")
    
    # Clean up large files that shouldn't be in production
    print("\nChecking for large files...")
    large_files = []
    for root, dirs, files in os.walk("."):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                if os.path.getsize(file_path) > 100 * 1024 * 1024:  # 100MB
                    large_files.append((file_path, os.path.getsize(file_path)))
            except OSError:
                pass
    
    if large_files:
        print("WARNING: Large files found (consider removing if not needed):")
        for file_path, size in large_files:
            size_mb = size / (1024 * 1024)
            print(f"   {file_path} ({size_mb:.1f} MB)")
    else:
        print("[OK] No large files found")
    
    # Summary
    print("\n" + "=" * 60)
    print("Cleanup completed!")
    print("\nSummary:")
    print("   - Removed Python cache files")
    print("   - Removed temporary files")
    print("   - Removed IDE files")
    print("   - Removed OS-specific files")
    print("   - Removed backup files")
    print("   - Removed test artifacts")
    print("   - Removed empty directories")
    
    print("\nTips:")
    print("   - Run this script regularly to keep the project clean")
    print("   - Add patterns to cleanup_patterns for custom cleanup")
    print("   - Review large files before removing them")
    print("   - Consider adding .gitignore entries for temporary files")

if __name__ == "__main__":
    main()
