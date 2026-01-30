# Delete All AlexNet.pth path files to make it clear (the path file has been saved in other places)
from pathlib import Path
import os

base_dir = Path('/Users/bjy/Desktop/Projects/PsychometricFittingCurve/ConvResults') 

target_file = 'AlexNet.pth'

# If it is True, just print the result；Make sure there's no problems, turn it into False and strat to delete
DRY_RUN = False

print(f"Directory: {base_dir}")
print(f"Find files: {target_file}")

found_count = 0

for file_path in base_dir.rglob(target_file):
    if file_path.is_file():
        found_count += 1
        
        if DRY_RUN:
            print(f"[Test] Will delete: {file_path}")
        else:
            try:
                file_path.unlink() # Mac/Linux
                print(f"[Deleted] {file_path}")
            except PermissionError:
                print(f"Premission Error: {file_path} ")
            except Exception as e:
                print(f"[Wrong] {file_path}: {e}")

print("-" * 30)
if DRY_RUN:
    print(f"Find {found_count} files to delete。")
    print("Change DRY_RUN into False to act deletation。")
else:
    print(f"Finished。Deleted {found_count} files。")