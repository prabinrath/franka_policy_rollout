import os

def rename_files_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        if os.path.isfile(file_path):
            name, ext = os.path.splitext(filename)
            new_filename = f"{name}_new{ext}"
            new_file_path = os.path.join(folder_path, new_filename)
            
            os.rename(file_path, new_file_path)
            print(f"Renamed: {filename} → {new_filename}")

if __name__ == "__main__":
    folder = "/root/catkin_ws/src/codig_robot/combined_bags_block_pick"  # Change this to your folder path
    rename_files_in_folder(folder)
