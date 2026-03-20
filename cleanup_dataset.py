import os

def delete_files_by_name(root_dir: str):
    """
    Recursively iterates over subfolders and deletes files
    that contain target_name in their filename.

    Args:
        root_dir   : Path to the root directory to search
    """
    deleted_count = 0
    retain_count = 0
    allowed_file_names = [
        "shortpassagea_CT.wav",
        "shortpassageb_CT.wav",
        "shortpassagec_CT.wav"
    ]

    for dir_path, _, file_names in os.walk(root_dir):
        for file_name in file_names:
            file_path = os.path.join(dir_path, file_name)
            if file_name in allowed_file_names:
                print(f"RETAINING: {file_path}")
                retain_count += 1
            else:
                print(f"DELETING: {file_path}")
                os.remove(file_path)
                deleted_count += 1

    print("\nDone.")
    print(f"{retain_count} file(s) retained.")
    print(f"{deleted_count} file(s) deleted.")


if __name__ == "__main__":
    ROOT_DIR = "datasets/abi-1-corpus"

    delete_files_by_name(ROOT_DIR)