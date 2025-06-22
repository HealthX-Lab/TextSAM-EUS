import os

# Set the directory you want to scan
directory = 'data/EUS/test/images'

# Iterate over all files in the directory
for filename in os.listdir(directory):
    file_path = os.path.join(directory, filename)

    # Check if it's a file and starts with 'H'
    if os.path.isfile(file_path) and filename.startswith('H'):
        try:
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

directory = 'data/EUS/test/masks'

# Iterate over all files in the directory
for filename in os.listdir(directory):
    file_path = os.path.join(directory, filename)

    # Check if it's a file and starts with 'H'
    if os.path.isfile(file_path) and filename.startswith('H'):
        try:
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
