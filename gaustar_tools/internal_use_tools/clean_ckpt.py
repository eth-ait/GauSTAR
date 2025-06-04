import os
import torch
from tqdm import tqdm

def list_folders(directory):
    """
    List all folders in the given directory with their full paths, sorted alphabetically.

    :param directory: Path to the folder to search in.
    :return: List of full paths to the folders.
    """
    try:
        # Get a list of folders and check if they are directories
        folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]

        # Sort the folders alphabetically
        folders_sorted = sorted(folders)

        # Construct full paths
        folders_full_path = [os.path.join(directory, folder) for folder in folders_sorted]

        # print("Folders in '{}' (with full paths, sorted):".format(directory))
        # for folder_path in folders_full_path:
        #     print(folder_path)

        return folders_full_path

    except FileNotFoundError:
        print("The directory '{}' does not exist.".format(directory))
        return []
    except PermissionError:
        print("You do not have permission to access '{}'.".format(directory))
        return []



# Example usage:
if __name__ == "__main__":
    # Replace this with the folder path you want to list
    target_directory = "/mnt/euler/SUGAR/SuGaR/output/"
    # target_directory = "/mnt/euler/SUGAR/SuGaR/output/"
    file_list = ["2000.pt", "1000.pt"]
    res_list = list_folders(target_directory)[:1]

    print(res_list)

    for res_dir in res_list:
        frame_list = list_folders(res_dir)

        for frame_dir in tqdm(frame_list):
            for file_name in file_list:
                file_path = os.path.join(frame_dir, file_name)
                if os.path.exists(file_path) and os.path.isfile(file_path):
                    ckpt = torch.load(file_path, map_location=torch.device('cpu'))
                    if 'optimizer_state_dict' in ckpt:
                        ckpt.pop('optimizer_state_dict')
                        torch.save(ckpt, file_path)
                        print("clean: ", file_path)
