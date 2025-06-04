import os
import shutil

path = "//output/track_240906T3_update_NC01/"
# dirs = os.listdir(path)
# s = []

dirs = [f"{i:04d}" for i in range(100, 290, 1)]

def safe_delete_folder(delete_path):
    # delete_path = os.path.join(path, dir, f'{i:04d}/2000.pt')
    if os.path.exists(delete_path):
        print(delete_path)
        shutil.rmtree(delete_path)
        # os.remove(delete_path)
    # if not os.path.isdir(file):

for dir in dirs:
    delete_path = os.path.join(path, dir, f'extract')
    safe_delete_folder(delete_path)
    delete_path = os.path.join(path, dir, f'detect/render_001000')
    safe_delete_folder(delete_path)
    delete_path = os.path.join(path, dir, f'detect/visual_001000')
    safe_delete_folder(delete_path)

