import os
import re
from urllib.request import urlretrieve


def file_is_lfs_or_erda_pointer(file):
    with open(file, "r") as f:
        try:
            return bool(re.search(r"git-lfs\.github\.com|ERDA Pointer", f.read()))
        except UnicodeDecodeError:
            return False
    
def check_file_with_remote_fallback(file, file_storage : str="https://anon.erda.au.dk/share_redirect/ecgKtuRWe5"):
    if not os.path.exists(file) or file_is_lfs_or_erda_pointer(file):
        remote_uri = f"{file_storage}/{os.path.basename(file)}"
        try:
            urlretrieve(remote_uri, file)
        except Exception as e:
            raise type(e)(f"Failed to download test file {file} from remote file storage ({remote_uri}), perhaps the file is not available." + str(e))
    return file