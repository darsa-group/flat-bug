import logging
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional

from tqdm import tqdm

REMOTE_REPOSITORY = "https://anon.erda.au.dk/share_redirect/Bb0CR1FHG6/"

# Thanks to: https://stackoverflow.com/a/53877507/19104786
class DownloadProgressBar(tqdm):
    def update_to(self, b : int=1, bsize : int=1, tsize : Optional[int]=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_from_repository(url : str, output_path : Optional[str]=None, strict : bool=True):
    if output_path is None:
        output_path = url
    url = urllib.parse.urljoin(REMOTE_REPOSITORY, url)
    try:
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=f'Downloading {url} to {output_path}') as t:
            urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
    except Exception as e:
        if not strict:
            return None
        else:
            raise e

# TODO: Improve this perhaps using https://gist.github.com/aldur/f356f245014523330a7070ab12bcfb1f, 
# as I have done in PyRemoteData https://github.com/asgersvenning/pyremotedata/blob/f0e3506c1abe2bb20106ffa2a1c3fc0f380f3dd8/src/pyremotedata/__init__.py
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

def set_log_level(level):
    logger.setLevel(level)
    logger.info(f'Log level set to {level}')