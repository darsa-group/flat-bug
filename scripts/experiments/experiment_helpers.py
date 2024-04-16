import os, sys, subprocess, yaml, re

EXEC_DIR = "/home/altair/flat-bug"
DATASETS = [
    '00-prospective-ALUS-mixed', '00-prospective-InsectCV', '00-prospective-chavez2024', '00-prospective-crall2023', 
    '01-partial-AMI-traps', '01-partial-Diopsis', '01-partial-NHM-beetles-crops', '01-partial-abram2023', '01-partial-gernat2018', 
    'ALUS', 'AMI-traps', 'AMT', 'ArTaxOr', 
    'BIOSCAN', 'CollembolAI', 'DIRT', 'DiversityScanner', 
    'Mothitor', 'PeMaToEuroPep', 
    'amarathunga2022', 'anTraX',
    'biodiscover-arm', 'biodiversa-arm', 
    'cao2022', 'pinoy2023', 
    'pitfall', 
    'scanned-sticky-cards', 'sittinger2023', 'sticky-pi',
    'ubc-pitfall-traps', 'ubc-scanned-sticky-cards'
]
DEFAULT_CONFIG = "<UNSET>"

def set_default_config(config):
    global DEFAULT_CONFIG
    DEFAULT_CONFIG = config

def get_config():
    if DEFAULT_CONFIG == "<UNSET>":
        raise RuntimeError("The default config file has not been set. Use `experiment__helpers.set_default_config()` to set it.")
    with open(DEFAULT_CONFIG, "r") as conf:
        config = yaml.load(conf, Loader=yaml.FullLoader)

    return config

TQDM_DETECT_PATTERN = re.compile(r"([\d\.]+\s*\w+\/\w+(\]|, ))")
TQDM_FINISHED_PATTERN = re.compile(r"(\| (\d+)\/\2 \[)")

def custom_print(s):
    if re.search(TQDM_DETECT_PATTERN, s) and not ("100%" in s and re.search(TQDM_FINISHED_PATTERN, s)):
            s = s.removeprefix("\n").removesuffix("\n") + "\r"
    sys.stdout.write(s)
    sys.stdout.flush()

def run_command(command, python_binary=None):
    environment = os.environ.copy()
    environment['PYTHONUNBUFFERED'] = '1'
    if python_binary is not None:
        environment["PATH"] = python_binary + ":" + environment["PATH"]
        environment["PYTHONPATH"] = python_binary
    output_buffer = ""
    with subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=environment) as process:
        while True:
            output = process.stdout.read(1)
            if not output and process.poll() is not None:
                break
            if output:
                output_buffer += output
                if output == "\n":
                    custom_print(output_buffer)
                    output_buffer = ""
        process.wait()