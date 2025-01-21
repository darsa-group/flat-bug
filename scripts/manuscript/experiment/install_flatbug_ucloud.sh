#!/bin/bash

time_str () {
  date +"%d-%m-%Y_%H-%M-%S"
}

INIT_DIR="/work/fb_init"
FB_DIR="$INIT_DIR/fb_yolo"

###########################################################################
########## Uncomment as necessary for cloning private repository ##########
# # SSH Config
# echo "($(time_str) Setting up SSH...)"
# mkdir -p "$HOME/.ssh"
# chmod 700 "$HOME/.ssh"
# cat <<EOF >"$HOME/.ssh/config"
# Host github.com github
#   User git
#   Hostname github.com
#   PreferredAuthentications publickey
#   IdentityFile $HOME/.ssh/flatbug_readonly

# EOF

# # Copy keys to the .ssh directory and set the proper permissions
# cp "$INIT_DIR/flatbug_readonly" "$HOME/.ssh/flatbug_readonly"
# cp "$INIT_DIR/flatbug_readonly.pub" "$HOME/.ssh/flatbug_readonly.pub"
# chmod 600 "$HOME/.ssh/flatbug_readonly"
# chmod 644 "$HOME/.ssh/flatbug_readonly.pub"

# # Start an SSH agent and add the deployment key to it
# eval "$(ssh-agent -s)"
# ssh-add "$HOME/.ssh/flatbug_readonly"

# # Add github.com and io.erda.au.dk (at port 2222) to known hosts
# ssh-keyscan github.com >> "$HOME/.ssh/known_hosts"
# ssh-keyscan -p 2222 io.erda.au.dk >> "$HOME/.ssh/known_hosts"
# chmod 644 "$HOME/.ssh/known_hosts"
###########################################################################
###########################################################################

# Install LFTP (for storing results on ERDA)
sudo add-apt-repository universe
sudo apt-get update
sudo apt-get install -y lftp

### Install and setup flat-bug experiment environment
## Install Python 3.12
# Not necessary for UCloud (`Terminal` application 03/10/2024) comes with Python 3.12.3 aliased as 'Python'

## Install libgl (for openCV)
sudo apt-get install -y libgl1

## Install R (see https://cran.r-project.org/bin/linux/ubuntu/)
echo "($(time_str)) Installing R..."
# install two helper packages we need
sudo apt install --no-install-recommends -y software-properties-common dirmngr
# add the signing key (by Michael Rutter) for these repos
# To verify key, run gpg --show-keys /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc 
# Fingerprint: E298A3A825C0D65DFD57CBB651716619E084DAB9
wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | sudo tee -a /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc
# add the R 4.0 repo from CRAN -- adjust 'focal' to 'groovy' or 'bionic' as needed
sudo add-apt-repository "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran40/"
# install R
sudo apt install --no-install-recommends -y r-base
# Install our R dependencies (and their binary dependencies)
sudo apt install -y libssl-dev libcurl4-openssl-dev unixodbc-dev libxml2-dev libmariadb-dev libfontconfig1-dev libharfbuzz-dev libfribidi-dev libfreetype6-dev libpng-dev libtiff5-dev libjpeg-dev
sudo add-apt-repository -y ppa:c2d4u.team/c2d4u4.0+ 
sudo apt install -y r-cran-data.table r-cran-magrittr r-cran-ggplot2 r-cran-scales
sudo Rscript -e 'install.packages(c("optparse"), repos="https://cloud.r-project.org/")'
echo "($(time_str)) Finished installing R."

## Install PyTorch and Flat-Bug
echo "($(time_str)) Installing flat-bug..."
# Clone flat-bug repository
if ! git clone git@github.com:darsa-group/flat-bug.git; then
    echo "Failed to clone repository"
    exit 1
fi
# Install PyTorch (using 2.3.1 for better compatibility)
pip install "torch>=2.3.1" torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
# Install flat-bug from source
cd flat-bug
git checkout dev_experiments
pip install -e .
pip install submitit
echo "($(time_str)) Finished installing flat-bug."

# Check integrity of datasets
echo "($(time_str)) Checking dataset integrity..."
# fb_yolo should contain 12297 files and use 8055248229 bytes of space
fb_yolo_expected_files="12289"
fb_yolo_expected_size="8055248229"

fb_yolo_files=$(find "$FB_DIR" -type f | wc -l)
fb_yolo_size=$(du -sb "$FB_DIR" | cut -f1)

if [[ $fb_yolo_size != $fb_yolo_expected_size || $fb_yolo_files != $fb_yolo_expected_files ]]; then
  echo "($(time_str)) Error: Size of 'fb_yolo' is $fb_yolo_size from $fb_yolo_files, but expected size $fb_yolo_expected_size from $fb_yolo_expected_files."
fi

echo "($(time_str)) Datasets checked."

