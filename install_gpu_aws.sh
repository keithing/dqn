# Requires cudnn-7.0-linux-x64-v4.0-prod.tgz to be saved in working directory
# This download requires registration with nvidia :(
apt-get update
apt-get -y dist-upgrade
apt-get install -y gcc g++ gfortran build-essential git wget linux-image-generic libopenblas-dev python-setuptools python-dev python3-dev cmake libhdf5-dev xvfb libav-tools awscli
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_7.0-28_amd64.deb
dpkg -i cuda-repo-ubuntu1404_7.0-28_amd64.deb
apt-get update
apt-get install -y cuda
tar -zxf cudnn-7.0-linux-x64-v4.0-prod.tgz && rm cudnn-7.0-linux-x64-v4.0-prod.tgz
sudo cp -R cudnn-7.0-linux-x64-v4.0-prod/lib* /usr/local/cuda/lib64/
sudo cp cudnn-7.0-linux-x64-v4.0-prod/cudnn.h /usr/local/cuda/include/
echo -e "\nexport PATH=/usr/local/cuda/bin:$PATH\n\nexport LD_LIBRARY_PATH=/usr/local/cuda/lib64" >> .bashrc
reboot
