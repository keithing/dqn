apt-get update
apt-get -y dist-upgrade
apt-get install -y gcc g++ gfortran build-essential git wget linux-image-generic libopenblas-dev python-setuptools python-dev python3-dev cmake
easy_install pip
pip install --upgrade virtualenv
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_7.0-28_amd64.deb
dpkg -i cuda-repo-ubuntu1404_7.0-28_amd64.deb
apt-get update
apt-get install -y cuda
echo -e "\nexport PATH=/usr/local/cuda/bin:$PATH\n\nexport LD_LIBRARY_PATH=/usr/local/cuda/lib64" >> .bashrc
#Theano RC here
reboot

# After reboot, run following commands
# virtualenv -p python3 py3
# source ~/.py3/bin/activate
# echo -e "\nsource ~/.py3/bin/activate"
# pip install -e dqn
