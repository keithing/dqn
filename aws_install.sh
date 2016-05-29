sudo apt-get update
sudo apt-get -y dist-upgrade
sudo apt-get install -y gcc g++ gfortran build-essential git wget linux-image-generic libopenblas-dev python-setuptools python-dev python3-dev cmake
sudo easy_install pip
sudo pip install --upgrade virtualenv
virtualenv -p python3 py3
echo -e "source ~/py3/bin/activate" >> ~/.bashrc
source ~/py3/bin/activate
which python
pip install --upgrade pip
pip install --upgrade cython
pip install --upgrade Pillow
pip install --upgrade scipy
pip install --upgrade numpy
pip install --upgrade pandas
pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git
sudo wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_7.0-28_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1404_7.0-28_amd64.deb
sudo apt-get update
sudo apt-get install -y cuda
echo -e "\nexport PATH=/usr/local/cuda/bin:$PATH\n\nexport LD_LIBRARY_PATH=/usr/local/cuda/lib64" >> .bashrc
#Theano RC here
sudo apt-get remove xserver-xorg-core
sudo reboot
