# bin/sh
CWD=$(pwd)
# Required files
# Cmake for ubuntu
sudo wget -qO /etc/apt/trusted.gpg.d/kitware-key.asc https://apt.kitware.com/keys/kitware-archive-latest.asc
echo "deb https://apt.kitware.com/ubuntu/ focal main" | sudo tee /etc/apt/sources.list.d/kitware.list
sudo apt update
sudo apt install -y cmake
cmake --version
# python3 required
sudo apt install python3-pip
# pip3 required
sudo pip3 install conan # Install conan and adds to path
# Is used as a c/c++ package manager
sudo conan profile new default --detect
sudo conan profile update settings.compiler.libcxx=libstdc++11 default
sudo conan profile update env.CC=clang default
sudo conan profile update env.CXX=clang++ default
# Add eigen to include
curl -O "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip" && unzip -d include/ eigen-3.4.0.zip && rm eigen-3.4.0.zip
mv $CWD/include/eigen-3.4.0 $CWD/include/eigen3
#sudo conan install eigen_recipe.py -g=cmake_find_package 


# For automatic documentation generation
# sudo apt-get install doxygen