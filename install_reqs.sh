# bin/bash
CWD=$(pwd)
# Required files
# Update gcc and cmake
sudo apt update
sudo apt install build-essential
# GDB for debugging
sudo apt-get install gdb

# install gdb extensions as well for eigen
TEMP="$CWD/_gdb_extension"
cat _gdb_extension/gdbinit.txt > _gdb_extension/gdbinit.txt.temp
pattern="s%<REPLACEMEWITHPATH>%${TEMP}%g"
sed -i $pattern _gdb_extension/gdbinit.txt.temp  
cat _gdb_extension/gdbinit.txt.temp > ~/.gdbinit

# It aint stupid if it works

# BLAS & LAPACK for blazing fast dense matrix operations, supported by Eigen
apt search openblas
sudo apt-get install libblas-dev liblapack-dev
sudo apt-get install libopenblas-dev 
sudo update-alternatives --config libblas.so.3

# clang-tidy for linting
sudo apt-get install clang-tidy 

# cpp check for static analysis
sudo apt-get install cppcheck

# unzip
sudo apt install unzip
# Cmake for ubuntu
sudo wget -qO /etc/apt/trusted.gpg.d/kitware-key.asc https://apt.kitware.com/keys/kitware-archive-latest.asc
echo "deb https://apt.kitware.com/ubuntu/ focal main" | sudo tee /etc/apt/sources.list.d/kitware.list
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
# This is uneeded as conan fixes this for us
#curl -O "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip" && unzip -d include/ eigen-3.4.0.zip && rm eigen-3.4.0.zip
#mv $CWD/include/eigen-3.4.0 $CWD/include/eigen3
#sudo conan install eigen_recipe.py -g=cmake_find_package 

# boost lib for
sudo apt-get install libboost-all-dev

# CLANG for linting
sudo apt install clang-12 --install-suggests


# For automatic documentation generation
sudo apt-get install doxygen