# bin/sh
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
sudo conan profile update settings.compiler.li


# For automatic documentation generation
# sudo apt-get install doxygen