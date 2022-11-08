# bin/sh
# Required files
# python3 required
sudo apt install python3-pip
# pip3 required
sudo pip3 install conan # Install conan and adds to path
# Is used as a c/c++ package manager
sudo conan profile new default --detect
sudo conan profile update settings.compiler.li