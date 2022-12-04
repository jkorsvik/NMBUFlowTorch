sudo rm -rf build
sudo rm -rf docs

mkdir build/ && cd build/
cmake .. -Dnmbuflowtorch_ENABLE_DOXYGEN=1 
cmake --build . --target doxygen-docs
cd ..