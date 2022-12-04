sudo rm -rf build

mkdir build/ && cd build/
cmake .. -Dnmbuflowtorch_ENABLE_DOXYGEN=1 
cmake --build . --target doxygen-docs
cd ..