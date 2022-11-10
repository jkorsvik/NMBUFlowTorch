rm -rf build/
mkdir build/ && cd build/
cmake .. -DCMAKE_INSTALL_PREFIX=~/install # -DCMAKE_C_COMPILER:FILEPATH=/usr/bin/clang-12 -DCMAKE_CXX_COMPILER:FILEPATH=/usr/bin/clang++-12 # -DCMAKE_BUILD_TYPE:STRING=Release
cmake --build . --target install
cd ..