sudo rm -rf build/
mkdir build/ && cd build/
#cmake .. #-DCMAKE_INSTALL_PREFIX=~/install # -DCMAKE_C_COMPILER:FILEPATH=/usr/bin/clang-12 -DCMAKE_CXX_COMPILER:FILEPATH=/usr/bin/clang++-12 # -DCMAKE_BUILD_TYPE:STRING=Release


installhere=$(bash -c 'read -n1 -p "Install in this dir? if no --> user install, (S is 03 user install) [y,n,S]"  tmp; echo $tmp')
echo "\n"

case $installhere in  
  y|Y) echo "INSTALLING in install folder \n" && cmake .. -DCMAKE_INSTALL_PREFIX=../install && sudo cmake --build . --target install;; 
  s|S) echo "USER INSTALL \n" && cmake .. -DCMAKE_BUILD_TYPE:STRING=Release -DCMAKE_CXX_FLAGS:STRING="-fopenmp -O3"  && sudo cmake --build . --target install;; 
  n|N) echo "USER INSTALL \n" && cmake .. -DCMAKE_BUILD_TYPE:STRING=Debug -DCMAKE_CXX_FLAGS:STRING="-fopenmp" && sudo cmake --build . --target install;; 
  *) echo dont know ;; 
  # -fopenblas"
esac

#sudo cmake --build . --target install
cd ..