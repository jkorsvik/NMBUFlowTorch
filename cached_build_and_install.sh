cd build

installhere=$(bash -c 'read -n1 -p "Install in this dir? if no --> user install [y,n]"  tmp; echo $tmp')
echo "\n"

case $installhere in  
  y|Y) echo "INSTALLING in install folder \n" && cmake .. -DCMAKE_INSTALL_PREFIX=../install && sudo cmake --build . --target install;; 
  n|N) echo "USER INSTALL \n" && cmake .. && sudo cmake --build . --target install;; 
  *) echo dont know ;; 
esac

#sudo cmake --build . --target install
cd ..