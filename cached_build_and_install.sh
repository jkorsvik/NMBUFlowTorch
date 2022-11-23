cd build

echo "Cmake cached build..."
echo 
cmake .. && sudo cmake --build . --target install

#sudo cmake --build . --target install
cd ..