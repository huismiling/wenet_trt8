

mkdir wenet_plugin/build/
cd wenet_plugin/build/
cmake ..
make -j
cd -
cp wenet_plugin/build/libmhalugin.so . -s

