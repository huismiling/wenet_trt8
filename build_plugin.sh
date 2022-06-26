


git submodule update --init

cd FasterTransformer_wenet
sh build_ft.sh
cd -
rm -rf ./libwenet_plugin.so
ln -s FasterTransformer_wenet/build/lib/libwenet_plugin.so .

