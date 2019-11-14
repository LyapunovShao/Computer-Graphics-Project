#/bin/bash
/home/shaoyuming/cudatoolkit10.0/bin/nvcc pointSIFT.cu -o pointSIFT_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

# TF1.4
g++ -std=c++11 main.cpp pointSIFT_g.cu.o -o tf_pointSIFT_so.so -shared -fPIC -I /home/shaoyuming/anaconda3/lib/python3.6/site-packages/tensorflow_core/include -I /home/shaoyuming/cudatoolkit10.0/include -I /home/shaoyuming/anaconda3/lib/python3.6/site-packages/tensorflow_core/include/external/nsync/public -lcudart -L /home/shaoyuming/cudatoolkit10.0/lib64/ -L/home/shaoyuming/anaconda3/lib/python3.6/site-packages/tensorflow_core -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0