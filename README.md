<<<<<<< HEAD
Use package: numpy, theano

Use GPU:
environment Setting:
export CUDA_HOME=[your cuda path]
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64 
PATH=${CUDA_HOME}/bin:${PATH} 
export PATH

execute command: THEANO_FLAGS='cuda.root=/usr/local/cuda,device=gpu,floatX=float32' python mlp.py [out] [in]
[out]: 39 or 1943 , it means DNN output 39 or 1943 dimension
[in]: MFCC ,fbank69 : feature tpye

Use CPU:
python mlp.py [out] [in]
[out]: 39 or 1943 , it means DNN output 39 or 1943 dimension
[in]: MFCC ,fbank69 : feature tpye
=======
ask chander and he'll tell you anything you want
>>>>>>> a396f216970101d2d9a7fc4fdbc5d87faeb69a27
