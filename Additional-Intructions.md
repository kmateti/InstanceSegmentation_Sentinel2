# Additional Instructions

## Clone the exact version of MXNet that the FCIS specifies.  

As of October 1, 2019, MXNet is at version 1.5, and can be installed relatively automatically via ```pip``` but GPU support is disabled by default.

So, for me, at a terminal
```
sudo -H pip3 install mxnet-mkl
```
installed the most recent MXNet for python, and I was able to run
```
python3
Python 3.5.2 (default, Jul 10 2019, 11:58:48) 
[GCC 5.4.0 20160609] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import mxnet as mx
>>> a = mx.nd.ones((2, 3))
>>> b = a * 2 + 1
>>> b.asnumpy()
array([[ 3.,  3.,  3.],
       [ 3.,  3.,  3.]], dtype=float32)
```

To enable GPU support, you have to have an NVIDIA GPU and the CUDA dependencies met. There are additional instructions for building on Ubuntu 16.04 (only one with instructions) [here](https://mxnet.apache.org/get_started/ubuntu_setup.html)

## Installation from Source at Specified FCIS Version (INCOMPLETE)

 FCIS recommends version 0.9.3 [MXNet@(commit 62ecb60)](https://github.com/dmlc/mxnet/tree/62ecb60), but lets see how difficult it is to build from source with GPU enabled

Lets assume we have a development folder at ~/dev.

```
cd ~/dev
git clone https://github.com/apache/incubator-mxnet.git
```