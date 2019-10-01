# Additional Instructions

## Dependencies

### NVIDIA CUDA 

This might be a deal breaker for FCIS.  The installation requires it.


```
git clone https://github.com/msracver/FCIS.git
```

Run ```sh ./init.sh```  -- on my computer I had to modify ```python``` to ```python3```.


I am currently at this point in the install script:

```
Traceback (most recent call last):
  File "setup_linux.py", line 56, in <module>
    CUDA = locate_cuda()
  File "setup_linux.py", line 44, in locate_cuda
    raise EnvironmentError('The nvcc binary could not be '
OSError: The nvcc binary could not be located in your $PATH. Either add it to your path, or set $CUDAHOME
Traceback (most recent call last):
  File "setup_linux.py", line 56, in <module>
    CUDA = locate_cuda()
  File "setup_linux.py", line 44, in locate_cuda
    raise EnvironmentError('The nvcc binary could not be '
OSError: The nvcc binary could not be located in your $PATH. Either add it to your path, or set $CUDAHOME
```

### MXNet

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

Lets attempt to proceed without GPU support for now.  I typically add ```sudo -H pip3``` instead of ```pip``` for the following:

```
pip install Cython
pip install opencv-python==3.2.0.6
pip install easydict==1.6
pip install hickle
```


## Installation from Source at Specified FCIS Version (INCOMPLETE)

At this point, the FCIS README.md says we need an NVIDIA GPU with at least 5GB memory.  So we might need that.



Clone the exact version of MXNet that the FCIS specifies.  



 FCIS recommends version 0.9.3 [MXNet@(commit 62ecb60)](https://github.com/dmlc/mxnet/tree/62ecb60), but lets see how difficult it is to build from source with GPU enabled

Lets assume we have a development folder at ~/dev.

```
cd ~/dev
git clone https://github.com/apache/incubator-mxnet.git
```