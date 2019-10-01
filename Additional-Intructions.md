# Additional Instructions

## Dependencies

### NVIDIA CUDA 

This might be a deal breaker for FCIS.  The installation requires it.

#### On Ubuntu with NVIDIA GPU

My Kubuntu desktop has a NVIDIA GeForce GTX 745 system with 4 GB memory (still not as much as FCIS recommends, but better than nothing)

```
kiron@kiron-XPS-8900:~$ nvidia-smi
Tue Oct  1 07:58:44 2019       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 390.116                Driver Version: 390.116                   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 745     Off  | 00000000:01:00.0  On |                  N/A |
| 21%   49C    P0    N/A /  N/A |    727MiB /  4038MiB |      1%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0      1092      G   /usr/lib/xorg/Xorg                           325MiB |
|    0      1478      G   kwin_x11                                      64MiB |
|    0      1483      G   /usr/bin/krunner                              16MiB |
|    0      1485      G   /usr/bin/plasmashell                         201MiB |
|    0     18822      G   ...uest-channel-token=10668720831531840875    59MiB |
|    0     19158      G   /usr/bin/systemsettings5                      30MiB |
|    0     21297      G   ...-token=2D75ED3F2C7FA6576E00C4B60A1A3325    23MiB |
+-----------------------------------------------------------------------------+
```

#### Installing the CUDA 9.2 Toolkit

If you type ```nvcc``` at a terminal and get this message, you don't have the toolkit installed
```
kiron@kiron-XPS-8900:~/dev/InstanceSegmentation_Sentinel2$ nvcc

Command 'nvcc' not found, but can be installed with:

sudo apt install nvidia-cuda-toolkit
```
While it is possible to use ```apt``` to install CUDA, this [link](https://askubuntu.com/questions/799184/how-can-i-install-cuda-on-ubuntu-16-04) doesn't recommend it because "Notes: Yes, there is the possibility to install it via apt-get install cuda. I strongly suggest not to use it, as it changes the paths and makes the installation of other tools more difficult."


MXNet recommended version 9.2 of the toolit (which is a legacy download as of Oct 1, 2019).  I downloaded the local .deb at around 1.2 GB.  Instructions [here](https://developer.nvidia.com/cuda-92-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1710&target_type=deblocal)


In trying to follow their instructions, this is where I left off:

```
Unpacking cuda (9.2.148-1) ...
Errors were encountered while processing:
 /tmp/apt-dpkg-install-lXty4J/098-nvidia-396_396.37-0ubuntu1_amd64.deb
E: Sub-process /usr/bin/dpkg returned an error code (1)
```




## Install FCIS





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