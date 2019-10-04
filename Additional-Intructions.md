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



## Appendix: Configuration of a Jetson TX1

The NVIDIA documentation is out there, but I feel like they could have done a revision or two to make it better.  I am copy/pasting the [quick start guide](
http://developer.download.nvidia.com/embedded/L4T/r23_Release_v1.0/l4t_quick_start_guide.txt) here, with some added notes/formatting.

### NVIDIA TEGRA LINUX DRIVER PACKAGE QUICK-START GUIDE

The information here is intended to help you quickly get started using NVIDIA Tegra Linux Driver package (L4T).  Also, if the SDK Manager's file are ona non-ext4 drive, it will have the same failures.  But the SDK Manager's settings do not allow you to specify where the files are downloaded.

#### ASSUMPTIONS:

* You have an NVIDIA Jetson TX1 Developer Kit, equipped with the Jetson TX1 module.
* You have a host machine that is running Linux.
* Your developer system is cabled as follows:
     - USB Micro-A cable connecting Jetson TX1 carrier board (USB0) to your Linux host for flashing. Note: this can be a standard micro USB phone cable, but ensure it is high quality and capable of transferring data
     - (Not included in the developer kit) To connect USB peripherals such as keyboard and mouse, a USB hub should be connected to the USB port (USB1) on the Jetson TX1 carrier board. Note: I am using an Amazon basics externally powered hub

The following directions will create a 14 GB partition on the eMMC device (internal storage) and will flash the root file system to that location.
If you would like to have network access on your target (e.g., for installing additional packages), ensure an Ethernet cable is attached to the Jetson TX1 carrier board.

### INSTRUCTIONS:

HUGE NOTE: All the downloaded files need to be put on a hard drive that is ```ext4``` - if this is not followed, you will have a broken install on your TX1.

#### Download and prepare the files

Download the latest L4T release package for your developer system and the sample file system from https://developer.nvidia.com/linux-tegra

Note: Download the L4T Driver Package (BSP) (~150 MB) and the Sample Root Filesystem (1.2 GB)

If NVIDIA does not yet provide public release for the developer system you have, please contact your NVIDIA support representative to obtain the latest L4T release package for use with the developer board.

Untar the files and assemble the rootfs (this took quite a while for me):

```
sudo tar xpf Tegra210_Linux_R23.1.1_armhf.tbz2 
cd Linux_for_Tegra/rootfs/ 
sudo tar xpf ../../Tegra_Linux_Sample-Root-Filesystem_R23.1.1_armhf.tbz2 
cd ../ 
sudo ./apply_binaries.sh
```

#### Flash the rootfs onto the system's internal eMMC.

a) Put your system into "reset recovery mode" by holding down the REC (S3) button and press the RST (S1) button once on the carrier board. b) Ensure your Linux host system is connected to the carrier board through the USB Micro-A cable. The flashing command is:

```
sudo ./flash.sh jetson-tx1 mmcblk0p1
```

This will take several minutes. (Started 7:37 AM -- had breakfast, came back down at 8 AM and it was done...)

you will see this:

```
[ 157.5007 ] tegradevflash --write BCT P2180_A00_LP4_DSC_204Mhz.bct
[ 157.5021 ] Cboot version 00.01.0000
[ 157.5043 ] Writing partition BCT with P2180_A00_LP4_DSC_204Mhz.bct
[ 157.5049 ] [................................................] 100%
[ 157.9248 ] 
[ 157.9249 ] Flashing completed

[ 157.9250 ] Coldbooting the device
[ 157.9266 ] tegradevflash --reboot coldboot
[ 157.9286 ] Cboot version 00.01.0000
[ 157.9313 ] 
*** The target t210ref has been flashed successfully. ***
Reset the board to boot from internal eMMC.

```

The target will automatically reboot upon completion of the flash (I just power cycled for good measure -- hold down power for 10 seconds, wait 10 seconds, then press power again).

You now have Linux running on your developer system. Depending on the sample file system used, you will see one of the following on the screen:

The Ubuntu graphical desktop. 

Notes: Before the ext4 correction, this is what I saw, and I configured my username/password, language, then the configuration window went missing, and my graphical desktop had literally nothing on it, and no taskbar.

After the ext4 change, there were two more dialogs that showed progress, and the setup properly finished and then rebooted.  Wow, nowhere in the documentation is that expressed.

The command prompt. Log in as user login:ubuntu and password:ubuntu. See step 5 if you wish to configure the graphical desktop on your setup.

#### Installing the graphical desktop on your target board (if not already installed):

a) Connect Ethernet to target via the RJ45 connector.

b) Acquire an IP address:

sudo dhclient <interface>

where <interface> is eth0.

c) Check to see if Ethernet is up and running. You should see an IP address associated with eth0.

ifconfig sudo apt-get update sudo apt-get install ubuntu-desktop

d) Reboot and the system will boot to the graphical desktop.

NOTE: the above steps can be used to install other packages with "sudo apt-get install".

Please refer to the release notes provided with your software for up-to-date information on platform features and use.


#### Installing CUDA Toolkit on Jetson (TODO)