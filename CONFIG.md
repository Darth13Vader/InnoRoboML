# Config

[![DM](https://raw.githubusercontent.com/DormantMan/KlgEdu/master/thumb.png)](https://dormantman.tilda.ws)

### Installing TensorFlow on Windows
[TensorFlow Docs for installing on Windows](https://www.tensorflow.org/install/install_windows)

To install TensorFlow, start a terminal. Then issue the appropriate pip3 install command in that terminal. To install the CPU-only version of TensorFlow, enter the following command:
```cmd
pip install tensorflow
```

Requirements to run TensorFlow with GPU support
If you are installing TensorFlow with GPU support using one of the mechanisms described in this guide, then the following NVIDIA software must be installed on your system:

 - CUDA® Toolkit 9.0. For details, see NVIDIA's documentation Ensure that you append the relevant Cuda pathnames to the %PATH% environment variable as described in the NVIDIA documentation.
The NVIDIA drivers associated with CUDA Toolkit 9.0.
 - cuDNN v7.0. For details, see NVIDIA's documentation. Note that cuDNN is typically installed in a different location from the other CUDA DLLs. Ensure that you add the directory where you installed the cuDNN DLL to your %PATH% environment variable.
 - GPU card with CUDA Compute Capability 3.0 or higher for building from source and 3.5 or higher for our binaries. See NVIDIA documentation for a list of supported GPU cards.

You must download specific tensorflow-gpu.
[GitHub Repo for Specific Architecture](https://github.com/fo40225/tensorflow-windows-wheel)

To install the GPU version of TensorFlow, enter the following command:
```cmd
pip install "Download URL [Specific tensorflow-gpu windows wheel]"
```

Test the ability to run on the GPU:
```python
import os
from tensorflow.python.client import device_lib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Disabling notifications from TensorFlow

for i in device_lib.list_local_devices():
    print(i.device_type, '[{}]'.format(i.name), 
          '{} MB'.format(i.memory_limit // 1024 ** 2))
    if i.device_type == 'GPU':
        print(i.physical_device_desc)
    print()
```



**Good luck with coding!**
