# mnist_helper

## Download, unpack, and reformat MNIST dataset

To use:
```
import mnist_helper
```
First time only, to download the data:  
```
m = mnist_helper.mnist()  
datapath='/path/to/local/datadir/'  
m.download_raw_mnist(datadir=datapath)  
m.inflate_mnist(datadir=datapath, txtformat=False)
```
Subsequent calls to download_raw_mnist() will check for presence of raw files before trying again.  
  
inflate_mnist() can be called with txtformat=True, which will result in both Numpy (.npy) and text (.txt) files being created. Text format is the most portable, but is much slower than Numpy binary format.
```
result = m.load_mnist(datadir)
```
See also: http://yann.lecun.com/exdb/mnist/
