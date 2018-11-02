# helper functions to download MNIST data, unpack, reformat, and cache a local copy in numpy format
# this downloads data from Yann LeCun's MNIST page, which also describes the packed format of the files
# if unspecified, the default target directory is ./mnist

import os, sys
import numpy as np 
import urllib3
import gzip

import timeit


class mnist:
    def __init__(self):
        self.dlhostname = "http://yann.lecun.com/exdb/mnist/"
        self.dlfilelist = ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz", "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz")
        #self.lfilelist = ("mnist-train-images.txt", "mnist-train-labels.txt", "mnist-test-images.txt", "mnist-test-labels.txt")
        self.lfilelist = ("mnist-train-images", "mnist-train-labels", "mnist-test-images", "mnist-test-labels")
        self.dldatadir = "./mnist/"
        self.LABEL_FILE = 2049
        self.IMAGE_FILE = 2051
        return

    def download_raw_mnist(self, site=None, datadir=None, force_download = True):
        if site is None:
            site = self.dlhostname
        if datadir is None:
            datadir = self.dldatadir

        if os.path.exists(datadir) is False:
            print("can't find data directory", datadir)
            raise FileNotFoundError

        http = urllib3.PoolManager()
        for in_f, out_f in zip(self.dlfilelist, self.lfilelist):
            out_filepath = os.path.join(datadir, out_f)
            if os.path.exists(out_filepath ) and force_download is False:
                print("output file already exists, skipping download", out_filepath)
                continue

            fileurl = site + in_f
            print("downloading " + fileurl)
            r = http.request('GET', fileurl)
            if str(r.status) == '200':
                try:
                    with open(out_filepath, 'wb') as f:
                        print("writing to ", out_filepath)
                        f.write(r.data)
                except FileNotFoundError as e:
                    print("can't create output file", out_filepath)
                    print(e)
            else:
                print("bad response ", r.status)
                print(r.headers)
            print("done")
        return

    def _read_raw_mnist_file(self, f):
        result = dict()
        magic = int.from_bytes(f.read(4), 'big')
        labelbuf = None
        pixelbuf = None
        im_rows = 0
        im_cols = 0
        count = 0

        if magic == self.LABEL_FILE:
            count = int.from_bytes(f.read(4), 'big')
            labelbuf = f.read()

        elif magic == self.IMAGE_FILE:
            count = int.from_bytes(f.read(4), 'big')
            im_rows = int.from_bytes(f.read(4), 'big')
            im_cols = int.from_bytes(f.read(4), 'big')
            pixelbuf = f.read()
        else:
            raise ValueError("bad magic number in input stream")
        
        result['magic'] = magic
        result['count'] = count
        result['rows'] = im_rows
        result['cols'] = im_cols
        result['labels'] = labelbuf
        result['data'] = pixelbuf

        return result

    # read raw files, expand labels to 1-hot, and save in numpy and/or text format
    def inflate_mnist(self, datadir=None, txtformat=False):

        for in_f, out_f in zip(self.dlfilelist, self.lfilelist):
            # read in raw file
            result = dict()
            in_filepath = os.path.join(datadir,in_f)
            out_filepath = os.path.join(datadir, out_f)
            print("reading from "+ in_filepath + ", output will be " + out_filepath + ".npy")
            with gzip.open(in_filepath, 'rb') as f:
                if f is not None:
                    result.update(self._read_raw_mnist_file(f))
                    print("magic = {0}, count = {1}".format(result['magic'], result['count']))
                    print("im_rows, im_cols = {0}, {1}".format(result['rows'], result['cols']))

            # write out text or binary numpy format
            if result['magic'] == self.IMAGE_FILE:
                # image data in rows of 784 unsigned byte integers
                data = np.frombuffer(result['data'], dtype=np.uint8)
                data = data.reshape((data.size//784, 784))
                print("data = " + repr(data) + " shape = ", data.shape)
                if txtformat is True:
                    np.savetxt(out_filepath+".txt", data, fmt='%d')
                else:
                    np.save(out_filepath+".npy", data)
                print("finished writing to file")

            elif result['magic'] == self.LABEL_FILE:
                labels = np.frombuffer(result['labels'], dtype=np.uint8)
                # expand class labels from digits [0-9] to 1-hot row vector
                I = np.identity(10)
                labels = I[labels]
                print("data = " + repr(labels) + " shape = ", labels.shape)
                if txtformat is True:
                    np.savetxt(out_filepath + ".txt", labels, fmt='%d')
                else:
                    np.save(out_filepath + ".npy", labels)

                print("finished writing to file")
            #pass

    def load_mnist(self, datadir=None, testset = False):
        if datadir is None:
            datadir = self.dldatadir

        result = dict()

        try:
            if testset is False:
                filepath = os.path.join(datadir, self.lfilelist[0])+".npy"
#                data = np.loadtxt(filepath)
                data = np.load(filepath)
                result['data'] = data
                filepath = os.path.join(datadir, self.lfilelist[1])+".npy"
#                labels = np.loadtxt(filepath)
                labels = np.load(filepath)
                result['labels'] = labels
            else:
                filepath = os.path.join(datadir, self.lfilelist[2])+".npy"
#                data = np.loadtxt(filepath)
                data = np.load(filepath)
                result['data'] = data
                filepath = os.path.join(datadir, self.lfilelist[3])+".npy"
#                labels = np.loadtxt(filepath)
                labels = np.load(filepath)
                result['labels'] = labels
        
        except FileNotFoundError as e:
            print("can't read formatted MNIST input file from", datadir)
            print (e)

        return result
    
if __name__ == '__main__':
    datadir = os.path.join(os.path.sep, "Users", "hjl", "Downloads", "mnist")
    m = mnist()
#    m.download_raw_mnist(datadir=datadir)
    m.inflate_mnist(datadir, txtformat=False)
    result = m.load_mnist(datadir)
    print(repr(result['data']))


