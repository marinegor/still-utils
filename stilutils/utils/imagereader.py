import h5py
import cbf
import os
import numpy as np
from abc import ABC, abstractmethod


class ImageReader(ABC):
    @abstractmethod
    def next_reader(self, reader):
        pass

    @abstractmethod
    def get_image(self, path):
        pass


class AbstractImageReader(ImageReader):

    _next_reader = None

    def next_reader(self, reader):
        self._next_reader = reader
        return reader

    @abstractmethod
    def get_image(self, path):
        if self._next_reader:
            return self._next_reader.get_image(path)

        return None


class CXIReader(AbstractImageReader):
    def __init__(self, path_to_data=None):
        if path_to_data:
            self.path_to_data = path_to_data
        else:
            self.path_to_data = "/entry_1/data_1/data"

    def get_image(self, path):
        if not path.startswith("/"):
            path = os.getcwd() + path
        if (" //" in path) and path.split(" //")[0].endswith(".cxi"):
            return self._get_event(path)
        else:
            return super().get_image(path)

    def _get_event(self, path):
        cxi_path, event = path.split(" //")
        event = int(event)
        with h5py.File(cxi_path, "r") as dataset:
            data = dataset[self.path_to_data]
            # For better performance direct read could be used, but it needs file to be C-contigious!
            # image = np.ones((data.shape[1], data.shape[2]), dtype='int32')
            # data.read_direct(image, np.s_[event, :, :], np.s_[:])
            # return image
            return data[event]

    def get_events_number(self, path):
        with h5py.File(path, "r") as dataset:
            return dataset[self.path_to_data].shape[0]


class CBFReader(AbstractImageReader):
    def get_image(self, path):
        if path.endswith(".cbf"):
            return cbf.read(path).data
        else:
            return super().get_image(path)


class H5Reader(AbstractImageReader):
    def __init__(self, path_to_data=None):
        if path_to_data:
            self.path_to_data = path_to_data
        else:
            self.path_to_data = "/data/rawdata0"

    def get_image(self, path):
        if path.endswith(".h5"):
            with h5py.File(path, "r") as fle:
                return fle[self.path_to_data][:, :]
        else:
            return super().get_image(path)


class ImageLoader:
    """
    ImageLoader class that abstracts the image loading process,
    loading images one by one and returning a handle to write them
    """

    def __init__(self, input_list, chunksize=100, cxi_path=None, h5_path=None):
        self.input_list = input_list
        self.chunksize = chunksize

        # initialize chain of image readers
        self.cxi_reader = CXIReader(path_to_data=cxi_path)
        self.cbf_reader = CBFReader()
        self.h5_reader = H5Reader(path_to_data=h5_path)
        self.cxi_reader.next_reader(self.cbf_reader).next_reader(self.h5_reader)
        self.image_reader = self.cxi_reader

        # load all frames from input list
        data = set()
        with open(input_list, mode="r") as fin:
            for line in fin:
                if line.rstrip().endswith(".cxi"):
                    num_events = self.cxi_reader.get_events_number(line.rstrip())
                    for i in range(num_events):
                        data.add(line.rstrip() + " //" + str(i))
                else:
                    data.add(line.rstrip())

        self._data = list(data)

    def __iter__(self):
        return self

    def __next__(self, mode="r"):
        """Here the magic happens that helps to iterate"""
        """
        Pseudocode:

        data_to_return, handles_to_return = self.data[:chunksize]
        self.data = self.data - data_to_return
        return data_to_return, handles_to_return
        """
        current_chunk_list = self._data[: self.chunksize]
        if len(current_chunk_list) == 0:
            raise StopIteration
        result = []
        for event in current_chunk_list:
            result.append(self.image_reader.get_image(event))
        self._data = self._data[self.chunksize :]
        result = np.stack(result, axis=0)
        return current_chunk_list, result


if __name__ == "__main__":
    pass
