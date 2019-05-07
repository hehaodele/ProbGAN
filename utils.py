from torch_head import *
from common_head import *


# ======================================================================================================================
def to_tensor(x):
    return torch.tensor(x).to(torch.float).cuda()


def to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


# ======================================================================================================================
class Tee(object):
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self

    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()

    def close(self):
        self.__del__()


def pickle_save(filename, obj):
    import pickle
    pickle.dump(obj, open(filename, 'wb'), protocol=4)


def pickle_load(filename):
    import pickle
    return pickle.load(open(filename, 'rb'))