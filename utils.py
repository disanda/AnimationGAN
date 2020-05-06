import os, gzip, torch
import torch.nn as nn
import numpy as np
import scipy.misc
import imageio
import matplotlib.pyplot as plt
from torchvision import datasets, transforms


#-----------------------data----------------------

def load_mnist(dataset):
    data_dir = os.path.join("./data", dataset)
    def extract_data(filename, num_data, head_size, data_size):
        with gzip.open(filename) as bytestream:
            bytestream.read(head_size)
            buf = bytestream.read(data_size * num_data)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float)
        return data
    data = extract_data(data_dir + '/train-images-idx3-ubyte.gz', 60000, 16, 28 * 28)
    trX = data.reshape((60000, 28, 28, 1))
    data = extract_data(data_dir + '/train-labels-idx1-ubyte.gz', 60000, 8, 1)
    trY = data.reshape((60000))
    data = extract_data(data_dir + '/t10k-images-idx3-ubyte.gz', 10000, 16, 28 * 28)
    teX = data.reshape((10000, 28, 28, 1))
    data = extract_data(data_dir + '/t10k-labels-idx1-ubyte.gz', 10000, 8, 1)
    teY = data.reshape((10000))
    trY = np.asarray(trY).astype(np.int)
    teY = np.asarray(teY)
    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0).astype(np.int)
    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)
    y_vec = np.zeros((len(y), 10), dtype=np.float)
    for i, label in enumerate(y):
        y_vec[i, y[i]] = 1
    X = X.transpose(0, 3, 1, 2) / 255.
    # y_vec = y_vec.transpose(0, 3, 1, 2)
    X = torch.from_numpy(X).type(torch.FloatTensor)
    y_vec = torch.from_numpy(y_vec).type(torch.FloatTensor)
    return X, y_vec

def load_celebA(dir, transform, batch_size, shuffle):
    # transform = transforms.Compose([
    #     transforms.CenterCrop(160),
    #     transform.Scale(64)
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    # ])
    # data_dir = 'data/celebA'  # this path depends on your computer
    dset = datasets.ImageFolder(dir, transform)
    data_loader = torch.utils.data.DataLoader(dset, batch_size, shuffle, drop_last=True)
    return data_loader

#-----------------------network-----------------------

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')


#-------------------img&frame----------------------

def generate_animation(path, num):
    images = []
    for e in range(num):
        img_name = path + '_epoch%03d' % (e+1) + '.png'
        images.append(imageio.imread(img_name))
    imageio.mimsave(path + '_generate_animation.gif', images, fps=5)

def loss_plot(hist, path = 'Train_hist.png', model_name = ''):
    x = range(len(hist['D_loss']))
    y1 = hist['D_loss']
    y2 = hist['G_loss']
    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')
    plt.xlabel('Iter')
    plt.ylabel('Loss')
    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()
    path = os.path.join(path, model_name + '_loss.png')
    plt.savefig(path)
    plt.close()

#---------------------------moving-Mnist---------------------
class MovingMNIST(torch.utils.data.Dataset):
    """`MovingMNIST <http://www.cs.toronto.edu/~nitish/unsupervised_video/>`_ Dataset.
    Args:root 数据存放路径/ train 是训练集还是数据集 / split 测试集数量 /dataload 是否加载下载数据(第一次) /transform /target_transform 数据格式转换 
    """
    training_file = '/_yucheng/dataSet/moving_mnist/data/moving_mnist_train_oneFrame'
    test_file = 'moving_mnist_test'
    def __init__(self, root='./data', train=True, split=1000, transform=None, target_transform=None, dataload=False):
        self.root = os.path.expanduser(root)#创建路径root
        self.transform = transform
        self.target_transform = target_transform
        self.split = split
        self.train = train  # training set or test set
#第一次执行需要
        if dataload == True:
            self.dataload()
        if self.train:
            self.train_data = torch.load(os.path.join(self.root, self.training_file))
        else:
            self.test_data = torch.load(os.path.join(self.root, self.test_file))
    def __getitem__(self, index):
        if self.train:
            seq= self.train_data[index]
            seq,label = self.transform(seq),index
        else:
            seq= self.test_data[index]
            seq,label = self.transform(seq),index
        return seq,label
    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)
#给出文件路径，用numpy读取，torch序列化存入磁盘
    def dataload(self):
        print('Processing...')
        training_set = torch.from_numpy(np.load(os.path.join('mnist_test_seq.npy')).swapaxes(0, 1)[:-self.split])#np原型为[20,10000,64,64],tensor为[-1,20,64,64]
        #print(training_set.shape) #[9000,20,64,64]
        training_set = training_set.reshape(180000,64,64)
        training_set = training_set.unsqueeze(1)#[180000,1,64,64]
        training_set = training_set/1.0
        test_set = torch.from_numpy(np.load(os.path.join('mnist_test_seq.npy')).swapaxes(0, 1)[-self.split:])
        test_set = test_set.reshape(20000,64,64)
        test_set = test_set.unsqueeze(1)#[20000,1,64,64]
        test_set = test_set/1.0
        with open(os.path.join(self.root,self.training_file), 'wb') as f: #第一个参数是存入的路径
            torch.save(training_set, f)
        with open(os.path.join(self.root,self.test_file), 'wb') as f:
            torch.save(test_set, f)
        print('Done!')
    def __repr__(self):
    #输入对象名时反馈的信息
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Train/test: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str



