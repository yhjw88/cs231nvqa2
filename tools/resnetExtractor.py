import argparse
import cPickle
import h5py
import os
import os.path
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm

class ImageDataset(Dataset):

    def __init__(self, folder, split):
        self.images = self.loadImages(folder, split)
        self.transform = self.getTransform(448)

    def __getitem__(self, index):
        image = self.images[index]
        imageF = Image.open(image["path"]).convert("RGB")
        if self.transform is not None:
            imageF = self.transform(imageF)
        return image["imageId"], imageF

    def __len__(self):
        return len(self.images)

    def loadImages(self, folder, split):
        images = []
        for filename in os.listdir(folder):
            if filename[-4:] == ".jpg":
                filepath = os.path.join(folder, filename)
                prefixLength = len("COCO_"+split+"2014_")
                imageId = filename[prefixLength:-4]
                assert len(imageId) == 12
                images.append({"imageId": int(imageId), "path": filepath})
        return images

    def getTransform(self, size):
        transform = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        return transform

class ResnetStripped(nn.Module):
    def __init__(self):
        super(ResnetStripped, self).__init__()
        resnet = models.resnet152(pretrained=True)
        self.layer = nn.Sequential(*list(resnet.children())[:-2])
        self.pool  = nn.AvgPool2d(2, stride=2)

    def forward(self, x):
        x = self.layer(x)
        return self.pool(x)

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    args = parser.parse_args()
    return args

def extractFeatures(args, inFolder, outFilename, outIdxFilename, split):
    print "Loading data"
    imageDataset = ImageDataset(inFolder, split)
    imageLoader = DataLoader(imageDataset, args.batch_size, shuffle=False)
    print "Loaded {} images".format(len(imageDataset))

    resnet = ResnetStripped()
    resnet = nn.DataParallel(resnet).cuda()
    resnet.train(False)

    outFile = h5py.File(outFilename, "w")
    outFeatures = outFile.create_dataset(
        "image_features",
        (len(imageDataset), 49, 2048),
        "f")
    indices = {}
    counter = 0

    with torch.no_grad():
        for imageIds, images in tqdm(imageLoader):
            numImages = len(imageIds)
            imageFs = resnet(images.cuda()) # Output (-1, 2048, 7, 7)
            imageFs = imageFs.cpu().numpy()
            imageFs = imageFs.transpose([0, 2, 3, 1])
            imageFs = imageFs.reshape((numImages, 49, 2048))

            outFeatures[counter:counter+numImages, :, :] = imageFs
            for imageId in imageIds:
                indices[int(imageId.numpy())] = counter
                counter += 1

    print "Dumping indices"
    cPickle.dump(indices, open(outIdxFilename, 'wb'))
    outFile.close()

if __name__ == "__main__":
    args = parseArgs()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    extractFeatures(
        args,
        "data/train2014img",
        "data/resnet/train49.hdf5",
        "data/resnet/train49Idx.pkl",
        "train")

    # resnet = models.resnet152(pretrained=True)
    # for child in resnet.children():
    #     print child
    # print resnet.layer4[-1].out_features
    # for name, child in resnet.named_children():
    #     print name

