import os
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

import base_model

class ResnetStripped(nn.Module):
    def __init__(self):
        super(ResnetStripped, self).__init__()
        resnet = models.resnet152(pretrained=True)
        self.layer = nn.Sequential(*list(resnet.children())[:-2])
        self.pool  = nn.AvgPool2d(2, stride=2)

    def forward(self, x):
        x = self.layer(x)
        return self.pool(x)

class CombinedModel(nn.Module):
    def __init__(self, imageModel, attentionModel):
        super(CombinedModel, self).__init__()
        self.imageModel = imageModel
        self.attentionModel = attentionModel

    def forward(self, images, b, q, labels):
        v = self.imageModel(images)
        v = v.permute([0, 2, 3, 1]).view(-1, 49, 2048)
        logits = self.attentionModel(v, b, q, labels)
        return logits

class ImageLoader():
    def __init__(self, folder, split):
        self.imageDict = self.loadImages(folder, split)
        self.transform = self.getTransform(448)

    def getImage(self, imageId):
        imageEntry = self.imageDict[imageId]
        if "image" not in imageEntry:
            imageEntry["image"] = Image.open(imageEntry["path"]).convert("RGB")
            imageEntry["image"] = self.transform(imageEntry["image"])
        return imageEntry["image"]

    def loadImages(self, folder, split):
        imageDict = {}
        for filename in os.listdir(folder):
            if filename[-4:] == ".jpg":
                filepath = os.path.join(folder, filename)
                prefixLength = len("COCO_"+split+"2014_")
                imageId = filename[prefixLength:-4]
                assert len(imageId) == 12
                imageDict[int(imageId)] = {"path": filepath}
        return imageDict

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

def getCombinedModel(args, dataset):
    constructor = 'build_%s' % args.model
    attentionModel = getattr(base_model, constructor)(dataset, args.num_hid).cuda()
    attentionModel.w_emb.init_embedding('data/glove6b_init_300d.npy')
    attentionModel = nn.DataParallel(attentionModel).cuda()
    if args.load_path:
        load_path = os.path.join(args.load_path, 'model.pth')
        print "Loading model from {}".format(load_path)
        attentionModel.load_state_dict(torch.load(load_path))

    combinedModel = CombinedModel(ResnetStripped(), attentionModel)
    return combinedModel
