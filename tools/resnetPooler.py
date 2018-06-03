import h5py
import torch

def poolFeatures(args, inFilename, outFilename):
    with h5py.File(inFilename, "r") as inFile, \
        h5py.File(outFilename, "w") as outFile:

        inFeatures = inFile["image_features"]
        outFeatures = outFile.create_dataset(
            "image_features",
            (len(inFeatures), 9, 2048),
            "f")

        pool = torch.nn.AvgPool2d(3, stride=2)
        for i, inFeature in enumerate(inFeatures):
            outFeature = pool(torch.from_numpy(inFeature))
            outFeatures[i, :, :] = outFeature.numpy()

            if (i+1)%500 == 0:
                print "Processed {}".format(i+1)
    print "Done"

if __name__ == "__main__":
    poolFeatures(
        "data/resnet/train49.hdf5",
        "data/resnet/train9.hdf5")
