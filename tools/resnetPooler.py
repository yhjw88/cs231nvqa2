import h5py
import torch

def poolFeatures(inFilename, outFilename):
    with h5py.File(inFilename, "r") as inFile, \
        h5py.File(outFilename, "w") as outFile:

        inFeatures = inFile["image_features"]
        outFeatures = outFile.create_dataset(
            "image_features",
            (len(inFeatures), 9, 2048),
            "f")

        pool = torch.nn.AvgPool2d(3, stride=2)
        for i, inFeature in enumerate(inFeatures):
            inFeature = inFeature.transpose([1, 0]).reshape((2048, 7, 7))
            outFeature = pool(torch.from_numpy(inFeature))
            outFeatures[i, :, :] = outFeature.numpy().reshape((2048, 9)).transpose([1, 0])

            if (i+1)%500 == 0:
                print "Processed {}".format(i+1)
    print "Done"

if __name__ == "__main__":
    poolFeatures(
        "data/resnet/train49.hdf5",
        "data/resnet/train9.hdf5")
