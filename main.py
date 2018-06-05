import argparse
import cPickle
import numpy as np
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import base_model
from dataset import Dictionary, VQAFeatureDataset
import imageModel
import train


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--model', type=str, default='baseline0_newatt')
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--output', type=str, default='saved_models/exp0')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--evalset_name',type=str,default='val')
    args = parser.parse_args()
    return args

def trainNormal(args):
    # Fetch data.
    dictionary = Dictionary.load_from_file('data/dictionary.pkl')
    print "Fetching train data"
    train_dset = VQAFeatureDataset('train', 'train', dictionary)
    print "Fetching eval data"
    eval_dset = VQAFeatureDataset('valSample', args.evalset_name, dictionary)

    # Fetch model.
    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(train_dset, args.num_hid).cuda()
    model.w_emb.init_embedding('data/glove6b_init_300d.npy')
    model = nn.DataParallel(model).cuda()
    if args.load_path:
        load_path = os.path.join(args.load_path, 'model.pth')
        print "Loading model from {}".format(load_path)
        model.load_state_dict(torch.load(load_path))

    # Train.
    train_loader = DataLoader(train_dset, args.batch_size, shuffle=True)
    eval_loader =  DataLoader(eval_dset, args.batch_size, shuffle=True)
    train.train(model, train_loader, eval_loader, args.epochs, args.output)

def evalNormal(args):
    # Fetch data.
    dictionary = Dictionary.load_from_file('data/dictionary.pkl')
    print "Fetching eval data"
    eval_dset = VQAFeatureDataset('val', args.evalset_name, dictionary)

    # Fetch model.
    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(eval_dset, args.num_hid).cuda()
    model.w_emb.init_embedding('data/glove6b_init_300d.npy')
    model = nn.DataParallel(model).cuda()
    if args.load_path:
        load_path = os.path.join(args.load_path, 'model.pth')
        print "Loading model from {}".format(load_path)
        model.load_state_dict(torch.load(load_path))

    # Evaluate
    eval_loader =  DataLoader(eval_dset, args.batch_size, shuffle=True)
    print "Evaluating..."
    model.train(False)
    eval_score, bound = train.evaluate(model, eval_loader)
    print "eval score: %.2f (%.2f)" % (100 * eval_score, 100 * bound)

def evalFromImages(args):
    # Fetch data.
    dictionary = Dictionary.load_from_file('data/dictionary.pkl')
    print "Fetching eval data"
    imageLoader = imageModel.ImageLoader("data/val2014img", "val")
    eval_dset = VQAFeatureDataset('valSample', args.evalset_name, dictionary, imageLoader=imageLoader)

    # Fetch model.
    model = imageModel.getCombinedModel(args, eval_dset)
    model = nn.DataParallel(model).cuda()

    # Evaluate
    eval_loader =  DataLoader(eval_dset, args.batch_size, shuffle=True)
    print "Evaluating..."
    model.train(False)
    eval_score, bound = train.evaluate(model, eval_loader)
    print "eval score: %.2f (%.2f)" % (100 * eval_score, 100 * bound)

def imageAdv(args, imageAdvF):
    # Fetch data.
    questionIds = cPickle.load(open("data/goodQuestions.pkl", "rb"))
    dictionary = Dictionary.load_from_file('data/dictionary.pkl')
    print "Fetching eval data"
    imageLoader = imageModel.ImageLoader("data/val2014img", "val")
    dataset = VQAFeatureDataset(
        'valSample',
        args.evalset_name,
        dictionary,
        imageLoader=imageLoader,
        questionIds=questionIds)

    # Fetch model.
    model = imageModel.getCombinedModel(args, dataset)
    model = nn.DataParallel(model).cuda()

    # Train and save.
    label2ans = dataset.label2ans
    # imageSaverOld = imageModel.ImageSaver("data/adv1Old")
    imageSaverNew = imageModel.ImageSaver("data/adv2New")
    numSuccess = 0
    successList = []
    iterList = []
    targetList = []
    predictedList = []
    for i, vqaInfo in enumerate(dataset):
        entry = dataset.entries[i]
        target = None
        while target is None or target in entry["answer"]["labels"]:
            target = np.random.randint(len(label2ans))
        print ""
        print "questionId: {}, imgId: {}".format(entry["question_id"], entry["image_id"])
        print "Answers: {}".format([label2ans[label] for label in entry["answer"]["labels"]])
        print "Target: {}".format(label2ans[target])

        startTime = time.time()
        success, imgOld, imgNew, predicted, iters = imageAdvF(model, vqaInfo, entry, target, dataset)
        print "Num: {0}, Success: {1}, Predicted: {2}, Iters: {3}, Taken: {4:.2f}".format(
            i,
            success,
            label2ans[predicted],
            iters,
            time.time() - startTime)
        
        if success:
            numSuccess += 1
            successList.append(True)
        else:
            successList.append(False)
        iterList.append(iters)
        targetList.append(target)
        predictedList.append(int(predicted.numpy()))

        qidStr = str(entry["question_id"])
        # imageSaverOld.saveImage(imgOld, prefix + imgIdStr + ".jpg")
        imageSaverNew.saveImage(imgNew, qidStr + ".jpg")

    print ""
    print "Success: {}".format(numSuccess)
    print "SuccessList: {}".format(successList)
    print "IterList: {}".format(iterList)
    print "TargetList: {}".format(targetList)
    print "PredictedList: {}".format(predictedList)

    np.savez("data/adv2Out.npz",
             successList=successList,
             iterList=iterList,
             targetList=targetList,
             predictedList=predictedList)

if __name__ == '__main__':
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    if args.mode == "train":
        trainNormal(args)
    elif args.mode == "eval":
        evalNormal(args)
    elif args.mode == "evalFromImages":
        evalFromImages(args)
    elif args.mode == "imageAdv1":
        imageAdv(args, train.imageAdv1)
    elif args.mode == "imageAdv2":
        imageAdv(args, train.imageAdv2)
    else:
        print "Mode not supported: {}".format(args.mode)
