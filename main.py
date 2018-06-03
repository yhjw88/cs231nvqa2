import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import Dictionary, VQAFeatureDataset
import base_model
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
    eval_dset = VQAFeatureDataset('val', args.evalset_name, dictionary)

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

if __name__ == '__main__':
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    if args.mode == "train":
        trainNormal(args)
    elif args.mode == "eval":
        evalNormal(args)
    else:
        print "Mode not supported: {}".format(args.mode)
