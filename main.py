import argparse
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
    parser.add_argument('--eval_only', type=str, default=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    dictionary = Dictionary.load_from_file('data/dictionary.pkl')
    if not args.eval_only:
        train_dset = VQAFeatureDataset('train', dictionary)
    eval_dset = VQAFeatureDataset('val', dictionary)
    batch_size = args.batch_size

    # Fetch model.
    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(train_dset, args.num_hid).cuda()
    model.w_emb.init_embedding('data/glove6b_init_300d.npy')
    model = nn.DataParallel(model).cuda()
    if args.load_path:
        print "Loading model from {}".format(args.load_path)
        checkpoint = torch.load(args.load_path)
        model.load_state_dict(checkpoint["state_dict"])

    # Train/eval.
    eval_loader =  DataLoader(eval_dset, batch_size, shuffle=True)
    if not args.eval_only:
        train_loader = DataLoader(train_dset, batch_size, shuffle=True)
        train.train(model, train_loader, eval_loader, args.epochs, args.output)
    else:
        model.train(False)
        eval_score, bound = train.evaluate(model, eval_loader)
        print "eval score: %.2f (%.2f)" % (100 * eval_score, 100 * bound)
