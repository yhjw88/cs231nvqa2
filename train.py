import os
import time
import torch
import torch.nn as nn
import utils
from torch.autograd import Variable
import cPickle

def instance_bce_with_logits(logits, labels):
    assert logits.dim() == 2

    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss *= labels.size(1)
    return loss


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


def train(model, train_loader, eval_loader, num_epochs, output):
    utils.create_dir(output)
    optim = torch.optim.Adamax(model.parameters())
    logger = utils.Logger(os.path.join(output, 'log.txt'))
    best_eval_score = 0

    for epoch in range(num_epochs):
        total_loss = 0
        train_score = 0
        t = time.time()

        for i, (v, b, q, a) in enumerate(train_loader):
            v = Variable(v).cuda()
            b = Variable(b).cuda()
            q = Variable(q).cuda()
            a = Variable(a).cuda()

            pred = model(v, b, q, a)
            loss = instance_bce_with_logits(pred, a)
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), 0.25)
            optim.step()
            optim.zero_grad()

            batch_score = compute_score_with_logits(pred, a.data).sum()
            batch_loss = loss.data[0] * v.size(0)
            total_loss += batch_loss
            train_score += batch_score

            if i%100 == 0:
                logger.write(
                    'epoch %d, batch %d, batchLoss %.2f, batchScore %.2f' % (
                        epoch,
                        i,
                        loss.data[0],
                        100 * batch_score / v.size(0)))

        total_loss /= len(train_loader.dataset)
        train_score = 100 * train_score / len(train_loader.dataset)
        model.train(False)
        eval_score, bound = evaluate(model, eval_loader)
        model.train(True)

        logger.write('epoch %d, time: %.2f' % (epoch, time.time()-t))
        logger.write('\ttrain_loss: %.2f, score: %.2f' % (total_loss, train_score))
        logger.write('\teval score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))

        if eval_score > best_eval_score:
            model_path = os.path.join(output, 'model.pth')
            torch.save(model.state_dict(), model_path)
            best_eval_score = eval_score


def evaluate(model, dataloader):
    score = 0
    upper_bound = 0
    num_data = 0
    prediction = []
    with torch.no_grad():
        for i, (v, b, q, a) in enumerate(dataloader):
            pred = model(v.cuda(), b.cuda(), q.cuda(), None)
            batch_score = compute_score_with_logits(pred, a.cuda()).sum()
            score += batch_score
            upper_bound += (a.max(1)[0]).sum()
            num_data += pred.size(0)
            prediction.append(pred)

            if i % 2 == 0:
                print 'batch %d, batchScore %.2f' % (i, 100 * batch_score / v.size(0))

    # cPickle.dump(prediction,open("predictions.pkl","wb"))
    score = score / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)
    return score, upper_bound

# Currently works on only one image.
def imageAdv1(model, dataset):
    model.train(False)
    for param in model.parameters():
        param.requires_grad = False

    imgOld, b, q, a = dataset[0]

    print ""
    print "Printing good answers"
    label2ans = dataset.label2ans
    for i, score in enumerate(a):
        if score > 0:
            print "{}: {}".format(label2ans[i], score)
    target = 1661 #TODO: Pick target?
    print "Target is {}".format(label2ans[target])
    print ""

    imgOld = imgOld.unsqueeze(0).cuda()
    b = b.unsqueeze(0).cuda()
    q = q.unsqueeze(0).cuda()
    a = a.unsqueeze(0).cuda()
    img = imgOld.clone().cuda()

    while True:
        startTime = time.time()

        img.requires_grad = True
        logits = model(img, b, q, a)

        predicted = torch.argmax(logits)
        targetScore = logits[0][target]
        print "predicted: {0}, predictedScore: {1:.2f}, targetScore: {2:.2f}, time: {3:.2f} ".format(
            label2ans[predicted],
            logits[0][predicted],
            targetScore,
            time.time() - startTime)
        if predicted == target:
            print "Done"
            break

        targetScore -= 0.1 * torch.sum((img - imgOld)**2)
        targetScore.backward()
        with torch.no_grad():
            norm = torch.sqrt(torch.sum(img.grad**2))
            img += img.grad / norm
            img.grad.zero_()

    return imgOld.cpu().squeeze(), img.cpu().squeeze()
