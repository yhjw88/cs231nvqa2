import cPickle
import json
import os


def examineQuestions(filename):
    questionJSON = json.load(open(filename))
    questions = questionJSON["questions"]

    numFound = 0
    imageDict = {}
    for question in questions:
        image_id = str(question["image_id"])
        image_id = ("0" * (12 - len(image_id))) + image_id
        image_path = "data/val2014img/COCO_val2014_" + image_id + ".jpg"
        if os.path.isfile(image_path):
            numFound += 1
        imageDict[int(image_id)] = image_path
    print "{} out of {} found".format(numFound, len(questions))
    print len(imageDict)
    with open("data/valSample2014imgs.json", "w") as outfile:
        json.dump(imageDict, outfile)

if __name__ == "__main__":
    # filename = "data/v2_OpenEnded_mscoco_val2014_questions.json"
    # filename = "data/v2_OpenEnded_mscoco_valSample2014_questions.json"
    # examineQuestions(filename)

    # originJson = json.load(open("data/v2_mscoco_val2014_annotations.json"))
    # allQs = originJson["annotations"]

    # questionIds = set()
    # truncatedQs = json.load(open("data/vqa_val_final_sample.json"))
    # for q in truncatedQs:
    #     questionIds.add(int(q["question_id"]))

    # remainingQs = []
    # for q in allQs:
    #     questionId = int(q["question_id"])
    #     if questionId in questionIds:
    #         remainingQs.append(q)
    # print len(remainingQs)
    # originJson["annotations"] = remainingQs
    # with open("data/v2_mscoco_valSample2014_annotations.json", 'w') as outFile:
    #     json.dump(originJson, outFile)

    filename = "data/resnet/train49Idx.pkl"
    a = cPickle.load(open(filename))
    b = {}
    for key, value in a.iteritems():
        b[int(key.numpy())] = value
    cPickle.dump(b, open("data/resnet/train49Idx2.pkl", "wb"))
