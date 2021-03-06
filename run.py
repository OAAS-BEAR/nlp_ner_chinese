import pickle
import logging
import argparse
import os
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from model import CWS
from dataloader import Sentence

def get_param():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_dim', type=int, default=768)
    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--max_epoch', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=30)
    parser.add_argument('--hidden_dim', type=int, default=200)
    parser.add_argument('--cuda', action='store_true', default=True)
    return parser.parse_args()


def set_logger():
    log_file = os.path.join('save', 'log.txt')
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.DEBUG,
        datefmt='%Y-%m%d %H:%M:%S',
        filename=log_file,
        filemode='w',
    )

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def entity_split(x, y, id2tag, entities, cur):
    start, end = -1, -1
    for j in range(len(x)):
        if id2tag[y[j]] == 'B':
            start = cur + j
        elif id2tag[y[j]] == 'M' and start != -1:
            continue
        elif id2tag[y[j]] == 'E' and start != -1:
            end = cur + j
            entities.add((start, end))
            start, end = -1, -1
        elif id2tag[y[j]] == 'S':
            entities.add((cur + j, cur + j))
            start, end = -1, -1
        else:
            start, end = -1, -1


def main(args):
    use_cuda = args.cuda and torch.cuda.is_available()
    print("cuda is ready")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open('data/datasave.pkl', 'rb') as inp:
        word2id = pickle.load(inp)
        id2word = pickle.load(inp)
        tag2id = pickle.load(inp)
        id2tag = pickle.load(inp)
        x_train = pickle.load(inp)
        y_train = pickle.load(inp)
        x_test = pickle.load(inp)
        y_test = pickle.load(inp)

    model = CWS(len(word2id), tag2id, args.embedding_dim, args.hidden_dim)
    if use_cuda:
        model = model.cuda()
    for name, param in model.named_parameters():
        logging.debug('%s: %s, require_grad=%s' % (name, str(param.shape), str(param.requires_grad)))

    optimizer = Adam(model.parameters(), lr=args.lr)

    train_data = DataLoader(
        dataset=Sentence(x_train, y_train),
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=Sentence.collate_fn,
        drop_last=False,
        num_workers=6
    )

    test_data = DataLoader(
        dataset=Sentence(x_test[:1000], y_test[:1000]),
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=Sentence.collate_fn,
        drop_last=False,
        num_workers=6
    )

    for epoch in range(args.max_epoch):
        step = 0
        log = []
        for sentence, label, mask, length,bert_mask,segments in train_data:
            if use_cuda:
                sentence = sentence.to(device)
                label = label.to(device)
                mask = mask.to(device)
                bert_mask=bert_mask.to(device)
                segments=segments.to(device)

             
    #        print(sentence.size()) 
     #       print(label.size()) 
            # forward
            loss,emissions = model(sentence, label, mask, length,bert_mask,segments)
            
            
            log.append(loss.item())

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            if step % 100 == 0:
                print(sentence[0])
                print(label[0])
                print(mask[0])
               # print(emissions[0])
                print(bert_mask[0])
                print(segments[0])
                logging.debug('epoch %d-step %d loss: %f' % (epoch, step, sum(log)/len(log)))
                log = []

        # test
        entity_predict = set()
        entity_label = set()
        with torch.no_grad():
            model.eval()
            cur = 0
            for sentence, label, mask, length,b,s in test_data:
                if use_cuda:
                    sentence = sentence.cuda()
                    label = label.cuda()
                    mask = mask.cuda()
                    
                    b=b.cuda()
                    s=s.cuda()
                predict = model.infer(sentence, mask, length,b,s)

                for i in range(len(length)):
                    entity_split(sentence[i, :length[i]-1], predict[i], id2tag, entity_predict, cur)
                    entity_split(sentence[i, :length[i]-1], label[i, :length[i]-1], id2tag, entity_label, cur)
                    cur += length[i]

            right_predict = [i for i in entity_predict if i in entity_label]
            if len(right_predict) != 0:
                precision = float(len(right_predict)) / len(entity_predict)
                recall = float(len(right_predict)) / len(entity_label)
                logging.info("precision: %f" % precision)
                logging.info("recall: %f" % recall)
                logging.info("fscore: %f" % ((2 * precision * recall) / (precision + recall)))
            else:
                logging.info("precision: 0")
                logging.info("recall: 0")
                logging.info("fscore: 0")
            model.train()

        path_name = "./save/model_epoch" + str(epoch) + ".pkl"
        torch.save(model, path_name)
        logging.info("model has been saved in  %s" % path_name)


if __name__ == '__main__':
    set_logger()
    main(get_param())
