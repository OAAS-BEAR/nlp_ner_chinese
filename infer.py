import torch
import pickle
from transformers import BertTokenizer, BertModel
if __name__ == '__main__':
    model = torch.load('save/model.pkl', map_location=torch.device('cpu'))
    output = open('cws_result.txt', 'w', encoding='utf-8')
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    with open('data/datasave.pkl', 'rb') as inp:
        word2id = pickle.load(inp)
        id2word = pickle.load(inp)
        tag2id = pickle.load(inp)
        id2tag = pickle.load(inp)
        x_train = pickle.load(inp)
        y_train = pickle.load(inp)
        x_test = pickle.load(inp)
        y_test = pickle.load(inp)

    with open('data/test_x_ner.txt', 'r', encoding='utf-8') as f:
        for test in f:
            flag = False
            test = test.strip()

            line_x = []
            line_x.append("[CLS]")

            for i in range(len(test)):
                if test[i] == " ":
                    continue
                line_x.append(test[i])
            line_x.append("[SEP]")
            encoded_input = tokenizer.convert_tokens_to_ids(line_x)
            x = torch.tensor([encoded_input])
            #     mask = torch.ones_like(x, dtype=torch.uint8)
            #    length = [len(test)]

            # encoded_input = tokenizer(line,truncation=True)
            # x=torch.LongTensor(encoded_input["input_ids"])
            l = len(x[0])
            length = []
            length.append(l)
            x = x.cuda()

            bert_mask = torch.ones_like(x, dtype=torch.long).cuda()
            segments = torch.ones_like(x, dtype=torch.long).cuda()
            mask = torch.ones_like(x, dtype=torch.uint8).cuda()
            mask[0][l - 1] = 0
            predict = model.infer(x, mask, length, bert_mask, segments)[0]
            for i in range(len(test)):
                print(test[i], end=' ', file=output)
                print(id2tag[predict[i+1]],end=' ',file=output)
                print(file=output)