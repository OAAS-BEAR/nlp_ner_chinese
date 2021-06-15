import torch
import pickle
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
class Sentence(Dataset):
    def __init__(self, x, y, batch_size=10):
        self.x = x
        self.y = y
        self.batch_size = batch_size
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if len(self.x[idx]) != len(self.y[idx]):
            print("x size")
            print(len(self.x[idx]))
            print("  y size")
            print(len(self.y[idx]))
        assert len(self.x[idx]) == len(self.y[idx])
        return self.x[idx], self.y[idx]

    @staticmethod
    def collate_fn(train_data):
        train_data.sort(key=lambda data: len(data[0]), reverse=True)
        data_length = [len(data[0]) for data in train_data]
        data_x = [torch.LongTensor(data[0]) for data in train_data]
        data_y = [torch.LongTensor(data[1]) for data in train_data]
        
        bert_mask=[torch.ones(l,dtype=torch.long) for l in data_length]
        segments=[torch.ones(l,dtype=torch.long) for l in data_length]
        mask = [torch.ones(l, dtype=torch.uint8) for l in data_length]
        
        data_x = pad_sequence(data_x, batch_first=True, padding_value=0)
        data_y = pad_sequence(data_y, batch_first=True, padding_value=0)
        mask = pad_sequence(mask, batch_first=True, padding_value=0)
        
        bert_mask=pad_sequence(bert_mask, batch_first=True, padding_value=0)
        segments=pad_sequence(segments, batch_first=True, padding_value=1)
        for i in range(mask.size(0)):
            mask[i][data_length[i]-1]=0
        return data_x.long(), data_y.long(), mask, data_length,bert_mask,segments


if __name__ == '__main__':
    # test
    with open('../data/datasave.pkl', 'rb') as inp:
        word2id = pickle.load(inp)
        id2word = pickle.load(inp)
        tag2id = pickle.load(inp)
        id2tag = pickle.load(inp)
        x_train = pickle.load(inp)
        y_train = pickle.load(inp)
        x_test = pickle.load(inp)
        y_test = pickle.load(inp)

    train_dataloader = DataLoader(Sentence(x_train, y_train), batch_size=10, shuffle=True, collate_fn=Sentence.collate_fn)

    for input, label, mask, length in train_dataloader:
        print(input, label)
        break
