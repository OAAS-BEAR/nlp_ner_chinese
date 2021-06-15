import torch
import torch.nn as nn
from torchcrf import CRF
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertTokenizer, BertModel

class CWS(nn.Module):

    def __init__(self, vocab_size, tag2id, embedding_dim, hidden_dim):
        super(CWS, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.dropout = nn.Dropout(0.1)
        self.tag2id = tag2id
        self.tagset_size = len(tag2id)
        self.model=BertModel.from_pretrained('bert-base-chinese',
                                  output_hidden_states = True, # Whether the model returns all hidden-states.
                                  )
        self.word_embeds = nn.Embedding(vocab_size + 1, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1,
                            bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        self.loss=nn.CrossEntropyLoss()
        self.crf = CRF(21, batch_first=True)

    def init_hidden(self, batch_size, device):
        return (torch.randn(2, batch_size, self.hidden_dim // 2, device=device),
                torch.randn(2, batch_size, self.hidden_dim // 2, device=device))

    def _get_lstm_features(self, sentence, length,bert_mask,segments):
        batch_size, seq_len = sentence.size(0), sentence.size(1)
         # idx->embedding
        #embeds = self.word_embeds(sentence.view(-1)).reshape(batch_size, seq_len, -1)
        outputs=self.model(sentence,segments,bert_mask)
        hidden_s=outputs[2]
        token_embeddings = torch.stack(hidden_s, dim=0)
      #  print(len(hidden_s))
        embeds=torch.sum(token_embeddings[-4:],dim=0)
        #embeds=outputs[0]
        #embeds=self.dropout(embeds)
       # print(embeds.size())
     #   print(embeds.size())
        #embeds=self.dropout(embeds)
        embeds = pack_padded_sequence(embeds, length, batch_first=True)
        
        # LSTM forward
        self.hidden = self.init_hidden(batch_size, sentence.device)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        lstm_feats = self.hidden2tag(lstm_out)
        
        return lstm_feats

    def forward(self, sentence, tags, mask, length,bert_mask,segments):
        emissions = self._get_lstm_features(sentence, length,bert_mask,segments)
        loss = -self.crf(emissions, tags, mask, reduction='mean')
        s=[]
        return loss,s

    def infer(self, sentence, mask, length,b,s):
        emissions = self._get_lstm_features(sentence, length,b,s)
        return self.crf.decode(emissions, mask)
