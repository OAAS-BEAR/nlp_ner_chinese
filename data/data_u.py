from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
import pickle

INPUT_DATA = "RMRB_NER_CORPUS.txt"
SAVE_PATH = "./datasave.pkl"
TEST_PATH="./test_ner.txt"
id2tag = []# B：分词头部 M：分词词中 E：分词词尾 S：独立成词
tag2id = {}
word2id = {}
id2word = []


def getList(input_str):
    '''
    单个分词转换为tag序列
    :param input_str: 单个分词
    :return: tag序列
    '''
    outpout_str = []
    if len(input_str) == 1:
        outpout_str.append(tag2id['S'])
    elif len(input_str) == 2:
        outpout_str = [tag2id['B'], tag2id['E']]
    else:
        M_num = len(input_str) - 2
        M_list = [tag2id['M']] * M_num
        outpout_str.append(tag2id['B'])
        outpout_str.extend(M_list)
        outpout_str.append(tag2id['E'])
    return outpout_str


def handle_data():
    '''
    处理数据，并保存至savepath
    :return:
    '''
    x_data = []
    y_data = []
    wordnum = 0
    line_num = 0
    label_num=0
    with open(INPUT_DATA, 'r', encoding="utf-8") as ifp:
        sentence_x=[]

        sentence_y=[]
        sentence_x.append("[CLS]")
        sentence_y.append(0)
        for line in ifp:
            if line =='\n':
                sentence_x.append("[SEP]")
                sentence_y.append(0)
                encoded_input = tokenizer.convert_tokens_to_ids(sentence_x)
                x_data.append(encoded_input)
                y_data.append(sentence_y)
                sentence_x=[]

                sentence_y=[]
                sentence_x.append("[CLS]")
                sentence_y.append(0)
            else:
                line=line.split()
                if line[1] in id2tag:
                    sentence_y.append(tag2id[line[1]])
                else:
                    id2tag.append(line[1])
                    tag2id[line[1]]=label_num
                    sentence_y.append(label_num)
                    label_num+=1
                sentence_x.append(line[0])




    print(x_data[0])
#    print([id2word[i] for i in x_data[0]])
    print(y_data[0])

    print(id2tag)


    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1, random_state=43)
    output = open('test_x_ner.txt', 'w', encoding='utf-8')
    for line in x_test:
        sentence=tokenizer.convert_tokens_to_ids(line)
        sentence=sentence[1:len(sentence)-1]
        for i in range(len(sentence)):
            print(sentence[i], end=' ', file=output)
        print(file=output)



    x_t1, x_t2, y_t1, y_t2=train_test_split(x_train,y_train,test_size=0.1,random_state=43)
    with open(SAVE_PATH, 'wb') as outp:
        pickle.dump(word2id, outp)
        pickle.dump(id2word, outp)
        pickle.dump(tag2id, outp)
        pickle.dump(id2tag, outp)
        pickle.dump(x_t1, outp)
        pickle.dump(y_t1, outp)
        pickle.dump(x_t2, outp)
        pickle.dump(y_t2, outp)


if __name__ == "__main__":
    handle_data()
