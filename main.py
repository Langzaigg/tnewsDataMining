from keras.layers import *
from keras.models import Model
from keras.callbacks import *
from keras.optimizers import Adam
from keras.utils.np_utils import *
from bert4keras.models import build_transformer_model
from keras_bert import Tokenizer
from tqdm import tqdm
import numpy as np
import jieba
import json
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

label2id = {'100': 0, '101': 1, '102': 2, '103': 3, '104': 4, '106': 5, '107': 6, '108': 7, '109': 8,
            '110': 9, '112': 10, '113': 11, '114': 12, '115': 13, '116': 14}
labels = ['100', '101', '102', '103', '104', '106', '107',
          '108', '109', '110', '112', '113', '114', '115', '116']
label2desc = {'100': "news_story", '101': "news_culture", '102': "news_entertainment", '103': "news_sports",
              '104': "news_finance", '106': "news_house", '107': "news_car", '108': "news_edu", '109':  "news_tech",
              '110': "news_military", '112': "news_travel", '113': "news_world", '114': "news_stock",
              '115': "news_agriculture", '116': "news_game"}
key_word = []
for i in range(15):
    jieba.load_userdict('keyword/{}.txt'.format(i))
for i in range(15):
    key_word.append([word.strip() for word in open(
        'keyword/{}.txt'.format(i), encoding='utf-8') if word.strip()])


def load_data(path, mode):
    if mode == 'train' or mode == 'valid':
        label = []
        label_desc = []
        sentence = []
        keyword = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                keywords = np.zeros((15,), dtype=np.int32)
                data = eval(line)
                label.append(label2id[data['label']])
                label_desc.append(data['label_desc'])
                sentence.append(data['sentence'])
                key = data['keywords'].split(',')
                if key[0]:
                    for i in range(15):
                        tmp = key_word[i]
                        for j in key:
                            if j in tmp:
                                keywords[i] = 1  # 出现某一类的关键词，在该类位置标1
                else:
                    # 原数据中若未给出关键词，使用jiaba分词
                    key = list(jieba.lcut(data['sentence']))
                    for i in range(15):
                        tmp = key_word[i]
                        for j in key:
                            if j in tmp:
                                keywords[i] = 1
                keyword.append(keywords)
        return label, label_desc, sentence, keyword
    if mode == 'test':
        id = []
        sentence = []
        keyword = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                keywords = np.zeros((15,), dtype=np.int32)
                data = eval(line)
                id.append(data['id'])
                sentence.append(data['sentence'])
                key = data['keywords'].split(',')
                if key:
                    for i in range(15):
                        tmp = key_word[i]
                        for j in key:
                            if j in tmp:
                                keywords[i] = 1
                else:
                    key = list(jieba.lcut(data['sentence']))
                    for i in range(15):
                        tmp = key_word[i]
                        for j in key:
                            if j in tmp:
                                keywords[i] = 1
                keyword.append(keywords)
        return id, sentence, keyword


train_label, train_label_desc, train_sentence, train_keywords = load_data(
    'tnews_public/train.json', 'train')
dev_label, dev_label_desc, dev_sentence, dev_keywords = load_data(
    'tnews_public/dev.json', 'valid')
test_label, test_label_desc, test_sentence, test_keywords = load_data(
    'tnews_public/test.json', 'valid')

train_label += dev_label  # 训练集测试集合并
train_label_desc += dev_label_desc
train_sentence += dev_sentence
train_keywords += dev_keywords

train_label_desc = np.array(train_label_desc)
train_sentence = np.array(train_sentence)
train_keywords = np.array(train_keywords)

'''
重要参数（可调整）
'''
MAX_LEN = 128
epoch = 3
config_path = 'chinese/bert_config.json'
checkpoint_path = 'chinese/bert_model.ckpt'
dict_path = 'chinese/vocab.txt'
batch_size = 16
learning_rate = 2e-5
token_dict = {}
with open(dict_path, 'r', encoding='utf-8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)
tokenizer = Tokenizer(token_dict)


class DataGenerator:
    """数据生成器"""

    def __init__(self, sentence, label, keywords, batch_size=batch_size):
        self.sentence = sentence
        self.label = label
        self.keywords = keywords
        self.batch_size = batch_size
        self.steps = len(self.sentence) // self.batch_size
        if len(self.sentence) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            ids = list(range(len(self.sentence)))
            np.random.shuffle(ids)
            X1, X2, X3, Y = [], [], [], []
            for i in ids:
                text, label = self.sentence[i], self.label[i]
                x1, x2 = tokenizer.encode(first=text, max_len=MAX_LEN)
                X1.append(x1)
                X2.append(x2)
                X3.append(self.keywords[i])
                Y.append(label)
                if len(X1) == self.batch_size or i == ids[-1]:
                    Y = np.array(Y)
                    X1 = np.array(X1)
                    X2 = np.array(X2)
                    X3 = np.array(X3)
                    yield [X1, X2, X3], Y
                    X1, X2, X3, Y = [], [], [], []


def get_model():
    bert_model = build_transformer_model(
        config_path, checkpoint_path)
    for l in bert_model.layers:
        l.trainable = True  # 设置bert每一层都参与训练
    x1 = Input(shape=(MAX_LEN,))  # token embedding
    x2 = Input(shape=(MAX_LEN,))  # segment embedding
    x3 = Input(shape=(15,))  # 关键词特征
    output = bert_model([x1, x2])
    output = Lambda(lambda x: x[:, 0])(output)  # bert CLS输出向量
    output = Concatenate()([output, x3])  # CLS拼接关键词特征
    output = Dense(64, activation='relu')(output)  # 全连接层
    output = Dense(15, activation='softmax')(output)
    model = Model([x1, x2, x3], output)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate), metrics=['accuracy'])
    model.summary()
    return model


def predicted(sentence, keyword):
    prediction = []
    for i in tqdm(range(len(sentence))):
        text = sentence[i]
        x1, x2 = tokenizer.encode(first=text, max_len=MAX_LEN)
        x3 = keyword[i]
        tmp = model.predict([[x1], [x2], [x3]])[0]
        prediction.append(tmp)
    return prediction


skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=223344)  # 五折交叉验证
train_dev = np.zeros((len(train_sentence), 15), dtype=np.float32)  # 初始化验证结果
test = np.zeros((len(test_sentence), 15), dtype=np.float32)  # 初始化测试集预测结果
for fold, (train_index, valid_index) in enumerate(skf.split(train_sentence, train_label)):
    tmp = to_categorical(train_label)
    sentence_train = train_sentence[train_index]
    label_train = tmp[train_index]
    keyword_train = train_keywords[train_index]

    sentence_dev = train_sentence[valid_index]
    label_dev = tmp[valid_index]
    keyword_dev = train_keywords[valid_index]

    train_D = DataGenerator(sentence_train, label_train, keyword_train)
    valid_D = DataGenerator(sentence_dev, label_dev, keyword_dev)
    model = get_model()
    # 保存对验证集效果最好的模型

    checkpoint = ModelCheckpoint('model_save/bert{}.h5'.format(fold), monitor='val_acc',
                                 verbose=2, save_best_only=True, mode='max', save_weights_only=True)
    history = model.fit_generator(
        train_D.__iter__(), steps_per_epoch=len(train_D), epochs=epoch,
        validation_data=valid_D.__iter__(),
        validation_steps=len(valid_D),
        callbacks=[checkpoint]
    )
    model.load_weights('model_save/bert{}.h5'.format(fold))
    train_dev[valid_index] = predicted(sentence_dev, keyword_dev)
    test += predicted(test_sentence, test_keywords)
test /= 5  # 5个模型测试集预测结果求平均
s = '交叉验证结果：' + str(accuracy_score(train_label, np.argmax(train_dev, axis=1)))
s += '\n' + str(classification_report(train_label,
                                      np.argmax(train_dev, axis=1)))
res_file = open('res_dev.txt', 'w', encoding='utf-8')
res_file.write(s)
res_file.close()

json_file = open('tnews_test_predict.json', 'w', encoding='utf-8')  # 写入提交文件
for i in tqdm(range(len(test_sentence))):
    text, keywords = test_sentence[i], test_keywords[i]
    tmp = np.argmax(test[i])
    test_label = labels[int(tmp)]
    test_label_desc = label2desc[test_label]
    result = {"label": test_label, "label_desc": test_label_desc,
              "sentence": text}
    json_file.write(json.dumps(result))
    json_file.write('\n')
json_file.close()

"""输入一句话，输出类别"""
text = input('请输入文本：')
x1, x2 = tokenizer.encode(first=text, max_len=MAX_LEN)
key = list(jieba.lcut(text))
keywords = np.zeros((15,), dtype=np.int32)
for i in range(15):
    tmp = key_word[i]
    for j in key:
        if j in tmp:
            keywords[i] = 1

result = np.zeros((15,), dtype=np.float32)
for i in range(5):
    model.load_weights('model_save/bert{}.h5'.format(i))
    result += model.predict([[x1], [x2], [keywords]])[0]
result = np.argmax(result)
print(label2desc[labels[int(result)]])
