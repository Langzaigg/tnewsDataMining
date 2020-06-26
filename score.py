from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from tqdm import tqdm
import json

label2id = {'100': 0, '101': 1, '102': 2, '103': 3, '104': 4, '106': 5, '107': 6, '108': 7, '109': 8,
            '110': 9, '112': 10, '113': 11, '114': 12, '115': 13, '116': 14}
label2desc = {'100': "news_story", '101': "news_culture", '102': "news_entertainment", '103': "news_sports",
              '104': "news_finance", '106': "news_house", '107': "news_car", '108': "news_edu", '109':  "news_tech",
              '110': "news_military", '112': "news_travel", '113': "news_world", '114': "news_stock",
              '115': "news_agriculture", '116': "news_game"}


def load_data(path):
    label = []
    label_desc = []
    sentence = []
    keyword = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            data = eval(line)
            label.append(label2id[data['label']])
            label_desc.append(data['label_desc'])
            sentence.append(data['sentence'])
    return label, label_desc, sentence


labels, _, _ = load_data('tnews_public/test.json')
predicts, descs, sentence = load_data('tnews_predict.json')

print(classification_report(labels, predicts, target_names=label2desc.values()))
print(f1_score(labels, predicts, average='weighted'))
