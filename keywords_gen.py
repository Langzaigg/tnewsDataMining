'''
提取每一类的关键词
根据训练集中给出的关键词，统计词频，选出词频前100的词
'''
import collections
train_keywords100 = []
train_keywords101 = []
train_keywords102 = []
train_keywords103 = []
train_keywords104 = []
train_keywords106 = []
train_keywords107 = []
train_keywords108 = []
train_keywords109 = []
train_keywords110 = []
train_keywords112 = []
train_keywords113 = []
train_keywords114 = []
train_keywords115 = []
train_keywords116 = []

with open('tnews_public/train.json', 'r', encoding='utf-8') as f:
    for line in f:
        data = eval(line)
        if data['keywords'] and data['label'] == '100':
            train_keywords100 += data['keywords'].split(',')
        if data['keywords'] and data['label'] == '101':
            train_keywords101 += data['keywords'].split(',')
        if data['keywords'] and data['label'] == '102':
            train_keywords102 += data['keywords'].split(',')
        if data['keywords'] and data['label'] == '103':
            train_keywords103 += data['keywords'].split(',')
        if data['keywords'] and data['label'] == '104':
            train_keywords104 += data['keywords'].split(',')
        if data['keywords'] and data['label'] == '106':
            train_keywords106 += data['keywords'].split(',')
        if data['keywords'] and data['label'] == '107':
            train_keywords107 += data['keywords'].split(',')
        if data['keywords'] and data['label'] == '108':
            train_keywords108 += data['keywords'].split(',')
        if data['keywords'] and data['label'] == '109':
            train_keywords109 += data['keywords'].split(',')
        if data['keywords'] and data['label'] == '110':
            train_keywords110 += data['keywords'].split(',')
        if data['keywords'] and data['label'] == '112':
            train_keywords112 += data['keywords'].split(',')
        if data['keywords'] and data['label'] == '113':
            train_keywords113 += data['keywords'].split(',')
        if data['keywords'] and data['label'] == '114':
            train_keywords114 += data['keywords'].split(',')
        if data['keywords'] and data['label'] == '115':
            train_keywords115 += data['keywords'].split(',')
        if data['keywords'] and data['label'] == '116':
            train_keywords116 += data['keywords'].split(',')
keywords = [train_keywords100, train_keywords101, train_keywords102, train_keywords103, train_keywords104,
            train_keywords106, train_keywords107, train_keywords108, train_keywords109, train_keywords110,
            train_keywords112, train_keywords113, train_keywords114, train_keywords115, train_keywords116]
count = 0
for key in keywords:
    f = open('keyword/{}.txt'.format(count), 'w', encoding='utf-8')
    top200 = collections.Counter(key).most_common(200)
    for word, i in top200:
        f.write(word)
        f.write('\n')
    f.close()
    count += 1
