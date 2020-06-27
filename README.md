# fnewsDataMining
上海大学数据挖掘课程大作业

## 项目需求

```
jieba==0.42.1
Keras==2.2.4
keras-bert==0.81.0
numpy==1.18.1
scikit-learn==0.22.1
tensorflow-gpu==1.15.2
tqdm==4.42.1
bert4keras==0.8.1
```

## 模型下载

预训练模型：[chinese_L-12_H-768_A-12](https://drive.google.com/drive/folders/1tAIO4vcUHAxwf0KMSz73NLycjzVc7Q6v?usp=sharing)

训练结果(epoch = 3)：[model_save](https://drive.google.com/drive/folders/1l09n4B-7r0Qpe5lGEkF6q0XGZZfP93rV?usp=sharing)

## 结果

五折交叉训练结果：

f1 Score(weighted)：0.6599897189856065

||precision|recall|f1-score|support|
|:----|:----:|:----:|:----:|:---:|
|news_story|0.66|0.56|0.61|1210
|news_culture|0.67|0.66|0.67|4450
|news_entertainment|0.67|0.72|0.70|5443
|news_sports|0.84|0.78|0.81|4365
|news_finance|0.59|0.53|0.56|5660
|news_house|0.71|0.74|0.72|2302
|news_car|0.75|0.74|0.74|4514
|news_edu|0.70|0.65|0.68|3774
|news_tech|0.60|0.67|0.63|6475
|news_military|0.61|0.65|0.63|4004
|news_travel|0.57|0.56|0.57|3715
|news_world|0.58|0.61|0.60|5314
|news_stock|0.49|0.30|0.37|281
|news_agriculture|0.64|0.67|0.65|3134
|news_game|0.75|0.71|0.73|3719
|accuracy|0.66|58360
|macro avg|0.66|0.64|0.64|58360
|weighted avg|0.66|0.66|0.66|58360
训练集结果：

f1 Score(weighted)：0.681909011279105

||precision|recall|f1-score|support|
|:----|:----:|:----:|:----:|:---:|
|news_story|0.70|0.54|0.61|116
|news_culture|0.70|0.70|0.70|367
news_entertainment|0.66|0.74|0.70|443
|news_sports|0.85|0.79|0.82|393
|news_finance|0.60|0.55|0.57|496
|news_house|0.75|0.79|0.77|183
|news_car|0.79|0.79|0.79|395
|news_edu|0.71|0.68|0.70|309
|news_tech|0.61|0.67|0.64|569
|news_military|0.63|0.68|0.65|344
|news_travel|0.61|0.59|0.60|346
|news_world|0.59|0.61|0.60|442
|news_stock|0.60|0.14|0.23|21
|news_agriculture|0.69|0.72|0.71|246
|news_game|0.83|0.73|0.78|330
|accuracy|||0.68|5000
|macro avg|0.69|0.65|0.66|5000
|weighted avg|0.68|0.68|0.68|5000


