# fnewsDataMining
上海大学数据挖掘课程大作业

预训练模型：[chinese_L-12_H-768_A-12](https://drive.google.com/drive/folders/1tAIO4vcUHAxwf0KMSz73NLycjzVc7Q6v?usp=sharing)

训练结果(epoch = 4)：[model_save](https://drive.google.com/drive/folders/1l09n4B-7r0Qpe5lGEkF6q0XGZZfP93rV?usp=sharing)

五折交叉训练结果：

f1 Score(weighted)：0.5609313463725751

||precision|recall|f1-score|support|
|:----|:----:|:----:|:----:|:---:|
|news_story|0.49|0.41|0.44|1210|
|news_culture|0.54|0.61|0.57|4450|
|news_entertainment|0.58|0.59|0.58|5443|
|news_sports|0.75|0.69|0.72|4365|
|news_finance|0.48|0.50|0.49|5660|
|news_house|0.61|0.62|0.61|2302|
|news_car|0.69|0.64|0.66|4514|
|news_edu|0.58|0.56|0.57|3774|
|news_tech|0.52|0.55|0.53|6475|
|news_military|0.54|0.51|0.52|4004|
|news_travel|0.49|0.44|0.46|3715|
|news_world|0.51|0.50|0.50|5314|
|news_stock|0.45|0.37|0.40|281|
|news_agriculture|0.50|0.54|0.51|3134|
|news_game|0.63|0.64|0.63|3719|
|accuracy|||0.56|58360
|macro avg|0.56|0.54|0.55|58360
|weighted avg|0.56|0.56|0.56|58360

训练集结果：

f1 Score(weighted)：0.5880533813726921

||precision|recall|f1-score|support|
|:----|:----:|:----:|:----:|:---:|
|news_story|0.67|0.48|0.56|116|
|news_culture|0.54|0.61|0.57|367|
news_entertainment|0.58|0.62|0.60|443|
|news_sports|0.78|0.73|0.75|393|
|news_finance|0.49|0.52|0.50|496|
|news_house|0.62|0.65|0.63|183|
|news_car|0.70|0.65|0.67|395|
|news_edu|0.56|0.57|0.57|309|
|news_tech|0.56|0.59|0.58|569|
|news_military|0.57|0.51|0.54|344|
|news_travel|0.56|0.48|0.52|346|
|news_world|0.54|0.55|0.54|442|
|news_stock|0.42|0.24|0.30|21|
|news_agriculture|0.51|0.59|0.54|246|
|news_game|0.71|0.67|0.69|330|
|accuracy|||0.59|5000|
|macro avg|0.59|0.56|0.57|5000|
|weighted avg|0.59|0.59|0.59|5000|
