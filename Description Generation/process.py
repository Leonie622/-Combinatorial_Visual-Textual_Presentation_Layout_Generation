from re import T
import pandas as pd
from sklearn.model_selection import train_test_split
import json

data = pd.read_excel('语料表.xlsx')

x, y = data['P3标签'], data['宣传语（4-8）最高不超过20字']

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=1)

max_len = 0
max_len_s = 0

train_len_content = 0
train_len_title = 0

with open('train.json', 'w', encoding='utf-8') as f:
    for i, j in zip(train_x, train_y):
        for t in j.split('\n'):
            t = t.replace(' ', '').replace('\t', '')
            max_len = max(max_len, len(t))
            train_len_content += len(i)
            train_len_title += len(t)
            max_len_s = max(max_len_s, len(i))
            res = {"content": i, "title":t}
            f.write(f"{json.dumps(res, ensure_ascii=False)}\n")
print(max_len)
print(max_len_s)
print(f'train_len_content: {train_len_content}, train_len_title:{train_len_title}, train_len_content+train_len_title:{train_len_content+train_len_title}')

test_len_content = 0
test_len_title = 0
with open('test.json', 'w', encoding='utf-8') as f:
    for i, j in zip(test_x, test_y):
        for t in j.split('\n'):
            t = t.replace(' ', '').replace('\t', '')
            max_len = max(max_len, len(t))
            max_len_s = max(max_len_s, len(i))
            test_len_content += len(i)
            test_len_title += len(t)
            res = {"content": i, "title":t}
            f.write(f'{json.dumps(res, ensure_ascii=False)}\n')
print(max_len)
print(max_len_s)
print(f'test_len_content: {test_len_content}, test_len_title:{test_len_title}, test_len_content+test_len_title:{test_len_content+test_len_title}')
