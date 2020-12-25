import tensorflow as tf
import numpy as np
import os
import time

path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

# 読み込んだのち、Python 2 との互換性のためにデコード
text = open(path_to_file, 'rb').read().decode(encoding='utf-8') #1文字単位で文字が格納される(配列)
# テキストの長さは含まれる文字数
#print ('Length of text: {} characters'.format(len(text)))

# ファイル中のユニークな文字の数
vocab = sorted(set(text))
#print ('{} unique characters'.format(len(vocab)))

char2idx = {u:i for i, u in enumerate(vocab)}#辞書型をつくる特殊なfor文 {文字,インデックス}
#print(type(char2idx))
idx2char = np.array(vocab)#[文字]
#print(idx2char)

text_as_int = np.array([char2idx[c] for c in text]) #テキスト最初から読み込んでインデックス何番が出てきたのかを格納し続ける インデックスをtextにして何の数字にあたいするのか調べる
#print(text_as_int)

"""とりあえず10個
for i in range(10):
    print(text_as_int[i])
"""
#1つめの文字が何か調べる


# ひとつの入力としたいシーケンスの文字数としての最大の長さ
seq_length = 100
examples_per_epoch = len(text)//(seq_length+1)

# 訓練用サンプルとターゲットを作る
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int) #文字では扱えないので数値で取り扱う
#print(char_dataset)

for i in char_dataset.take(5):
    #print(i) <- tensor型だったからnumpyで型を変換する
    print(idx2char[i.numpy()])