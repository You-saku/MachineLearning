#これを上手く使えばb文章から最大頻出単語を並べられる
#話し方の癖を探れるかも

import numpy as np
import MeCab
wakati = MeCab.Tagger("-Owakati")
words = wakati.parse("私は何なのだろうか... その謎を解明するために我々はAmazonへ向かった。私、私私").split()

points = np.zeros(len(set(words))) #重複はなし

print(words)


array = [i for i in set(words)] #重複なしで全単語リストを作成
print(array)

print("集計前")
print(points)

for i in words:
    if i in array:
        points[array.index(i)]+=1

print("集計後")
print(points)#全部1になるのは当たり前だよな

print("頻出単語は")
max_index = np.argmax(points)#最大値のインデックスを返す
print("「"+array[max_index]+"」")