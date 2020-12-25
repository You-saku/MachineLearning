import numpy as np

list = ["a","b","c","d","e"]
dic = { j:i for i,j in enumerate(list)}#辞書型をfor文で作成できる

print(dic)

list2 = np.array(list)
print(list2)