#文字辞書作る
print('辞書をつくるための文字を入力')
words = input()

dic = {} #辞書初期化
for i,j in enumerate(set(words)):
    dic[j] = i
print("辞書は以下の通りです")
print(dic)

count = [0]*len(set(words))#カウント用配列

print("文字を入力してください")
check = input()

err = 0
for i in check:
    if(i in dic):
        count[dic[i]]+=1
    else:
        err+=1


print("辞書で検索した結果(ヒット数)")
print(count)
print("エラー数")
print(err)