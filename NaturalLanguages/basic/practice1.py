words = "パタトクカシーー"

print(words[1]+words[3]+words[5]+words[7])

word1 = "パトカー"
word2 = "タクシー"

output = ""
for i,j in zip(word1,word2):
    index = 0
    output+=(i[index]+j[index])
    index+=1

print(output)#2文字の合成