import MeCab
wakati = MeCab.Tagger("-Owakati")
with open("sample1.txt",encoding='utf-8') as f:
    text = f.read()

f.close()

words = wakati.parse(text).split()
print(words)