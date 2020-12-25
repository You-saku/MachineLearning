import tensorflow as tf
import MeCab
import glob
import numpy as np

wakati = MeCab.Tagger("-Owakati")#分かち書きオブジェクト

files = glob.glob("kanemura/*.txt")#ファイルないのテキストファイル全部


"""
for file in files:
    print(file)
"""

text = "" #ここにテキストを入れ込んでいく

for file in files:
    f = open(file,encoding="utf-8")
    text += f.read()

f.close()

words = wakati.parse(text).split()#分かち書き実行
#print(words)

vocab = sorted(set(words)) #setで重複をなくす
#print ('{} unique characters'.format(len(vocab))) #語彙数がわかる


char2idx = {u:i for i, u in enumerate(vocab)} #単語からインデックスを　#単語がキー、値が数値
idx2char = np.array(vocab) #配列インデックスが単語の番号になる
#print(len(char2idx))
#print(len(idx2char))

text_as_int = np.array([char2idx[c] for c in words])#分かち書き後のデータをループしような... これで文字を数値に置き換えることに成功

"""
#最初の10単語を数値化
print(words[:10],end=" = ") 
print(text_as_int[:10])
"""

# ひとつの入力としたいシーケンスの文字数としての最大の長さ
seq_length = 100

examples_per_epoch = len(words)//(seq_length+1) #整数へ丸める割り算　これでエポック数を求められる

text_dataset = tf.data.Dataset.from_tensor_slices(text_as_int) #データセット化する

"""
for i in text_dataset.take(5):
    print(idx2char[i.numpy()])
"""

sequences = text_dataset.batch(seq_length+1, drop_remainder=True) #バッチサイズに分ける

"""
for item in sequences.take(5):#5個のバッチサイズを表示する
    print(repr(''.join(idx2char[item.numpy()])))
    print()
"""

#インプットとターゲットにわかれる
#これで今の単語と次の単語ができる
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target) #map関数でsplit_input_targetを関数としてセットしておく これで読んだときにこの関数が実行される

#１つめのバッチサイズに対して行うよ
"""
for input_example, target_example in  dataset.take(1):
    print ('Input data: ', repr(''.join(idx2char[input_example.numpy()])))
    print ('Target data:', repr(''.join(idx2char[target_example.numpy()])))
"""

"""
print("データを分ける")
print()
for input_example, target_example in  dataset.take(1):
    print ('Input data: ', repr(''.join(idx2char[input_example.numpy()])))
    print ('Target data:', repr(''.join(idx2char[target_example.numpy()])))


print("単語を見てみよう")
print()
#バッチサイズに対してsplitしないとこれはできない
#バッチサイズ1に対して5文字までを見る
for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
    print("Step {:4d}".format(i))
    print("  input: {} ({:s})".format(input_idx, repr(idx2char[input_idx])))
    print("  expected output: {} ({:s})".format(target_idx, repr(idx2char[target_idx])))
"""

# バッチサイズ
#ここが大きすぎるとデータセットの入力に影響がでる
BATCH_SIZE = 25

# データセットをシャッフルするためのバッファサイズ
# （TF data は可能性として無限長のシーケンスでも使えるように設計されています。
# このため、シーケンス全体をメモリ内でシャッフルしようとはしません。
# その代わりに、要素をシャッフルするためのバッファを保持しています）

BUFFER_SIZE = 10000#どれだけのサンプルからランダムに選ぶかを決められるデータサイズ以上に設定すればデータは完全にシャッフルされる

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

#print(dataset)


# 文字数で表されるボキャブラリーの長さ
vocab_size = len(vocab)#重複をなくした全文字数

# 埋め込みベクトルの次元
embedding_dim = 256

# RNN ユニットの数
rnn_units = 1024


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):#モデル作成する関数
    model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                            batch_input_shape=[batch_size, None]),
    tf.keras.layers.GRU(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer=tf.keras.initializers.glorot_uniform),
    tf.keras.layers.Dense(vocab_size)
    ])
    return model

model = build_model(
    vocab_size = len(vocab),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units,
    batch_size=BATCH_SIZE)

model.summary()
#print(dataset.take(1))


#1つめのバッチをテストデータ化
for input_example_batch, target_example_batch in dataset.take(1):
    example_batch_predictions = model(input_example_batch)#ネットワークに訓練データを入力
    print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")#ネットワークの出力のデータサイズを出力



sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)#num_samples 何次元か(今回は1次元)
sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy() #.squeeze -> 同じ行列は消える オプションで決まった軸に対して行うことができる

print("Input: \n", repr("".join(idx2char[input_example_batch[0]])))#元データを出す(通常の文章)
print()
print("Next Char Predictions: \n", repr("".join(idx2char[sampled_indices ])))#生成したデータで出力する(未訓練)


