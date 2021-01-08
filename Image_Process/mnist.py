import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_train,x_test = x_train/255.0,x_test/255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128,activation="relu"),
    tf.keras.layers.Dense(128,activation="relu"),
    tf.keras.layers.Dense(10,activation="softmax")
])

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) #from_logits = True は計算が安定するぞ

model.compile(
    optimizer="adam",
    loss=loss,
    metrics=['accuracy'] #結果を返す?
)

model.fit(x=x_train,y=y_train,epochs=10,verbose=1,validation_data=(x_test,y_test),shuffle=True)

model.save("model_save(h5)/20210109.h5")