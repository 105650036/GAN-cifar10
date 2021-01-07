# GAN-cifar10

這次實作目標是透過GAN產生cifar10的圖片
在先前自己有做過0-9手寫字的GAN很順利的完成了
但在做cifar10的生成時一直失敗 我想原因也跟圖片的複雜度有關 一個是32*32*3 一個是28*28*1
後來也有查閱資料發現GAN一定的缺點就是不穩定 所以我也透過查詢一些別人的實驗結果來調整參數最後也順利完成

調整的部分為重relu變成leakyrelu 並且參數設為0.2 
以及Adam的lr beta_1
參考論文(UNSUPERVISED REPRESENTATION LEARNING WITH DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS)

原本有使用BatchNormalization 但結果不好 參考另外一位沒使用去掉此層後也就成功了

#定義判別器其內容為CNN輸出為是假的(0)或不是假的(1)
def define_discriminator(in_shape=(32,32,3)):
	model = Sequential()
	model.add(Conv2D(64, (3,3), padding='same', input_shape=in_shape))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Conv2D(256, (3,3), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Flatten())
	model.add(Dropout(0.4))
	model.add(Dense(1, activation='sigmoid'))
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model

# 定義生產器
def define_generator(latent_dim):
	model = Sequential()
	# 這邊的數字256*4*4是可以隨意改變的 我們的目的是要生產出32*32*3的圖片
	n_nodes = 256 * 4 * 4
	model.add(Dense(n_nodes, input_dim=latent_dim))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Reshape((4, 4, 256)))
	# 變成8*8*128
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# 變成16*16*128
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# 變成32*32*128
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# 輸出->變成32*32*3
	model.add(Conv2D(3, (3,3), activation='tanh', padding='same'))
	return model
