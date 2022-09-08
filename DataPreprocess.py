preprocess = DataPreprocess()

# 处理文本
texts_cut = preprocess.cut_texts(texts, word_len)
preprocess.train_tokenizer(texts_cut, num_words)
texts_seq = preprocess.text2seq(texts_cut, sentence_len)

# 得到标签
preprocess.creat_label_set(labels)
labels = preprocess.creat_labels(labels)