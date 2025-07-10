class Bpe:
    with open("D:/Desktop/input.txt", "r") as f:
        dataset = f.read()

    datalist = list(dataset)
    datalist =datalist[:1000]
    datalist

    # first find the vocabulary
    vocab = sorted(set(datalist))
    vocab_length = len(vocab)
    vocab[:5]
    vocab_length

    # build vocabulary to idx and idx to vocabulary
    vc2id = {v: k for k, v in enumerate(vocab)}
    id2vc = {k: v for k, v in enumerate(vocab)}
    print(vc2id)
    print(id2vc)

    # define the function to find the most frequent pair of the vocabulary list
    def freq_vocab(dataset):
        dataset1 = dataset[:-1]
        dataset2 = dataset[1:]
        freq_dist = {}
        for word1, word2 in zip(dataset1, dataset2):
            word = (word1, word2)
            freq_dist[word] = freq_dist.get(word, 0) + 1
        return freq_dist


    # when fing the most frequent word combination, merge it.
    def merge(word1, word2, dataset):
        i = 0
        while i < len(dataset) - 1:
            if dataset[i] == word1 and dataset[i + 1] == word2:
                dataset = dataset[:i] + [word1 + word2] + dataset[i+2:]
            else:
                i += 1
        return dataset

    # set the vocabulary size and merge the word until it reach the vocabulary size.
    max_length = 55
    i = 0
    while i + vocab_length < max_length:
        freq_list = freq_vocab(datalist)
        word1, word2 = max(freq_list, key=lambda x: x[1])
        datalist = merge(word1, word2, datalist)
        vocab.append(word1+word2)
        vc2id[word1+word2] = i + vocab_length
        id2vc[i + vocab_length] = word1+word2
        i += 1
        print(datalist)

    vocab


