from BPE_edu import BPETokenizer as BPETokenizer

def test_merge():
    tokenizer = BPETokenizer(
        file_path="D:/Desktop/input.txt",
        max_vocab_size=55,
        data_limit=1000
    )
    data = ['i', 'o', 'j']
    result = tokenizer._merge_tokens('i', 'o', data)
    assert result == ['io', 'j']

def test_fre():
    tokenizer = BPETokenizer(
        file_path="D:/Desktop/input.txt",
        max_vocab_size=55,
        data_limit=1000
    )
    # Test data that matches your expected output
    data = ['i', 'o', 'i', 'o', 'p', 'p', 'p', 'i']
    
    result = tokenizer._find_most_frequent_pair(data)
    expected = {('i', 'o'): 2, ('o', 'i'): 1, ('o', 'p'): 1, ('p', 'p'): 2, ('p', 'i'): 1}
    
    assert result == expected