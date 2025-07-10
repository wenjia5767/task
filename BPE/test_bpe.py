from BPE_py import Bpe as bpe

def test_merge():
    data = ['i', 'o', 'j']
    assert bpe.merge('i', 'o', data) == ['io', 'j']

def test_fre():
    data = ['i', 'o', 'p', 'p', 'i', 'o']
    assert bpe.freq_vocab(data) == {('i', 'o'):2, ('o', 'p'):1, ('p', 'p'):1,('p', 'i'):1}