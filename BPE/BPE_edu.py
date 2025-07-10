class BPETokenizer:
    def __init__(self, file_path, max_vocab_size=55, data_limit=1000):
        """Initialize the BPE tokenizer.
        
        Args:
            file_path (str): Path to the input text file
            max_vocab_size (int): Maximum vocabulary size after merging
            data_limit (int): Limit on number of characters to process
        """
        self.file_path = file_path
        self.max_vocab_size = max_vocab_size
        self.data_limit = data_limit

        # Initialize instance variables
        self.dataset = ""
        self.datalist = []
        self.vocab = []
        self.vocab_to_id = {}
        self.id_to_vocab = {}
        self.original_vocab_length = 0

    def load_data(self):
        """Load and preprocess data from file."""
        try:
            with open(self.file_path, "r", encoding='utf-8') as f:
                self.dataset = f.read()
            
            self.datalist = list(self.dataset)[:self.data_limit]
            print(f"Loaded {len(self.datalist)} characters from {self.file_path}")
        
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {self.file_path}")
        except Exception as e:
            raise Exception(f"Error loading file: {e}")
        
    def build_initial_vocabulary(self):
        """Build initial vocabulary from unique characters."""
        self.vocab = sorted(set(self.datalist))
        self.original_vocab_length = len(self.vocab)

        # Build mapping
        self.vocab_to_id = {char: idx for idx, char in enumerate(self.vocab)}
        self.id_to_vocab = {idx: char for idx, char in enumerate(self.vocab)}

        print(f"Initial vocabulary size: {self.original_vocab_length}")
        print(f"First 5 vocab items: {self.vocab[:5]}")

    def _find_most_frequent_pair(self, data):
        """
        Find the most frequent consecutive pair in the data.
        
        Args:
            data (list): List of tokens/characters
            
        Returns:
            dict: Frequency distribution of pairs
        """
        if len(data < 2):
            return {}
        
        freq_dist = {}
        for i in range(len(data) - 1):
            pair = (data[i], data[i + 1])
            freq_dist[pair] = freq_dist.get(pair, 0) + 1

        return freq_dist
    
    def _merge_tokens(self, token1, token2, data):
        """
        Merge consecutive occurrences of token1 and token2 in the data.
        
        Args:
            token1 (str): First token to merge
            token2 (str): Second token to merge
            data (list): List of tokens
            
        Returns:
            list: Data with merged tokens
        """
        merged_data = []
        i = 0

        while i < len(data):
            if i < len(data) - 1 and data[i] == token1 and data[i + 1] == token2:
                merged_data.append(token1 + token2)
