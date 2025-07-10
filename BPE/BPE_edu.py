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
        if len(data) < 2:
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
                i += 2
            else:
                merged_data.append(data[i])
                i += 1
            
        return merged_data
    
    def train(self, verbose=False):
        """
        Train the BPE tokenizer by iteratively merging most frequent pairs.
        
        Args:
            verbose (bool): Whether to print progress information
        """
        if not self.datalist:
            raise ValueError("No data loaded. Call load_data() first.")
        
        if not self.vocab:
            self.build_initial_vocabulary()

        merge_count = 0
        target_merges = self.max_vocab_size - self.original_vocab_length
        
        print(f"Starting BPE training. Target merges: {target_merges}")

        while merge_count < target_merges:
            # Find most frequent pair
            freq_dist = self._find_most_frequent_pair(self.datalist)

            if not freq_dist:
                print("No more pairs to merge.")
                break

            # Get the most frequent pair
            most_frequent_pair = max(freq_dist.items(), key=lambda x: x[1])
            (token1, token2), frequency = most_frequent_pair
            if frequency < 2:  # Stop if no pair appears more than once
                print("No frequent pairs remaining.")
                break
            
            # Merge the tokens
            merged_token = token1 + token2
            self.datalist = self._merge_tokens(token1, token2, self.datalist)
            
            # Update vocabulary
            new_token_id = self.original_vocab_length + merge_count
            self.vocab.append(merged_token)
            self.vocab_to_id[merged_token] = new_token_id
            self.id_to_vocab[new_token_id] = merged_token
            
            merge_count += 1
            
            if verbose:
                print(f"Merge {merge_count}: '{token1}' + '{token2}' -> '{merged_token}' (freq: {frequency})")
        
        print(f"Training completed. Final vocabulary size: {len(self.vocab)}")
    
    def encode(self, text):
        """
        Encode text using the trained BPE tokenizer.
        
        Args:
            text (str): Text to encode
            
        Returns:
            list: List of token IDs
        """
        if not self.vocab_to_id:
            raise ValueError("Tokenizer not trained. Call train() first.")
        
        # This is a simplified encoding - in practice, you'd need more sophisticated logic
        tokens = list(text)
        encoded = []
        
        for token in tokens:
            if token in self.vocab_to_id:
                encoded.append(self.vocab_to_id[token])
            else:
                # Handle unknown tokens (could use UNK token)
                print(f"Warning: Unknown token '{token}'")
        
        return encoded
    
    def decode(self, token_ids):
        """
        Decode token IDs back to text.
        
        Args:
            token_ids (list): List of token IDs
            
        Returns:
            str: Decoded text
        """
        if not self.id_to_vocab:
            raise ValueError("Tokenizer not trained. Call train() first.")
        
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_vocab:
                tokens.append(self.id_to_vocab[token_id])
            else:
                print(f"Warning: Unknown token ID {token_id}")
        
        return ''.join(tokens)
    
    def get_vocabulary(self):
        """Return the current vocabulary."""
        return self.vocab.copy()
    
    def get_vocab_size(self):
        """Return the current vocabulary size."""
        return len(self.vocab)
    
    def save_model(self, file_path):
        """Save the trained model to a file."""
        import json
        
        model_data = {
            'vocab': self.vocab,
            'vocab_to_id': self.vocab_to_id,
            'id_to_vocab': {str(k): v for k, v in self.id_to_vocab.items()},
            'max_vocab_size': self.max_vocab_size
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, ensure_ascii=False, indent=2)
        
        print(f"Model saved to {file_path}")


# Usage example
if __name__ == "__main__":
    # Initialize the tokenizer
    tokenizer = BPETokenizer(
        file_path="D:/Desktop/input.txt",
        max_vocab_size=55,
        data_limit=1000
    )
    
    # Train the tokenizer
    tokenizer.load_data()
    tokenizer.train(verbose=True)
    
    # Use the tokenizer
    print("\nFinal vocabulary:")
    vocab = tokenizer.get_vocabulary()
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    print(f"Sample vocab: {vocab[:10]}")
    
    # Example encoding/decoding
    sample_text = "hello"
    encoded = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(encoded)
    print(f"\nOriginal: '{sample_text}'")
    print(f"Encoded: {encoded}")
    print(f"Decoded: '{decoded}'")
    
    # Save the model
    tokenizer.save_model("bpe_model.json")
