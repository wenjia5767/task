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

        print(f"Initial vocabulary size: {s}")
