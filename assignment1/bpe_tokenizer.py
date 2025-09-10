import argparse
import collections
import pickle

class BPETokenizer:
    def __init__(self, max_vocab_size):
        self.max_vocab_size = max_vocab_size
        self.vocab = set()
        self.merges = []
    
    def train(self, corpus_file, vocab_file):
        with open(corpus_file, 'r', encoding='utf-8') as f:
            text = f.read().lower()
        
        
        vocab = collections.Counter([" ".join(word) + " </w>" for word in text.split()])
        vocab = {tuple(word.split()): count for word, count in vocab.items()}

        self.vocab = {symbol for word in vocab for symbol in word}  
        
        while len(self.vocab) < self.max_vocab_size:
            pairs = self.get_pairs(vocab)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            self.merges.append(best_pair)
            
            vocab = self.merge_vocab(best_pair, vocab)

            merged_token = "".join(best_pair)
            self.vocab.add(merged_token) 
        
        with open(vocab_file, 'wb') as f:
            pickle.dump((self.merges, self.vocab), f)
    
    def get_pairs(self, vocab):
        pairs = collections.defaultdict(int)
        for word, freq in vocab.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += freq
        return pairs
    
    def merge_vocab(self, pair, vocab):
        new_vocab = {}
        for word, freq in vocab.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                    new_word.append(word[i] + word[i + 1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_vocab[tuple(new_word)] = freq
        return new_vocab
    
    def load_vocab(self, vocab_file):
        with open(vocab_file, 'rb') as f:
            self.merges, self.vocab = pickle.load(f)
    
    def encode(self, text):
        words = text.lower().split()
        encoded_words = []
        
        for word in words:
            word_chars = list(word) + ['</w>']
            while len(word_chars) > 1:
                pairs = [(word_chars[i], word_chars[i + 1]) for i in range(len(word_chars) - 1)]
                best_pair = None
                for pair in self.merges:
                    if pair in pairs:
                        best_pair = pair
                        break
                if best_pair is None:
                    break
                new_word = []
                i = 0
                while i < len(word_chars):
                    if i < len(word_chars) - 1 and (word_chars[i], word_chars[i + 1]) == best_pair:
                        new_word.append(word_chars[i] + word_chars[i + 1])
                        i += 2
                    else:
                        new_word.append(word_chars[i])
                        i += 1
                word_chars = new_word
            
            encoded_words.append(" ".join([word if i == 0 else "##" + word for i, word in enumerate(word_chars)]).replace('</w>', ''))
        return " ".join(encoded_words)
    
    def tokenize(self, input_file, output_file):
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
        tokenized_text = self.encode(text)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(tokenized_text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="Train the tokenizer with corpus file")
    parser.add_argument("--max_vocab", type=int, help="Maximum vocabulary size")
    parser.add_argument("--vocab", help="Vocabulary output file")
    parser.add_argument("--infer", help="Vocabulary input file for inference")
    parser.add_argument("--input", help="Input text file for inference")
    parser.add_argument("--output", help="Output text file for inference")
    args = parser.parse_args()
    
    tokenizer = BPETokenizer(args.max_vocab if args.max_vocab else 3000)
    
    if args.train and args.vocab:
        tokenizer.train(args.train, args.vocab)
    elif args.infer and args.input and args.output:
        tokenizer.load_vocab(args.infer)
        tokenizer.tokenize(args.input, args.output)