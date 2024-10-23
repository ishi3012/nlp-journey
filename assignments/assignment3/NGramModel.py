import pandas as pd
import numpy as np
from enum import Enum
from collections import defaultdict, Counter
from typing import List, Tuple, Dict
import time
from nltk.probability import ConditionalFreqDist



class NgramType(Enum):
    # UNIGRAM = 1
    BIGRAM = 2
    TRIGRAM = 3
    # QUADGRAM = 4
    # PENTAGRAM = 5
    # HEXAGRAM = 6

class Strategy(Enum):
    GREEDY = "greedy"
    RANDOM = "random"
    TOP_P  = "top_p"

class NgramModel:
    def __init__(self, n):
        """
        Initialize the N-gram model.

        :param n: Size of the n-gram (1 for unigram, 2 for bigram, etc.)
        """
        self.n = n
        self.ngram_counts = defaultdict(int)
        self.n_minus_1_gram_counts = defaultdict(int)
        self.cfd = defaultdict(lambda: defaultdict(float))  # Conditional Frequency Distribution
        self.vocabulary = set()
        self.token_counts = Counter()
        self.start_token = "<START>"
        self.end_token = "<STOP>"

    def preprocess(self, sentence):
        """
        Preprocess the input sentence by lowercasing and tokenizing it.

        :param sentence: The input sentence (string).
        :return: List of tokens with sentence-level start and end padding.
        """
        # Tokenize the sentence into words
        tokens = sentence.split()  # Tokenization and lowercasing
        
        # Add <s> at the start and <STOP> at the end of the sentence
        tokens = [self.start_token] * (self.n - 1) + tokens + [self.end_token]
        
        return tokens

    def build_vocabulary(self, corpus, min_count=3):
        """
        Build the vocabulary from the training data. Any token occurring fewer than 'min_count'
        times is replaced by '<unk>'.

        :param corpus: List of sentences (strings).
        :param min_count: Minimum frequency for a token to be included in the vocabulary.
        """
        # Count tokens
        for sentence in corpus:
            tokens = self.preprocess(sentence)
            self.token_counts.update(tokens)

        # Filter out tokens that occur fewer than min_count times
        self.vocabulary = {token for token, count in self.token_counts.items() if count >= min_count and token != self.start_token}
        
        # Add <unk> for rare words and <STOP> symbol
        self.vocabulary.add("<unk>")
        self.vocabulary.add("<STOP>")

        # Ensure vocabulary size is exactly 26,602 as specified
        # print(f"Vocabulary Size = {len(self.vocabulary)}.") 

    def replace_rare_tokens(self, tokens):
        """
        Replace rare tokens in the token list (those not in the vocabulary) with '<unk>'.

        :param tokens: List of tokens.
        :return: List of tokens with rare tokens replaced by '<unk>'.
        """
        result = []
        for token in tokens:
            if token in self.vocabulary or token == self.start_token:
                result.append(token)
            else:
                result.append("<unk>")
        return result

    def get_ngrams(self, tokens):
        """
        Generate n-grams from the tokens.

        :param tokens: List of tokens.
        :return: List of n-grams.
        """
        ngrams = [tuple(tokens[i:i+self.n]) for i in range(len(tokens)-self.n+1)]
        return ngrams

    def train(self, corpus):
        """
        Train the N-gram model on the given corpus (list of sentences).

        :param corpus: List of sentences (strings).
        """
        tokens_list = []
        
        for sentence in corpus:
            tokens = self.preprocess(sentence)
            tokens = self.replace_rare_tokens(tokens)  # Replace rare tokens with <unk>
            ngrams = self.get_ngrams(tokens)
            tokens_list.append(tokens)

            # print(ngrams[:5])

            # Count n-grams and (n-1)-grams
            for ngram in ngrams:
                self.ngram_counts[ngram] += 1
                self.n_minus_1_gram_counts[ngram[:-1]] += 1
           
            # for ngram, count in self.ngram_counts.items():
            #     context = ngram[:-1]
            #     next_word = ngram[-1]
            #     context_count = self.n_minus_1_gram_counts[context]

            #     if context_count > 0:
            #         self.cfd[context][next_word] = count / context_count
    def compute_cfd(self):
        for ngram, count in self.ngram_counts.items():
                context = ngram[:-1]
                next_word = ngram[-1]
                context_count = self.n_minus_1_gram_counts[context]

                if context_count > 0:
                    self.cfd[context][next_word] = count / context_count

    def calculate_probability(self, ngram, smoothing=1):
        """
        Calculate the probability of an n-gram with optional Laplace smoothing.

        :param ngram: The n-gram tuple.
        :param smoothing: Laplace smoothing factor (default: 1).
        :return: The probability of the n-gram.
        """
        ngram_count = self.ngram_counts[ngram]
        n_minus_1_gram_count = self.n_minus_1_gram_counts[ngram[:-1]]
        total_count = n_minus_1_gram_count + smoothing * len(self.vocabulary)

        # Apply Laplace smoothing
        probability = (ngram_count + smoothing) / (total_count) if total_count > 0 else 0.0
        
        return probability

    def calculate_sentence_perplexity(self, sentence, smoothing=1):
        """
        Calculate the perplexity for a single sentence using the provided formula.

        :param sentence: A single sentence (string).
        :param smoothing: Laplace smoothing factor (default: 1).
        :return: The perplexity for the sentence.
        """        
        tokens = self.preprocess(sentence)
        tokens = self.replace_rare_tokens(tokens)  # Replace rare tokens with <unk>
        ngrams = self.get_ngrams(tokens)
        
        sentence_log_prob = 0.0
        for ngram in ngrams:
            prob = self.calculate_probability(ngram, smoothing)
            # print(f" ngram: {ngram}, probability: {prob}")
            # sentence_log_prob += math.log(prob)
            sentence_log_prob -= np.log(prob) if prob != 0 else 0.0
            
        # Number of words in the sentence (number of n-grams)
        num_ngrams = len(ngrams)

        # Calculate the perplexity for this sentence
        perplexity = np.exp(sentence_log_prob / num_ngrams)
        # print(f" sentence: {sentence}, perplexity: {perplexity}")
        return perplexity
  
    def calculate_average_perplexity(self, test_corpus, smoothing=1):
        """
        Calculate the average perplexity across all sentences in the test data.

        :param test_corpus: List of sentences in the test data.
        :param smoothing: Laplace smoothing factor (default: 1).
        :return: The average perplexity score.
        """
        total_perplexity = 0.0
        # total_sentences = len(test_corpus)
        total_sentences = 0.0

        for sentence in test_corpus:
            sentence_perplexity = self.calculate_sentence_perplexity(sentence, smoothing)
            total_perplexity += sentence_perplexity  # Sum of perplexity for each sentence
            total_sentences += 1            

        # Calculate the average perplexity
        average_perplexity = total_perplexity / total_sentences if total_sentences > 0 else 0.0
        return average_perplexity

    def load_data(self, file_path):
        """
        Load the data from a file.

        :param file_path: Path to the file containing sentences.
        :return: List of sentences (strings).
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            sentences = [line.strip() for line in file if line.strip()]  # Remove empty lines and strip whitespace
        return sentences

    def generate_text(self, max_length: int = 100, strategy: str = "greedy", top_p: float = 0.9) -> str:
        """
        Generate text using the trained language model.

        :param max_length: Maximum number of words to generate.
        :param strategy: Strategy to select the next word ('greedy', 'random', 'top_p').
        :param top_p: For top-p sampling, the cumulative probability threshold.
        :return: The generated sentence as a string.
        """
        current_ngram = (self.start_token,) * (self.n - 1)
        sentence = list(current_ngram)

        self.compute_cfd()

        for _ in range(max_length):
            next_word = self.choose_next_word(current_ngram, strategy, top_p)
            
            # Handle <STOP> and unknown contexts
            # if next_word == '<STOP>' or next_word == '<unk>':
            if next_word == '<STOP>':
                break
            
            sentence.append(next_word)
            
            # Update the context to the most recent (n-1) words
            current_ngram = tuple(sentence[-(self.n - 1):])

        return ' '.join(sentence[self.n - 1:])  # Exclude <START> tokens in the final output

    def choose_next_word(self, context: Tuple[str, ...], strategy: str = "greedy", top_p: float = 0.9) -> str:
        """
        Chooses the next word based on the provided strategy (greedy, random, or top-p).

        :param context: The context (n-1) words for the n-gram model.
        :param strategy: Strategy to select the next word ('greedy', 'random', 'top_p').
        :param top_p: For top-p sampling, the cumulative probability threshold.
        :return: The selected word.
        """
        # Handle unseen context by returning '<unk>' or some fallback strategy
        if context not in self.cfd:
            return '<unk>'

        if strategy == "greedy":
            # Choose the word with the highest probability
            return max(self.cfd[context], key=self.cfd[context].get)
        
        elif strategy == "random":
            # Choose based on probabilities (not uniformly)
            words, probs = zip(*self.cfd[context].items())
            return np.random.choice(words, p=probs)
        
        elif strategy == "top_p":
            # Top-p sampling
            probs = np.array(list(self.cfd[context].values()))
            words = list(self.cfd[context].keys())
            total_prob = np.sum(probs)
            
            # Sort probabilities and words by descending probability
            sorted_indices = np.argsort(-probs)
            sorted_probs = probs[sorted_indices]
            sorted_words = [words[i] for i in sorted_indices]
            
            # Calculate cumulative probabilities
            cumulative_prob = np.cumsum(sorted_probs) / total_prob
            cutoff_index = np.argmax(cumulative_prob >= top_p)
            
            # Select words within the top-p threshold
            top_p_words = sorted_words[:cutoff_index + 1]
        return np.random.choice(top_p_words)
    
    def save_generated_text_to_file(self, generated_text: str, file_path: str, strategy: str, top_p: float, ngramtype: NgramType):
        """
        Saves the generated text to a file with some metadata.

        :param generated_text: The text that was generated.
        :param file_path: The path to the file where the text will be saved.
        :param strategy: The strategy used for text generation (greedy, random, top_p).
        :param top_p: The top_p value used during text generation (only relevant for top-p strategy).

        """
        with open(file_path, "a") as file:
            # Write some metadata about the generation process
            file.write(f"--- Generated Text - {ngramtype} ---\n")
            file.write(f"Strategy: {strategy}\n")
            file.write(f"Top-p: {top_p}\n")
            file.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            file.write(f"Generated Text:\n{generated_text}\n")
            file.write("\n----------------------\n\n")


if __name__ == "__main__":
    # File paths for training and test data
    train_file = "assignments/assignment3/InputFiles/1b_benchmark.train.tokens.txt"  
    test_file = "assignments/assignment3/InputFiles/1b_benchmark.test.tokens.txt" 
    output_file_path = "assignments/assignment3/InputFiles/generated_texts.txt"

    for ngramtype in NgramType:

        print(f"\n PART I: N-gram Modeling and Perplexity \n")
        # print(f"\n ~~~~~~~~ EVALUATE {ngramtype.name} MODEL.~~~~~~~~ \n")
        n = ngramtype.value
        
        # Create a ngram model
        ngram_model = NgramModel(n)

        # Load training data from file
        train_corpus = ngram_model.load_data(train_file)
        
        # Build the vocabulary
        ngram_model.build_vocabulary(train_corpus)

        # Train the model on the training data
        ngram_model.train(train_corpus)

        # Load test data from file
        test_corpus = ngram_model.load_data(test_file)

        
        # Calculate average perplexity without smoothing
        average_perplexity_withoutsmoothing = ngram_model.calculate_average_perplexity(test_corpus, smoothing = 0)
        print(f"    Average Perplexity: {average_perplexity_withoutsmoothing}")

        # Calculate average perplexity with smoothing
        average_perplexity_withsmoothing = ngram_model.calculate_average_perplexity(test_corpus, smoothing = 1)
        print(f"    Average Perplexity: {average_perplexity_withsmoothing}")

        print(f"\n PART II: Text generation from ngram language models \n")
        # print(f"\n ~~~~~~~~~~ GENERATING TEXT ~~~~~~~~~~~~~~~~~~~~~~~\n ")
        
        # top_p_value = 0.9
        # for strategy in Strategy:
            

        #     # Generate the text
        #     generated_text = ngram_model.generate_text(max_length=100, strategy=strategy.value, top_p=top_p_value)
        
        #     # Save the generated text to a file for later review
        #     ngram_model.save_generated_text_to_file(generated_text, output_file_path, strategy, top_p_value, ngramtype.name)

        # print(f"Generated text saved to {output_file_path}")

    print("\n")

