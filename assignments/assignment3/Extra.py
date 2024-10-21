import nltk
from nltk.util import ngrams
from nltk.probability import ConditionalFreqDist
from typing import List, Tuple, Dict
import numpy as np
import math
from enum import Enum, auto
import random

start_label = '<START>'
end_label = '<STOP>'

# Read corpus and split into sentences
def read_corpus(input_filename: str) -> List[List[str]]:
    """Reads the input file and returns a list of tokenized sentences."""
    with open(input_filename, "r") as file:
        return [line.strip().split() for line in file.readlines()]

# Build vocabulary based on minimum frequency
def build_vocabulary(corpus: List[List[str]], min_freq: int = 3) -> Tuple[Dict[str, int], List[List[str]]]:
    """Builds a vocabulary from the corpus."""
    tokens = [token for sentence in corpus for token in sentence]
    frequencies = nltk.FreqDist(tokens)
    
    # Create the vocabulary: tokens occurring at least 'min_freq' times
    vocabulary = {token: i for i, (token, freq) in enumerate(frequencies.items()) if freq >= min_freq}
    
    # Add the special '<unk>' token to represent rare words
    # vocabulary['<unk>'] = len(vocabulary)
    vocabulary['<unk>'] = 0
    
    # Modify the corpus: Replace rare tokens with '<unk>'
    modified_corpus = [[token if token in vocabulary else '<unk>' for token in sentence] for sentence in corpus]
    
    # Add <STOP> at the end of each sentence
    for sentence in modified_corpus:
        sentence.append(end_label)
            
    return vocabulary, modified_corpus

# Generate n-grams based on the n value
def generate_ngrams(corpus: List[List[str]], n: int) -> List[Tuple[str]]:
    """Generate N-grams for the corpus."""
    return [ngram for sentence in corpus for ngram in ngrams(sentence, n, pad_left=True, left_pad_symbol=start_label)]

# Build N-gram model using ConditionalFreqDist
def build_ngram_model(ngrams_list: List[Tuple[str]]):
    """Build N-gram probability model using ConditionalFreqDist."""
    cfd = ConditionalFreqDist((ngram[:-1], ngram[-1]) for ngram in ngrams_list)
    return cfd

def calculate_perplexity(cfd: ConditionalFreqDist, test_corpus: List[List[str]], n: int) -> float:   
    """Calculate perplexity without smoothing."""
    total_perplexity_score = 0
    sentences = 0

    for sentence in test_corpus:
        perplexity_score = 0
        grams_count = 0
        sentence = [start_label] * (n - 1) + sentence  # Add padding for n-grams
        
        for i, word in enumerate(sentence):
            ngram_context = tuple(sentence[i - (n - 1):i])
            word = sentence[i]
            if word != end_label:                
                word_probability = cfd[ngram_context].freq(word)
                perplexity_score -= np.log(word_probability) if word_probability > 0 else 0
                grams_count += 1 
            elif word == end_label:       
                # grams_count -= 1   
                sentences += 1  
                total_perplexity_score += np.exp(perplexity_score / grams_count)          
                break

    avg_perplexity_score = total_perplexity_score / sentences if sentences > 0 else float('inf')   
    return avg_perplexity_score

def calculate_perplexity_laplace(cfd: ConditionalFreqDist, test_corpus: List[List[str]], n: int, vocab_size: int) -> float:
    """Calculate perplexity using Laplace smoothing."""
    total_perplexity_score = 0
    sentences = 0

    for sentence in test_corpus:
        perplexity_score = 0
        grams_count = 0
        sentence = [start_label] * (n - 1) + sentence 

        # Iterate over each word in the sentence
        for i in range(n - 1, len(sentence)):
            ngram_context = tuple(sentence[i - (n - 1):i])
            word = sentence[i]

            # If we encounter <STOP>, end the sentence processing
            if word == end_label:
                break

            # Get the frequency of the context and word
            context_count = sum(cfd[ngram_context].values())  # Total occurrences of the context
            word_count = cfd[ngram_context][word]  # Count of the specific word given the context

            # Apply Laplace smoothing
            word_probability = (word_count + 1) / (context_count + vocab_size)

            # Accumulate the log probability
            perplexity_score -= np.log(word_probability)
            grams_count += 1

        # After processing the sentence, calculate the sentence perplexity
        if grams_count > 0:
            total_perplexity_score += np.exp(perplexity_score / grams_count)
            sentences += 1

    # Final perplexity: average over all sentences
    avg_perplexity_score = total_perplexity_score / sentences if sentences > 0 else float('inf')
    return avg_perplexity_score

def generate_text(cfd: ConditionalFreqDist, n: int, strategy: str = "greedy", p: float = 0.9, max_length: int = 100) -> str:
    """
    Generates text using the given N-gram model with a specified strategy.
    """

    generated_text = []
    current_ngram = tuple([start_label] * (n - 1))  # Initial n-gram starting with '<START>'
    
    for _ in range(max_length):
        next_word_candidates = list(cfd[current_ngram].keys())
        if not next_word_candidates:
            break
        
        # Apply the chosen strategy
        if strategy == "greedy":
            
            next_word = cfd[current_ngram].max()

        elif strategy == "random":

            # Randomly sample the next word based on its frequency distribution
            total_count = sum(cfd[current_ngram].values())
            probabilities = [cfd[current_ngram].freq(word) for word in next_word_candidates]
            next_word = random.choices(next_word_candidates, probabilities)[0]

        elif strategy == "top_p":
            # Select from top-p words
            sorted_candidates = sorted(next_word_candidates, key=lambda word: cfd[current_ngram].freq(word), reverse=True)
            cumulative_prob = 0
            top_p_candidates = []
            for word in sorted_candidates:
                cumulative_prob += cfd[current_ngram].freq(word)
                top_p_candidates.append(word)
                if cumulative_prob >= p:
                    break
            # Randomly select from top-p candidates
            next_word = random.choice(top_p_candidates)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Append the generated word and update the context
        if next_word == end_label:
            break
        generated_text.append(next_word)
        
        # Update the n-gram context
        current_ngram = tuple(generated_text[-(n - 1):]) if n > 1 else ()
    
    return ' '.join(generated_text)


def unigram(train_corpus: List[List[str]], test_corpus: List[List[str]]):
    n = 1
    # Build vocabulary and process the training corpus
    vocabulary, processed_train_corpus = build_vocabulary(train_corpus, min_freq=3)
    vocab_size = len(vocabulary)
    train_ngrams = generate_ngrams(processed_train_corpus, n)
    cfd = build_ngram_model(train_ngrams)
    _, processed_test_corpus = build_vocabulary(test_corpus, min_freq=3)
    
    perplexity_no_smoothing = calculate_perplexity(cfd, processed_test_corpus, n)
    perplexity_laplace = calculate_perplexity_laplace(cfd, processed_test_corpus, n, vocab_size)
    
    return perplexity_no_smoothing, perplexity_laplace

# Function for Bigram model
def bigram(train_corpus: List[List[str]], test_corpus: List[List[str]]):
    n = 2
    # Build vocabulary and process the training corpus
    train_vocabulary, processed_train_corpus = build_vocabulary(train_corpus, min_freq=3)
    vocab_size = len(train_vocabulary)
    train_ngrams = generate_ngrams(processed_train_corpus, n)    
    cfd = build_ngram_model(train_ngrams)

    # Iterate over the contexts (conditions) in the ConditionalFreqDist
    for context in cfd.conditions():
        
        print(f"Context: {context}")
        if context[0] == '<START>':
            print(" Its a start. ")

        # For each context, retrieve the frequency distribution of words following that context
        freq_dist = cfd[context]
        

        # # Now iterate over the words in the frequency distribution and print their counts
        # for word in freq_dist:
        #     print(f"  Word: {word}, Count: {freq_dist[word]}")
        break

    # # process test corpus
    # _, processed_test_corpus = build_vocabulary(test_corpus, min_freq=3)
    
    # perplexity_no_smoothing = calculate_perplexity(cfd, processed_test_corpus, n)
    # perplexity_laplace = calculate_perplexity_laplace(cfd, processed_test_corpus, n, vocab_size)
    
    # print(f"~~~~~~~~~~ PART 1: Perplexity Calculations ~~~~~~~~~~~~~~~~~~~")
    # print(f"Bigram Perplexity without smoothing: {perplexity_no_smoothing}")
    # print(f"Bigram Perplexity with Laplace smoothing: {perplexity_laplace}")
    
    #  # Merge datasets for text generation
    # merged_corpus = train_corpus + test_corpus

    # # Build vocabulary from the merged corpus
    # merged_corpus_vocabulary, processed_merged_corpus = build_vocabulary(merged_corpus, min_freq=3)

    # # Generate bigrams and build the ConditionalFreqDist model
    # merged_bigram_ngrams = generate_ngrams(processed_merged_corpus, n=n)
    # merged_bigram_cfd = build_ngram_model(merged_bigram_ngrams)

    # print(f"~~~~~~~~~~ PART 2: Generate text ~~~~~~~~~~~~~~~~~~~\n")
    # # Generate text using bigram model
    # print(f"Generate text using greedy-choice")
    # # greedy_generated_text = generate_text(merged_bigram_cfd, n = n, strategy="greedy")
    # print(generate_text(merged_bigram_cfd, n = n, strategy="greedy"))

    # print("\nRandom sampling generation:")
    # # random_generated_text = generate_text(merged_bigram_cfd, n = n, strategy="random")
    # print(generate_text(merged_bigram_cfd, n = n, strategy="random"))

    # print("\nTop-p sampling generation (p=0.9):")
    # # topp_generated_text = generate_text(merged_bigram_cfd, n=2, strategy="top_p", p=0.9)
    # print(generate_text(merged_bigram_cfd, n=2, strategy="top_p", p=0.9))

# Function for Trigram model
def trigram(train_corpus: List[List[str]], test_corpus: List[List[str]]):
    n = 3
    # Build vocabulary and process the training corpus
    vocabulary, processed_train_corpus = build_vocabulary(train_corpus, min_freq=3)
    vocab_size = len(vocabulary)
    train_ngrams = generate_ngrams(processed_train_corpus, n)
    cfd = build_ngram_model(train_ngrams)
    _, processed_test_corpus = build_vocabulary(test_corpus, min_freq=3)
    
    perplexity_no_smoothing = calculate_perplexity(cfd, processed_test_corpus, n)
    perplexity_laplace = calculate_perplexity_laplace(cfd, processed_test_corpus, n, vocab_size)
    
    return perplexity_no_smoothing, perplexity_laplace


def main():
    train_file = "assignments/assignment3/InputFiles/1b_benchmark.train.tokens.txt"  
    test_file = "assignments/assignment3/InputFiles/1b_benchmark.test.tokens.txt"  
    
    # Read training and test corpus
    train_corpus = read_corpus(train_file)
    test_corpus = read_corpus(test_file)    


    # # Unigram
    # print(f"~~~~~~~~~~~ Unigram Execution ~~~~~~~~~~~~~~~~~")
    # unigram_perplexity, unigram_laplace = unigram(train_corpus, test_corpus)
    # print(f"Unigram Perplexity without smoothing: {unigram_perplexity}")
    # print(f"Unigram Perplexity with Laplace smoothing: {unigram_laplace}")

    # Bigram
    print(f"~~~~~~~~~~~ Bigram Execution ~~~~~~~~~~~~~~~~~")
    bigram(train_corpus, test_corpus)
    
    
    # # Trigram
    # print(f"~~~~~~~~~~~ Trigram Execution ~~~~~~~~~~~~~~~~~")
    # trigram_perplexity, trigram_laplace = trigram(train_corpus, test_corpus)
    # print(f"Trigram Perplexity without smoothing: {trigram_perplexity}")
    # print(f"Trigram Perplexity with Laplace smoothing: {trigram_laplace}")

if __name__ == "__main__":
    main()
