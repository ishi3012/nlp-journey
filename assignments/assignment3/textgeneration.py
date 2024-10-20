"""
Text generation.
"""
import os
from typing import List, Dict, Tuple
from collections import Counter
import nltk
from enum import Enum, auto

# class InputFileType(Enum):
#     TRAIN = auto()
#     TEST  = auto()

def read_corpus(input_filename:str) -> List[str]:
    assert input_filename, "Input filename must be provided and cannot be empty."
    input_file_path = os.path.join(input_filename)
    print(f" Reading the input file from the location : {input_file_path}")   

    try:        
        with open(input_file_path, "r") as file:
            lines = file.readlines()
        return [line.strip() for line in lines]
    
    except FileNotFoundError:
        print(f"File {input_filename} not found in the InputFiles folder.")
    
def create_vocabulary(text: List[str])-> List[str]:    
    words = [word for line in text for word in line.split()]
    frequencies = nltk.FreqDist(words)
    vocabulary = {word:count for word,count in frequencies.items() if count >= 0}
    vocabulary["<unk>"] = 0
    vocabulary["STOP"] = 1
    # Ensure vocabulary has the correct size
    
    # assert len(vocabulary) == 26602, f"Vocabulary size should be 26,602 but is {len(vocabulary)}."
    return vocabulary

def process_tokens(text: List[str], vocabulary: Dict[str, int]) -> List[str]:
    """
    Process the tokens in the text and return processed tokens after doing below steps 
    1. Convert any tokens not in the vocabulary to '<unk>'.
    2. For each sentence, prepend '<START>' and append '<STOP>' to the sequence of tokens.
    """
    assert text, "The input text should not be empty."
    assert vocabulary, "The vocabulary dictionary should not be empty."

    processed_tokens = []
    
    # Replace OOV tokens
    for line in text:
        processed_tokens.append("<START>")
        for token in line.split():
            if token in vocabulary:
                processed_tokens.append(token)
            else:
                processed_tokens.append('<unk>')  
        processed_tokens.append("<STOP>")
    
    return processed_tokens

def build_ngram_model(tokens: List[str], n:int = 1):
    """
    Builds a n-gram model from the text provided. 
    """
    assert tokens, "The input tokens should not be empty."

    ngrams = list(nltk.ngrams(tokens, n))
    
    ngrams_tupled = [(ngram[:-1], ngram[-1]) for ngram in ngrams]

    cfd = nltk.ConditionalFreqDist(ngrams_tupled)

    return cfd

def calculate_ngram_probability(ngram: Tuple[Tuple[str, ...], str], cfd:nltk.ConditionalFreqDist, vocab_size: int, n: int = 2, smoothing:bool = False) -> float:
    context, word = ngram
    print(context, word)



def main(input_filename:str):
    text = read_corpus(input_filename)
    vocabulary = create_vocabulary(text)
    processed_tokens = process_tokens(text, vocabulary)
    cfd = build_ngram_model(processed_tokens,3)
    # for context in cfd:
    #     print(f"Context = {context}")
    #     freq_dist = cfd[context]
    #     for word, _ in freq_dist.items():
    #         print(f"   Probability of word {word} = {cfd[context].freq(word)}") 

    for context in cfd:
        calculate_ngram_probability(processed_tokens, cfd, len(vocabulary), 2, smoothing=False)

if __name__ == "__main__":
    main("assignments/assignment3/InputFiles/test_file.txt")


