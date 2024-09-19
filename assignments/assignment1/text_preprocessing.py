import pandas as pd 
import numpy as np
import re
from typing import IO, List, Tuple, Dict, Any
import os
import nltk
import sys
import string
from collections import Counter
import math
import matplotlib.pyplot as plt

# download nltk.punkt tokenizer model.
nltk.download('punkt', quiet=True)

def _count_paragraphs(text: str) -> int:
    """
    The method returns number of paragraphs in the input text. 

    ### Parameters:
    - ** text (str) : A string containing content of the input file.

    ### Returns:
    - **int       :  Number of paragraphs in the input text.
    """
    text = text.replace("\r\n", "\n")
    paragraphs = re.split(r'\n\s*\n', text)
    
    return len(paragraphs)

def _handle_contractions(tokens: List[str]) -> List[str]:
    """
    Processes the tokens to handle contractions and expands them to their root forms or lemmas.

    This function applies custom rules to deal with specific English contractions in a way that expands them
    into their component words. The following contractions and special cases are handled:

    
    ### Parameters:
    - **tokens (List[str])**: A list of word tokens (strings) that may contain contractions.

    ### Returns:
    - **List[str]**: A new list of tokens where contractions are expanded based on the specified rules.
    """
    
    if tokens:
        for idx, token in enumerate(tokens):
            
            # Handle - Negative contractions
            if token == "n't":
                if tokens[idx-1] == "ca":
                    tokens[idx-1] = "can"
                elif tokens[idx-1] == "wo":
                    tokens[idx-1] = "will"
                elif tokens[idx-1] == "sha":
                    tokens[idx-1] = "shall"                
                tokens[idx] = "not"
            elif token.endswith("n't"):
                tokens[idx] = re.sub(r"n't$",'',token)
                tokens.insert(idx+1,"not")

            # Handle - Positive contractions
            if token == "'s":
                if tokens[idx-1] == "let":
                    tokens[idx] = "us"
                else:
                    tokens[idx] = "is"

            elif token == "'m":
                tokens[idx] = "am"

            elif token == "'re":
                tokens[idx] = "are"
            
            elif token == "'d":
                tokens[idx] = "would"

            elif token == "'ve":
                tokens[idx] = "have"
            
            elif token == "'ll":
                tokens[idx] = "will"
    return tokens

def _handle_punctuations(tokens: List[str]) -> List[str]:
    """
    The function processes a list of tokens to handle punctuations as follows:

    - Leading and trailing punctuation is separated into individual tokens.
    - Internal punctuation within tokens is not altered.

    ### Parameters:
    - **tokens (List[str]): A list of tokenized strings, which may include
                            punctuation.
    ### Returns:
    - **List[str]: A new list of tokens with leading and trailing punctuation
                   separated, but with internal punctuation preserved.
    """
    final_tokens = []

    # Regular expressions to match leading and trailing punctuation
    leading_punctuation = re.compile(r'^[^\w\s]+')
    trailing_punctuation = re.compile(r'[^\w\s]+$')

    for token in tokens:
        # Check if token is purely punctuation
        if re.fullmatch(r'[^\w\s]+', token):
            final_tokens.append(token)
            continue

        
        leading_punctuation_match = leading_punctuation.match(token)
        trailing_punctuation_match = trailing_punctuation.search(token)

        if leading_punctuation_match:            
            final_tokens.extend(list(leading_punctuation_match.group()))
        
        # Extract core word without leading/trailing punctuation
        word_start = len(leading_punctuation_match.group()) if leading_punctuation_match else 0
        word_end = -len(trailing_punctuation_match.group()) if trailing_punctuation_match else len(token)
        word = token[word_start:word_end]

        if word:
            final_tokens.append(word)
        
        if trailing_punctuation_match:
            # Add trailing punctuation as separate tokens
            final_tokens.extend(list(trailing_punctuation_match.group()))

    return final_tokens


def preprocess_text(textfilepath: str) -> Tuple[str, List[str], List[str]]:
    """
    Preprocess the content of the input text file and returns the tokens to create the vocabulary. 

    ### Parameters:
    - **textfilepath (str) : Path to the file to be processed.

    ### Returns:
    - **Tuple[str, List[str], List[str]] : A list of tokens extracted from the input text file.  

    """
    # Read the file 
    file = open(textfilepath, 'r', encoding = 'utf-8')
    content = file.read()
    content = content.lower()

    wordtokens = nltk.word_tokenize(content)    
    wordtokens = _handle_contractions(wordtokens)
    wordtokens = _handle_punctuations(wordtokens)
    sentenceTokens = nltk.sent_tokenize(content)
    return content, wordtokens, sentenceTokens

def calculate_statistics(content: str, wordtokens: List[str], sentenceTokens: List[str] ) -> Dict[str, Any]:
    """
    Calculates various text statistics based on the input text content and tokens.

    This function takes the raw text content, a list of word tokens, and a list of sentence tokens, and computes
    the following statistics:
    
    - **Total number of paragraphs**
    - **Total number of sentences** 
    - **Total number of words/tokens** 
    - **Total number of unique words/tokens**
    
    ### Parameters:
    - **content (str)**: The raw text content to be analyzed for paragraph counting.
    - **wordtokens (List[str])**: A list of word tokens obtained after tokenization.
    - **sentenceTokens (List[str])**: A list of sentence tokens obtained after sentence tokenization.

    ### Returns:
    - **Dict[str, Any]**: A dictionary containing the following keys:
        - `'NumberOfParagraphs'`: Total number of paragraphs in the text.
        - `'NumberOfSentences'`: Total number of sentences in the text.
        - `'NumberOfTokens'`: Total number of word tokens in the text.
        - `'NumberOfUniqueTokens'`: Total number of unique word tokens in the text.
    """
    results = {}

    # Calculate number of paragraphs
    results['NumberOfParagraphs'] = _count_paragraphs(content)

    # calculate number of sentences
    results['NumberOfSentences'] = len(sentenceTokens)

    # calculate number of tokens
    results['NumberOfTokens'] = len(wordtokens)

    # calculate number of unique tokens
    uniqueTokens = set(wordtokens)
    results['NumberOfUniqueTokens'] = len(uniqueTokens)

    results['FrequenciesOfTokens'] = Counter(wordtokens)

      
    return results

def write_statistics_to_file(statistics: Dict[str, Any], outputfilepath: str) -> None:
    """
    The function writes the statistical calculations of the input text file to output text file. 
    
    ### Parameters:
    - **Dict[str, Any]    : A dictionary containing the statistical calculation values. 
    - **outputfilepath    : A string the contains path of the file where the statistics will be written. 
    """
    try:
        file = open(outputfilepath, 'w', encoding = 'utf-8')
        file.write(f"# of paragraphs = {statistics['NumberOfParagraphs']}\n")
        file.write(f"# of sentences = {statistics['NumberOfSentences']}\n")
        file.write(f"# of tokens = {statistics['NumberOfTokens']}\n")
        file.write(f"# of unique tokens = {statistics['NumberOfUniqueTokens']}\n")
        file.write(f"====================================\n")
        
        ordered_tokens = sorted(statistics['FrequenciesOfTokens'].items(), key=lambda item:(-item[1], item[0]))
        for idx, (token, count) in enumerate(ordered_tokens, start = 1):
            file.write(f"{idx}: {token} {count}\n")

    except IOError as e:
        print(f"Unable to write to the file at path : {outputfilepath}")

    except OSError as e:
        print(f"Unable to work with the file at path : {outputfilepath}")
    
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def zipf_plot(statistics: Dict[str, Any]) -> None:

    log_rank = []
    log_frequency = []

    ordered_tokens = sorted(statistics['FrequenciesOfTokens'].items(), key=lambda item:(-item[1], item[0]))
    for idx, (token, count) in enumerate(ordered_tokens, start = 1):
        if idx != 0:
            log_rank.append(math.log(idx))
        
        if count != 0:
            log_frequency.append(math.log(count))
    
    plt.plot(log_rank, log_frequency)


def main():
    directory_name = os.path.dirname(__file__)

    print(f"--- Start : Task 1 ---")
    inputfilename = 'sample_2024.txt'
    outputfilename = 'output1.txt'
    
    content, wordtokens, sentenceTokens = preprocess_text(os.path.join(directory_name,inputfilename))
    results = calculate_statistics(content, wordtokens, sentenceTokens)
    # Write statistical results to file 
    write_statistics_to_file(results,os.path.join(directory_name, outputfilename))

    print(f"\t The result is written in the output1.txt file, which is at location : {os.path.join(directory_name, outputfilename)} ")
    
    zipf_plot(results)
    
    print(f"--- Task 1 completed ---")


    # print(f"--- Start : Task 2 ---")

    # inputfilename = 'war-and-peace.txt'
    # outputfilename = 'output2.txt'
    
    # content, wordtokens, sentenceTokens = preprocess_text(os.path.join(directory_name,inputfilename))
    # results = calculate_statistics(content, wordtokens, sentenceTokens)
    # # Write statistical results to file 
    # write_statistics_to_file(results,os.path.join(directory_name, outputfilename))
    
    # print(f"\t The result is written in the output1.txt file, which is at location : {os.path.join(directory_name, outputfilename)} ")
    # print(f"--- Task 2 completed ---")

if __name__ == "__main__":
    main()



