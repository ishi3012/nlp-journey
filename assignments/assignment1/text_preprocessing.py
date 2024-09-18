import pandas as pd 
import numpy as np
import re
from typing import IO, List, Tuple, Dict, Any
import os
import nltk
import sys

# download nltk.punkt tokenizer model.
nltk.download('all', quiet=True)

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
    updated_tokens = tokens
    return updated_tokens



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
    sentenceTokens = nltk.sent_tokenize(content)
    return content, wordtokens, sentenceTokens




def calculate_statistics(contect: str, wordtokens: List[str], sentenceTokens: List[str] ) -> Dict[str, Any]:
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
    results['NumberOfParagraphs'] = _count_paragraphs(contect)

    # calculate number of sentences
    results['NumberOfSentences'] = len(sentenceTokens)

    # calculate number of tokens
    results['NumberOfTokens'] = len(wordtokens)

    # calculate number of unique tokens
    uniqueTokens = set(wordtokens)
    results['NumberOfUniqueTokens'] = len(uniqueTokens)

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
    except IOError as e:
        print(f"Unable to write to the file at path : {outputfilepath}")

    except OSError as e:
        print(f"Unable to work with the file at path : {outputfilepath}")
    
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def main():
    directory_name = os.path.dirname(__file__)
    inputfilename = 'sample_2024.txt'
    outputfilename = 'output1.txt'
    
    content, wordtokens, sentenceTokens = preprocess_text(os.path.join(directory_name,inputfilename))
    results = calculate_statistics(content, wordtokens, sentenceTokens)
    # print(wordtokens)
    
    # # Write statistical results to file 
    # write_statistics_to_file(results,os.path.join(directory_name, outputfilename))

if __name__ == "__main__":
    main()



