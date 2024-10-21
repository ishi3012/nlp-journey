import nltk
from nltk.util import ngrams
from nltk.data import find
from nltk.probability import ConditionalFreqDist
from typing import List, Dict


start_label = '<START>'
end_label = '<STOP>'

class NgramModel:

    def __init__(self, n:int = 1, min_freq:int = 3) -> None:
        """
        Initialize the NgramModel class.
        
        :param n: The 'n' in N-gram (e.g., n=1 for unigram, n=2 for bigram, etc.)
        :param min_count: Minimum frequency for a token to be included in the vocabulary.
        """

        self.n = n
        self.min_freq = min_freq
        self.vocabulary: Dict[str, int] = {}
        self.cfd: ConditionalFreqDist = None

        try:
            find("tokenizer/punkt")
        except LookupError:                         
            nltk.download('punkt')
            nltk.download('punkt_tab')  

    def _tokenize(self, text:str) -> List[List[str]]:
        """
        Tokenizes the input text into sentences and words, adds <START> and <STOP> markers.

        :param text: The input text to tokenize.
        :return: A list of tokenized sentences where each sentence is a list of words.
        """
        
        assert text, "ERROR: The input text is empty."

        print(f"Tokenizing the text...\n")
        
        sentences = nltk.sent_tokenize(text)        
        tokens= [nltk.word_tokenize(sentence) for sentence in sentences]
        
        return tokens

        
        

        

    
