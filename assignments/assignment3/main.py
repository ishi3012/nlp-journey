import assignments.assignment3.Ngram_Model as m
import os

ngram = m.NgramModel(n = 2, min_freq = 3)

def read_corpus(input_filename: str) -> str:
    """Reads the input file and returns a list of tokenized sentences."""
    with open(input_filename, "r") as file:
        return file.read()
    

def main():

    
    train_file = "assignments/assignment3/InputFiles/1b_benchmark.train.tokens.txt"  
        
    # Read training and test corpus
    train_corpus = read_corpus(train_file)
       
    ngram._tokenize(text = train_corpus)

if __name__ == "__main__":
    main()
