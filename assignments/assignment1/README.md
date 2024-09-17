# **Assignment #1: Text Preprocessing** 📄

## **Overview**

In this assignment, the task is to develop a Python program that processes a plain text document and computes various statistics. The program will output the following:

- Total number of paragraphs.
- Total number of sentences.
- Total number of words/tokens after applying specified tokenization.
- Total number of unique words/tokens.
- Ranked frequency counts of word/token types, sorted in descending order.

The implementation must be done from scratch, using Python’s native string functions and some utility libraries (e.g., `re`, `numpy`, `pandas`). NLP-specific libraries are not allowed, with exceptions for `nltk.sent_tokenize()` and `nltk.word_tokenize()`.

## **Objectives**

- Compute text statistics (paragraphs, sentences, tokens, unique tokens).
- Apply specified tokenization rules.
- Generate frequency counts of word types.
- Handle special cases and contractions in text processing.
- Output results to specified files.

## **Implementation**

### **Tokenization and Preprocessing**

1. **Tokenization Steps**:
   - Use `nltk.word_tokenize()` for initial tokenization.
   - Apply custom rules for handling punctuation and contractions.

2. **Rules**:
   - Separate leading and trailing punctuation.
   - Expand contractions as specified.
   - Normalize text to lowercase.

### **Tasks**

#### **Task 1: Processing "sample_2024.txt"**

1. **Input**: `sample_2024.txt` - Sentences are on separate lines.
2. **Output**: `output1.txt` - Include statistics and ranked frequency counts of word types.

#### **Task 2: Processing "war-and-peace.txt"**

1. **Input**: `war-and-peace.txt` - Text file formatted with fixed column-width.
2. **Output**: `output2.txt` - Include statistics and ranked frequency counts.
3. **Additional**: Generate a chart comparing the (log base e) frequencies of words against their ranks.

## **Code Examples**

Here are some snippets of the implementation:

```python
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import re
from collections import Counter

def preprocess_text(text):
    # Implement text preprocessing steps
    pass

def calculate_statistics(tokens):
    # Calculate and return statistics
    pass

def main():
    # Load data, process text, and generate outputs
    pass

if __name__ == "__main__":
    main()
```
## **How to Run the Code**

1. **Clone this repository:**
    ```bash
    git clone https://github.com/ishi3012/nlp-journey
    ```

2. **Navigate to the assignment folder:**
    ```bash
    cd assignments/assignment1
    ```

3. **Install dependencies:**
   You can find the required dependencies in the `requirements.txt` file. To install them:
    ```bash
    pip install -r requirements.txt
    ```

4. **Download NLTK Tokenizer Data:**
   This project uses NLTK’s `sent_tokenize()` and `word_tokenize()`. Before running the code, make sure to download the necessary tokenizer data. Run the following commands in Python:
    ```python
    import nltk
    nltk.download('punkt')
    ```

5. **Run the Python script:**
    ```bash
    python text_preprocessing.py
    ```
## **Results**
- Task 1 Output: The file output1.txt includes counts and ranked frequencies for the provided text file.
- Task 2 Output: The file output2.txt includes counts and ranked frequencies for the "War and Peace" text, along with a chart showing word frequency distribution.

## **Deliverables**

1. **Source Code:**
    - Python notebook file (.ipynb), plus its HTML or PDF version.
    - Python script (text_preprocessing.py).
2. **Output Files:**
    - output1.txt for Task 1.
    - output2.txt for Task 2.
3. **Write-Up Document:**
    - Document in .docx or .pdf format
    - Contents of Write-Up:
        - Description of development environment (platform, IDE, libraries).
        - Chart generated for Task 2 and comments on frequency distribution.
        - Reflections, including output accuracy, difficulties, and improvements.
## **License**
This repository is licensed under the MIT License. See the [LICENSE](./LICENSE) file for more details.

## **Contact**
Feel free to reach out if you have any questions or suggestions!
- [Email](ishishiv3012@gmail.com)

