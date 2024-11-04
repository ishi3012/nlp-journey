import os
import re
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from datasets import Dataset, DatasetDict
import nltk
nltk.download('punkt')


class T5SummarizerModel:
    def __init__(self, model_name:str="google-t5/t5-small", task:str="summarize: "):
        self.model_name = model_name
        self.nlp_task = task
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.dataset = None
        self.tokenized_dataset = None
        self.data_dir = os.path.join(os.getcwd(), "data")
        self.output_dir = os.path.join(os.getcwd(), "results")
        
    def download_kaggle_dataset(self, dataset_name: str) -> str:
        """Downloads and extracts a Kaggle dataset, returning the CSV file path."""
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Check if a CSV file already exists in the directory
        csv_file = next((f for f in os.listdir(self.data_dir) if f.endswith(".csv")), None)
        if csv_file:
            print(f"Dataset already exists: {csv_file}")
            return os.path.join(self.data_dir, csv_file)
        
        # Download and unzip the dataset if not already downloaded
        os.system(f"kaggle datasets download -d {dataset_name} -p {self.data_dir}")
        os.system(f"unzip {self.data_dir}/{dataset_name.split('/')[-1]}.zip -d {self.data_dir}")
        
        # Return the CSV file path
        csv_file = next((f for f in os.listdir(self.data_dir) if f.endswith(".csv")), None)
        if csv_file:
            return os.path.join(self.data_dir, csv_file)
        
        raise FileNotFoundError("No CSV file found in the downloaded dataset.")

    def load_and_split_data(self, csv_filename: str) -> None:
        """Loads the dataset from the data directory and prepares train/val/test splits."""
        csv_path = os.path.join(self.data_dir, csv_filename)
    
        # Check if file exists
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"The file {csv_filename} does not exist in {self.data_dir}.")
        
        # Load data
        df = pd.read_csv(csv_path)
        
        if not {"essay", "description"}.issubset(df.columns):
            raise ValueError("CSV file must contain 'essay' and 'description' columns.")
        
        # Split dataset and convert to DatasetDict
        self.dataset = DatasetDict({
            "train": Dataset.from_pandas(df.iloc[:1600][["essay", "description"]]),
            "validation": Dataset.from_pandas(df.iloc[1600:1800][["essay", "description"]]),
            "test": Dataset.from_pandas(df.iloc[1800:][["essay", "description"]]),
        })

        
        if not self.dataset:
            print("Dataset is not loaded.")
            return
        else:
            for split, data in self.dataset.items():
                print(f"{split.capitalize()} set size: {len(data)} samples. Shape = {data.shape}")

    def clean_text(self, text:str):
        sentences = nltk.sent_tokenize(text.strip())
        sentences_cleaned = [s for sent in sentences for s in sent.split("\n")]

        # remove punctuations
        sentences_cleaned = [re.sub(r'[^\w\s]', '', s) for s in sentences_cleaned]

        # remove non-Acsii chanracters
        sentences_cleaned = [re.sub(r'[^\x00-\x7F]+', '', s) for s in sentences_cleaned]

        text_cleaned = "\n".join(sentences_cleaned)
        return text_cleaned
    
    def preprocess_data(self, examples):
        







def main():
    # Load dataset 
    summarizer = T5SummarizerModel()
    dataset_filename = summarizer.download_kaggle_dataset("mannacharya/aeon-essays-dataset").split('/')[-1]
    summarizer.load_and_split_data(dataset_filename)
    

    # # Tokenize the dataset
    # summarizer.tokenize_data()

    # # Train the model
    # summarizer.train(epochs=3)

    # # Evaluate the model
    # summarizer.evaluate()


if __name__ == "__main__":
    main()
