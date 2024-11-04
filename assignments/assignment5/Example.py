import os
import pandas as pd
import torch
import numpy as np
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from datasets import Dataset, DatasetDict
from typing import Dict
os.environ["WANDB_DISABLED"] = "true"

class T5SummarizerModel:
    def __init__(self, model_name: str = "google-t5/t5-small", task: str = "summarize: "):
        self.model_name = model_name
        self.task = task
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize tokenizer, model, and metrics
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.metrics = {
            "rouge": evaluate.load("rouge"),
            "bertscore": evaluate.load("bertscore"),
        }
        # Set paths and dataset attributes
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
    
    def display_dataset_sample(self) -> None:
        """Displays a sample of the loaded dataset."""
        if not self.dataset:
            print("Dataset is not loaded.")
            return

        for split, data in self.dataset.items():
            print(f"{split.capitalize()} set size: {len(data)} samples")

    def preprocess_data(self, examples):
      """Tokenizes the dataset examples."""
      model_inputs = self.tokenizer(examples["essay"], max_length=256, truncation=True)
      labels = self.tokenizer(examples["description"], max_length=64, truncation=True).input_ids
      
      # Ensure all tokens in labels are within valid range
      model_inputs["labels"] = [
          [token if 0 <= token < self.tokenizer.vocab_size else self.tokenizer.pad_token_id for token in label]
          for label in labels
      ]
      return model_inputs

    def tokenize_data(self) -> None:
      """Applies tokenization to the dataset."""
      if self.dataset is None:
          raise ValueError("Dataset not loaded. Run `load_and_split_data` first.")
      
      # Tokenize the dataset
      self.tokenized_dataset = self.dataset.map(self.preprocess_data, batched=True)

    def compute_metrics(self, eval_pred) -> Dict[str, float]:
        """Computes ROUGE, BERTScore, and generation length metrics."""
        predictions, labels = eval_pred
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)

        # Make sure `labels` are of the correct format for decoding
        decoded_labels = [
            self.tokenizer.decode(label_ids, skip_special_tokens=True)
            for label_ids in labels
        ]
        # Compute ROUGE scores
        rouge_result = self.metrics["rouge"].compute(predictions=decoded_preds, references=decoded_labels)
        rouge_scores = {k: round(v.mid.fmeasure * 100, 4) for k, v in rouge_result.items() if k in ["rouge1", "rouge2", "rougeL"]}

        # Compute BERTScore
        bertscore_result = self.metrics["bertscore"].compute(predictions=decoded_preds, references=decoded_labels, lang="en")
        bertscore_avg = round(np.mean(bertscore_result["f1"]) * 100, 4)

        # Calculate average generation length
        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in predictions]
        gen_len = round(np.mean(prediction_lens), 2)

        return {**rouge_scores, "bertscore_f1": bertscore_avg, "gen_len": gen_len}

    def train(self, epochs, batch_size=4, learning_rate=2e-5, max_new_tokens=50) -> None:
      """Trains the model."""
      data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.model)
      
      training_args = Seq2SeqTrainingArguments(
          output_dir=self.output_dir,
          eval_strategy="epoch",
          learning_rate=learning_rate,
          per_device_train_batch_size=batch_size,
          per_device_eval_batch_size=batch_size,
          weight_decay=0.01,
          save_total_limit=3,
          num_train_epochs=4,
          predict_with_generate=True,
          fp16=True, 
      )
      trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_dataset["train"],
            eval_dataset=self.tokenized_dataset["validation"],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
          
        )
      trainer.train()

    def evaluate(self, max_new_tokens=50) -> Dict[str, float]:
      """Evaluates the model on the test set."""
      gen_kwargs = {"max_new_tokens": max_new_tokens}

      trainer = Seq2SeqTrainer(
          model=self.model,
          args=Seq2SeqTrainingArguments(output_dir=self.output_dir, report_to="none"),
          eval_dataset=self.tokenized_dataset["test"],
          tokenizer=self.tokenizer,
          compute_metrics=self.compute_metrics,
          generation_max_length=max_new_tokens,
      )      
      results = trainer.evaluate()
      print("Evaluation Results:\n", results)
      return results

def main():
    # Load dataset 
    summarizer = T5SummarizerModel()
    dataset_filename = summarizer.download_kaggle_dataset("mannacharya/aeon-essays-dataset").split('/')[-1]
    summarizer.load_and_prepare_data(dataset_filename)
    summarizer.display_dataset_sample()

    # Tokenize the dataset
    summarizer.tokenize_data()

    # Train the model
    summarizer.train(epochs=3)

    # Evaluate the model
    summarizer.evaluate()


if __name__ == "__main__":
    main()
