import os
import re
import pandas as pd
import nltk
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    AutoModelForCausalLM
)
from datasets import Dataset, DatasetDict

from evaluate import load
nltk.download('punkt')
nltk.download('punkt_tab')


class T5SummarizerModel:
    def __init__(self, model_name: str = "google-t5/t5-small", prefix: str = "summarize: "):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        self.model_name = model_name
        self.prefix = prefix
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)  
        self.dataset = None
        self.tokenized_dataset = None
        self.data_dir = os.path.join(os.getcwd(), "data")
        self.output_dir = os.path.join(os.getcwd(), "results")
        self.log_dir = os.path.join(os.getcwd(), "logs")
        self.rouge_metric = load("rouge")
        self.bertscore_metric = load("bertscore")

        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.model.resize_token_embeddings(len(self.tokenizer))

        # Initialize GPT-2 for perplexity calculation
        self.gpt2_model = AutoModelForCausalLM.from_pretrained("gpt2").to(self.device)  
        self.gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")

        # Set padding token for gpt2_tokenizer if it doesn't exist
        if self.gpt2_tokenizer.pad_token is None:
            if self.gpt2_tokenizer.eos_token:
                self.gpt2_tokenizer.pad_token = self.gpt2_tokenizer.eos_token
            else:
                self.gpt2_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        # Resize embeddings for GPT-2 tokenizer if we added a new token
        self.gpt2_model.resize_token_embeddings(len(self.gpt2_tokenizer))

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

        # Filter dataset based on text and title length
        self.dataset = self.dataset.filter(lambda example: (len(example["essay"]) >= 512) and (len(example["description"]) >= 64))


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

    # def preprocess_data(self, examples):
    #     max_input_length = 512
    #     max_target_length = 64
    #     text_cleaned = [self.clean_text(text) for text in examples["essay"]]
    #     inputs = [self.prefix + text for text in text_cleaned]
    #     model_inputs = self.tokenizer(inputs, max_length=max_input_length, truncation=True)

    #     with self.tokenizer.as_target_tokenizer():
    #         labels = self.tokenizer(examples["description"], max_length=max_target_length, truncation=True)

    #     model_inputs["labels"] = labels["input_ids"]
    #     return model_inputs

    def preprocess_data(self, examples):
        max_input_length = 512
        max_target_length = 64
        text_cleaned = [self.clean_text(text) for text in examples["essay"]]
        inputs = [self.prefix + text for text in text_cleaned]
        model_inputs = self.tokenizer(inputs, max_length=max_input_length, truncation=True, padding="max_length")

        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(examples["description"], max_length=max_target_length, truncation=True, padding="max_length")

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def tokenize_dataset(self):
        """Tokenizes the entire dataset."""
        self.tokenized_dataset = self.dataset.map(self.preprocess_data, batched=True)

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred

        # Ensure `predictions` and `labels` are PyTorch tensors
        if isinstance(predictions, np.ndarray):
            predictions = torch.tensor(predictions).to(self.device)
        if isinstance(labels, np.ndarray):
            labels = torch.tensor(labels).to(self.device)

        # Generate predictions with controlled length
        predictions = self.model.generate(
            input_ids=predictions,
            max_length=64,
            attention_mask=(labels != self.tokenizer.pad_token_id).to(self.device)
        )

        # Decode predictions and labels
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = ["\n".join(pred.strip().split()) for pred in decoded_preds]
        decoded_labels = ["\n".join(label.strip().split()) for label in decoded_labels]

        # Compute ROUGE scores
        rouge_results = self.rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        rouge_scores = {key: value * 100 for key, value in rouge_results.items()}

        # Compute BERTScore
        bertscore_results = self.bertscore_metric.compute(predictions=decoded_preds, references=decoded_labels, lang="en")
        bert_precision = np.mean(bertscore_results["precision"])
        bert_recall = np.mean(bertscore_results["recall"])
        bert_f1 = np.mean(bertscore_results["f1"])

        # Calculate Perplexity with GPT-2
        if decoded_preds:  # Check if there are any predictions
            encodings = self.gpt2_tokenizer(decoded_preds, return_tensors="pt", padding=True, truncation=True).to(self.device)
            max_length = self.gpt2_model.config.n_positions
            stride = max_length // 2
            nlls = []
            
            for i in range(0, encodings.input_ids.size(1), stride):
                begin_loc = i
                end_loc = min(i + max_length, encodings.input_ids.size(1))
                trg_len = end_loc - begin_loc
                input_ids = encodings.input_ids[:, begin_loc:end_loc].to(self.device)
                target_ids = input_ids.clone()
                target_ids[:, :-trg_len] = -100

                with torch.no_grad():
                    outputs = self.gpt2_model(input_ids, labels=target_ids)
                    neg_log_likelihood = outputs.loss * trg_len
                    nlls.append(neg_log_likelihood)

            # Calculate mean perplexity if nlls is not empty
            if nlls:
                mean_perplexity = torch.exp(torch.stack(nlls).sum() / end_loc).item()
            else:
                mean_perplexity = float('inf')  # Assign a default value if nlls is empty
        else:
            mean_perplexity = float('inf')  # Assign a default value if decoded_preds is empty

        metrics = {
            "rouge1": rouge_scores.get("rouge1", 0),
            "rouge2": rouge_scores.get("rouge2", 0),
            "bert_precision": bert_precision,
            "bert_recall": bert_recall,
            "bert_f1": bert_f1,
            "mean_perplexity": mean_perplexity
        }

        return metrics


    
    def train(self, epochs:int = 3):
        """Trains the model using the training and validation datasets."""

        # Set up training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.output_dir,           # Directory to save the model
            eval_strategy="epoch",          # Evaluate on validation set at the end of each epoch
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            learning_rate=2e-5,
            weight_decay=0.01,
            num_train_epochs=1,
            predict_with_generate=True,           # Use generate for evaluation
            logging_dir=self.log_dir,
            logging_strategy="epoch",             # Log info at the end of each epoch
            save_strategy="epoch",                # Save the model at the end of each epoch
            load_best_model_at_end=True,
            report_to=None
            # max_new_tokens=64
        )

        # Define data collator
        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)

        # Initialize the Seq2SeqTrainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_dataset["train"],
            eval_dataset=self.tokenized_dataset["validation"],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics
        )

        # Train the model
        print("~~~~~~~~~~~~~~~~~~~~~ Starting training...\n")
        trainer.train()

        # Evaluate the model on the validation set
        print("~~~~~~~~~~~~~~~~~~~~~ Evaluating model...\n")
        eval_results = trainer.evaluate()

        # Report the results
        print("~~~~~~~~~~~~~~~~~~~~~ Evaluation results:", eval_results)

def main():
    # Load dataset 
    summarizer = T5SummarizerModel()
    dataset_filename = summarizer.download_kaggle_dataset("mannacharya/aeon-essays-dataset").split('/')[-1]
    summarizer.load_and_split_data(dataset_filename)
    print("~~~~~~~~~~~~~~~~~~~~~ Dataset is loaded and ready for preprocessing.\n")

    # Preprocess and tokenize Dataset
    summarizer.tokenize_dataset()
    print("~~~~~~~~~~~~~~~~~~~~~ Dataset tokenized and ready for training.\n")

    summarizer.train(epochs=3)


if __name__ == "__main__":
    main()
