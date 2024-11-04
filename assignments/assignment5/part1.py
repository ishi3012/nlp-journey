from datasets import Dataset, load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from evaluate import load
from typing import Dict, Any, List

class T5Summarizer:
    def __init__(self, dataset: Dataset, model_name: str, task: str = "summarize: "):
        self.dataset = dataset
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.prefix = task
        self.rouge = load("rouge")
        self.bertscore = load("bertscore")
        self.perplexity_metric = load("perplexity", module_type="metric")
        self.load_model()

    def load_model(self):
        """Loads the T5 model and tokenizer based on the provided model name."""
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model.eval()

    def generate_summary(self, text: str, max_new_tokens: int = 20) -> str:
        """Generates a summary for the given text using the loaded model and tokenizer."""
        inputs = self.tokenizer(self.prefix + text, return_tensors="pt", truncation=True)
        summary_ids = self.model.generate(
            inputs["input_ids"],
            max_new_tokens=max_new_tokens,
            early_stopping=True
        )
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

    def compute_rouge(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Computes ROUGE scores for predictions and references."""
        rouge_scores = self.rouge.compute(predictions=predictions, references=references)
        return {"rouge1": rouge_scores["rouge1"], "rouge2": rouge_scores["rouge2"]}
    
    def compute_bertscore(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Computes BERTScore for predictions and references, averaging precision, recall, and F1."""
        bert_scores = self.bertscore.compute(predictions=predictions, references=references, lang="en")
        return {
            "precision": sum(bert_scores["precision"]) / len(bert_scores["precision"]),
            "recall": sum(bert_scores["recall"]) / len(bert_scores["recall"]),
            "f1": sum(bert_scores["f1"]) / len(bert_scores["f1"])
        }
    
    def compute_perplexity(self, predictions: List[str]) -> float:
        """Computes the average perplexity of generated summaries using GPT-2 as the reference model."""
        results = self.perplexity_metric.compute(model_id="gpt2", predictions=predictions)
        return results["mean_perplexity"]
    
    def evaluate(self, max_new_tokens: int = 20) -> Dict[str, Any]:
        """
        Generates summaries for the dataset and evaluates them
        using ROUGE, BERTScore, and Perplexity metrics.
        """
        references = [example["highlights"] for example in self.dataset]
        predictions = [
            self.generate_summary(example["article"], max_new_tokens=max_new_tokens)
            for example in self.dataset
        ]

        # Compute metrics
        rouge_scores = self.compute_rouge(predictions, references)
        bert_scores = self.compute_bertscore(predictions, references)
        mean_perplexity = self.compute_perplexity(predictions)

        # Compile evaluation results
        evaluation_results = {
            "rouge1": rouge_scores["rouge1"],
            "rouge2": rouge_scores["rouge2"],
            "mean_perplexity": mean_perplexity,
            "precision": bert_scores["precision"],
            "recall": bert_scores["recall"],
            "f1": bert_scores["f1"]
        }
        return evaluation_results

# Factory function to create the appropriate summarizer
def create_summarizer(model_type: str, dataset: Dataset) -> T5Summarizer:
    """
    Factory function to create a T5Summarizer with the specified model type.

    Parameters:
    - model_type (str): The type of model to load, e.g., "general" or "fine_tuned".
    - dataset (Dataset): The dataset to use for evaluation.

    Returns:
    - T5Summarizer: An instance of T5Summarizer initialized with the correct model.
    """
    if model_type == "general":
        model_name = "google-t5/t5-small"
    elif model_type == "fine_tuned":
        model_name = "ubikpt/t5-small-finetuned-cnn"
    else:
        raise ValueError("Unsupported model type. Choose 'general' or 'fine_tuned'.")

    return T5Summarizer(dataset=dataset, model_name=model_name)

def main():
    # Load dataset 
    dataset_name = "abisee/cnn_dailymail"
    dataset_config = "1.0.0"
    split_test_size = "3%"
    ds = load_dataset(dataset_name, dataset_config, split=f'test[:{split_test_size}]')
    print(f" Test split ({str(split_test_size)}) records : {len(ds)}\n")

    # Initialize and evaluate the general summarization model
    print("~~~~~~~ Evaluate google-t5/t5-small (max_new_tokens = 20) ~~~~~~~~~~~~~\n")
    general_summarizer = create_summarizer("general", ds)
    general_results = general_summarizer.evaluate(max_new_tokens = 20)
    print("General Model Evaluation Results: \n")

    for key, result in general_results.items():
        print(f"{key}: {result}\n")

    # Initialize and evaluate the general summarization model
    print("~~~~~~~ Evaluate google-t5/t5-small (max_new_tokens = 100) ~~~~~~~~~~~~~\n")
    general_summarizer = create_summarizer("general", ds)
    general_results = general_summarizer.evaluate(max_new_tokens = 100)
    print("General Model Evaluation Results: \n")

    for key, result in general_results.items():
        print(f"{key}: {result}\n")

    # Initialize and evaluate the fine-tuned summarization model
    print("\n~~~~~~~ Evaluate ubikpt/t5-small-finetuned-cnn (max_new_tokens = 100) ~~~~~~~~~~~~~\n")
    fine_tuned_summarizer = create_summarizer("fine_tuned", ds)
    fine_tuned_results = fine_tuned_summarizer.evaluate(max_new_tokens = 100)
    print("Fine-Tuned Model Evaluation Results: \n")

    for key, result in fine_tuned_results.items():
        print(f"{key}: {result}\n")

if __name__ == "__main__":
    main()
