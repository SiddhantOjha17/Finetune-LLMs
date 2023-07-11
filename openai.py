import os
import openai
import json
import time 
from typing import List, Optional
import csv
from logging import Logger

# openai.organization = "YOUR_ORG_ID"
# APIKEY
# openai.Model.list()


class Finetune(object):

    def __init__(self, logger: Logger):
        self.logger = logger

    def generate_jsonl_from_csv(self, csv_file: str, output_file: str):
        # generate_jsonl_from_csv('input.csv', 'output.jsonl')

        prompt_completion_pairs = []

        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    prompt = row[0]
                    completion = row[1]
                    prompt_completion_pairs.append((prompt, completion))

        with open(output_file, 'w') as f:
            for pair in prompt_completion_pairs:
                json_obj = {
                    'prompt': pair[0],
                    'completion': pair[1]
                }
                json_str = json.dumps(json_obj)
                f.write(json_str + '\n')
        return output_file

    def create_file(self, output_file: str):
        try:
            openai.File.create(file=open(output_file, "rb"), purpose='fine-tune')
            return output_file
        except Exception as e:
            self.logger.error(f"Error creating file: {e}")
            raise e

    def model(self, model_name: str, input: str, instruction: str, n: int, temperature: float, top_p: float):
        try:
            model = openai.Edit.create(
                model=model_name, temperature=temperature,
                top_p=top_p, input=input,
                instruction=instruction, n=n,)
            return model
        except Exception as e:
            self.logger.error(f"Error creating model: {e}")
            raise e

    def finetune(self, training_file: str, model_name: Optional[str] = "curie", n_epoch: Optional[int] = 4, 
                 validation_file: Optional[str] = None, batch_size: Optional[int] = None, 
                 learning_rate_multiplier: Optional[int] = None, prompt_loss_weight: Optional[int] = 0.01, 
                 compute_classification_metrics: Optional[bool] = False,
                 classification_n_classes: Optional[int] = None, classification_positive_class: Optional[str] = None, 
                 classification_betas: Optional[List[float]] = None, suffix: Optional[str] = None):            
        # openai.FineTune.create(training_file="file-XGinujblHPwGLSztz8cPS8XY")
        
        job_id = None
        try:
            job_id = openai.FineTune.create(training_file=training_file,
                               model=model_name, n_epochs=n_epoch, validation_file=validation_file,
                               batch_size=batch_size, learning_rate_multiplier=learning_rate_multiplier,
                               prompt_loss_weight=prompt_loss_weight, 
                               compute_classification_metrics=compute_classification_metrics,
                               classification_n_classes=classification_n_classes, 
                               classification_positive_class=classification_positive_class,
                               classification_betas=classification_betas, suffix=suffix)
        except Exception as e:
            self.logger.error(f"Error creating fine-tune job: {e}")
            raise e

        if job_id is not None:
            while openai.FineTune.status(job_id) == "pending":
                time.sleep(1)
                self.logger.info("Fine-tuning job status: %s", openai.FineTune.status(job_id))

            if openai.FineTune.status(job_id) == "failed":
                self.logger.error("Fine-tuning job failed")
                raise Exception("Fine-tuning job failed")
