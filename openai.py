import os
import openai
import json
import time

from typing import List, Optional, Union
from pathlib import Path
import csv
from logging import Logger

# openai.organization = "YOUR_ORG_ID"
# APIKEY
# openai.Model.list()




class Finetune:

    def __init__(self, logger: Logger):
        self.logger = logger

    def generate_jsonl_from_csv(self, csv_file: str, output_file: str) -> str:
        """
        Creates a `.jsonl` file from a `.csv`

        Args:
            csv_file (str): Path to the .csv file
            output_file (str): Path to the .jsonl file

        Raises:
            Exception: Raised when `csv_file` path is not of `.csv` file
            Exception: Raised when `output_file` path is not of `.jsonl` file

        Returns:
            str: Path to the .jsonl file

        Usage:
        >>> generate_jsonl_from_csv('input.csv', 'output.jsonl')
        """
        # generate_jsonl_from_csv('input.csv', 'output.jsonl')

        if not csv_file.endswith(".csv"):
            self.logger.error(
                "args `csv_file` must be the **file** path to the .csv file"
            )
            raise Exception(
                "args `csv_file` must be the **file** path to the .csv file"
            )

        if not output_file.endswith(".jsonl"):
            self.logger.error(
                "args `output_file` must be the **file** path to the .jsonl file"
            )
            raise Exception(
                "args `output_file` must be the **file** path to the .jsonl file"
            )

        prompt_completion_pairs = []

        with open(csv_file, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    prompt = row[0]
                    completion = row[1]
                    prompt_completion_pairs.append((prompt, completion))

        with open(output_file, "w") as f:
            for pair in prompt_completion_pairs:
                json_obj = {"prompt": pair[0], "completion": pair[1]}
                json_str = json.dumps(json_obj)
                f.write(json_str + "\n")
        return output_file

    def create_file(self, output_file: str) -> str:
        """
        Uploads a file that contains document(s) to be used across endpoints/features

        Args:
            output_file (str): Path to the `.jsonl` file

        Raises:
            Exception: Raised when path is not of `.jsonl` file
            e: Captures exceptions when creating an `openai.File`

        Returns:
            str: Path of the `.jsonl` file
        """
        if not output_file.endswith(".jsonl"):
            raise Exception(
                "args `output_file` must be the **file** path to the .jsonl file"
            )
        try:
            openai.File.create(file=open(output_file, "rb"), purpose="fine-tune")
            return output_file
        except Exception as e:
            self.logger.error(f"Error creating file: {e}")
            raise e

    # TODO: Specify use of the method
    # def model(
    #     self,
    #     model_name: str,
    #     input: str,
    #     instruction: str,
    #     n: int,
    #     temperature: float,
    #     top_p: float,
    # ):
    #     try:
    #         model = openai.Edit.create(
    #             model=model_name,
    #             temperature=temperature,
    #             top_p=top_p,
    #             input=input,
    #             instruction=instruction,
    #             n=n,
    #         )
    #         return model
    #     except Exception as e:
    #         self.logger.error(f"Error creating model: {e}")
    #         raise e

    def finetune(
        self,
        training_file: str,
        model_name: Optional[str] = "curie",
        n_epoch: Optional[int] = 4,
        validation_file: Optional[str] = None,
        batch_size: Optional[int] = None,
        learning_rate_multiplier: Optional[int] = None,
        prompt_loss_weight: Optional[int] = 0.01,
        compute_classification_metrics: Optional[bool] = False,
        classification_n_classes: Optional[int] = None,
        classification_positive_class: Optional[str] = None,
        classification_betas: Optional[List[float]] = None,
        suffix: Optional[str] = None,
    ):
        """
        Fine-tunes the specified model

        Args:
            training_file (str): The ID of an uploaded file that contains training data.
            model_name (Optional[str], optional): The name of the base model to fine-tune. You can select one of "ada", "babbage", "curie", "davinci", or a fine-tuned model created after 2022-04-21. Defaults to "curie".
            n_epoch (Optional[int], optional):  Number of epochs to train the model for. Defaults to 4.
            validation_file (Optional[str], optional): The ID of an uploaded file that contains validation data. Defaults to None.
            batch_size (Optional[int], optional): Batch size to use for training. Defaults to None.
            learning_rate_multiplier (Optional[int], optional): Learning rate multiplier to use for training. Defaults to None.
            prompt_loss_weight (Optional[int], optional): Weight to use for loss on the prompt tokens. Defaults to 0.01.
            compute_classification_metrics (Optional[bool], optional): If True, classification metrics such as accuracy and f1-score are computed for validation set. Defaults to False.
            classification_n_classes (Optional[int], optional): Number of classes in a classification task. Defaults to None.
            classification_positive_class (Optional[str], optional): This parameter is needed to generate precision, recall, and F1 metrics when doing binary classification. Defaults to None.
            classification_betas (Optional[List[float]], optional): If this is provided, we calculate F-beta scores at the specified beta values. Defaults to None.
            suffix (Optional[str], optional): A string of up to 40 characters that will be added to your fine-tuned model name. Defaults to None.

        Raises:
            e: Errors generated while creating fine-tune job
            Exception: If fine-tuning job fails

        Returns:
            _type_: _description_
        """
        # openai.FineTune.create(training_file="file-XGinujblHPwGLSztz8cPS8XY")

        job_id = None
        try:
            job_id = openai.FineTune.create(
                training_file=training_file,
                model=model_name,
                n_epochs=n_epoch,
                validation_file=validation_file,
                batch_size=batch_size,
                learning_rate_multiplier=learning_rate_multiplier,
                prompt_loss_weight=prompt_loss_weight,
                compute_classification_metrics=compute_classification_metrics,
                classification_n_classes=classification_n_classes,
                classification_positive_class=classification_positive_class,
                classification_betas=classification_betas,
                suffix=suffix,
            )
            while openai.FineTune.status(job_id) == "pending":
                time.sleep(1)
                self.logger.info(
                    "Fine-tuning job status: %s", openai.FineTune.status(job_id)
                )

            if openai.FineTune.status(job_id) == "failed":
                self.logger.error("Fine-tuning job failed")
                raise Exception("Fine-tuning job failed")

            self.logger.info("Fine-tuning job completed successfully")
            return job_id

        except Exception as e:
            self.logger.error(f"Error creating fine-tune job: {e}")
            raise e
