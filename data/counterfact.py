from typing import Dict
import torch
import random

from data.base import BaseDataset



class COUNTERFACTDataset(BaseDataset):
    
    def __getitem__(self, idx) -> Dict[str, Dict[str, torch.LongTensor]]:
        row = self.data[idx]
        prompt = row["requested_rewrite"]["prompt"].format(row["requested_rewrite"]["subject"])
        equiv_prompt = random.choice(row["paraphrase_prompts"])
        answer = row["requested_rewrite"]["target_new"]["str"]
        unrel_prompt = random.choice(row["neighborhood_prompts"])
        unrel_answer = row["requested_rewrite"]["target_true"]["str"]
    
        return {
            "edit_tuples": self.tok_tuples(prompt, answer, unrel_answer),
            "equiv_tuples": self.tok_tuples(equiv_prompt, answer, unrel_answer),
            "unrel_tuples": self.tok_tuples(unrel_prompt, unrel_answer, answer)
        }
        

    def tok_tuples(
        self,
        prompt: str,
        answer: str,
        old_answer: str
    ) -> Dict[str, torch.LongTensor]:

        answer = " " + answer
        old_answer = " " + old_answer
        tok_prompt = self.tok(
            prompt,
            return_tensors = "pt",
            truncation=False,
            padding=False
        )
        tok_answer = self.tok(
            answer,
            return_tensors = "pt",
            add_special_tokens = False,
            truncation=False,
            padding=False
        )
        tok_old_answer = self.tok(
            old_answer,
            return_tensors = "pt",
            add_special_tokens = False,
            truncation=False,
            padding=False
        )   
        tok_tuples = {
            key: torch.cat((value, tok_answer[key][:, :-1]), -1)
            for key, value in tok_prompt.items()
        }
        tok_tuples["labels"] = torch.cat((
            torch.full(tok_prompt["input_ids"].shape, -100)[:, 1:],
            tok_answer["input_ids"]
        ), -1)
        tok_tuples["old_labels"] = torch.cat((
            torch.full(tok_prompt["input_ids"].shape, -100)[:, 1:],
            tok_old_answer["input_ids"]
        ), -1)

        return tok_tuples