import gc
from typing import Tuple

import numpy as np
import torch
import transformers
from loggers import ConsoleLogger, Logger, Verbosity
from visualization import RichTablePrinter


class Attributor:
    def __init__(self, logger: Logger = None):
        self.logger = logger

    def log(self, message: str, verbosity: Verbosity):
        if self.logger:
            self.logger.log(message=message, verbosity=verbosity)

    def get_attributions(
        self,
        model: transformers.models.gpt2.GPT2Model,
        tokenizer: transformers.PreTrainedTokenizerBase,
        input_string: str,
        generation_length: int,
    ):
        if not isinstance(model, torch.nn.Module):
            raise ValueError(
                "Model must be an instance of a class that inherits from torch.nn.Module"
            )

        if not isinstance(tokenizer, transformers.PreTrainedTokenizerBase):
            raise ValueError(
                "Tokenizer must be an instance of transformers.PreTrainedTokenizerBase"
            )
        if not isinstance(input_string, str):
            raise ValueError("Input string must be a string")
        if not isinstance(generation_length, int):
            raise ValueError("Generation length must be an integer")

        if model.training:
            raise ValueError("Model should be in evaluation mode, not training mode")

        token_ids: torch.Tensor = torch.tensor(tokenizer(input_string).input_ids).to(
            model.device
        )
        embeddings: torch.Tensor = model.transformer.wte.weight.detach()
        input_length: int = token_ids.shape[0]

        attr_scores = torch.zeros(generation_length, generation_length + len(token_ids))

        for it in range(generation_length):
            input_embeddings = self._get_input_embeddings(embeddings, token_ids)
            output = model(inputs_embeds=input_embeddings)

            gen_tokens, next_token_id = self._generate_tokens(
                model, token_ids, tokenizer
            )
            self.log(f"{tokenizer.decode(gen_tokens[0])}", Verbosity.INFO)
            grad = self._get_gradients(output, next_token_id, input_embeddings)

            attr_scores_next_token = self._get_attr_scores_next_token(
                grad, token_ids, generation_length + input_length
            )

            attr_scores[it] = attr_scores_next_token
            token_ids = torch.cat((token_ids, next_token_id.view(-1)), dim=0)

            self._cleanup()

        self.log(attr_scores.shape, Verbosity.INFO)
        self.log(token_ids.shape, Verbosity.INFO)
        return attr_scores, token_ids

    def _get_input_embeddings(
        self, embeddings: torch.Tensor, token_ids: torch.Tensor
    ) -> torch.Tensor:
        input_embeddings = embeddings[token_ids]
        return torch.nn.Parameter(input_embeddings, requires_grad=True)

    def _generate_tokens(
        self,
        model: transformers.models.gpt2.GPT2Model,
        token_ids: torch.Tensor,
        tokenizer: transformers.PreTrainedTokenizerBase,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attention_mask = torch.ones(token_ids.shape).unsqueeze(0).to(model.device)
        with torch.no_grad():
            gen_tokens = model.generate(
                token_ids.unsqueeze(0),
                max_new_tokens=1,
                attention_mask=attention_mask,
                pad_token_id=tokenizer.eos_token_id,
            )
        next_token_id = gen_tokens[0][-1]
        return gen_tokens, next_token_id

    def _get_gradients(
        self,
        output: transformers.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions,
        next_token_id: torch.Tensor,
        input_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        torch.softmax(output.logits, dim=1)[-1, next_token_id].backward()
        return input_embeddings.grad

    def _get_attr_scores_next_token(
        self, grad: torch.Tensor, token_ids: torch.Tensor, message_length: int
    ) -> torch.Tensor:
        attr_scores_next_token = torch.zeros(message_length)
        for i, _ in enumerate(token_ids):
            presence_grad = (grad[i]).norm(p=1)
            attr_scores_next_token[i] = presence_grad
        return attr_scores_next_token

    def _cleanup(self) -> None:
        torch.cuda.empty_cache()
        gc.collect()

    def print_attributions(
        self,
        word_list,
        attr_scores: np.array,
        token_ids: np.array,
        generation_length: int,
    ):
        max_abs_attr_val = attr_scores.abs().max().item()
        table_printer = RichTablePrinter(max_abs_attr_val)
        table_printer.print_attribution_table(
            word_list, attr_scores, token_ids, generation_length
        )


if __name__ == "__main__":
    model_name = "distilgpt2"
    tokenizer = transformers.GPT2Tokenizer.from_pretrained(
        model_name, padding_side="left"
    )
    tokenizer.pad_token = tokenizer.eos_token
    model = transformers.GPT2LMHeadModel.from_pretrained(model_name).eval()

    attributor = Attributor(logger=ConsoleLogger())
    attr_scores, token_ids = attributor.get_attributions(
        model, tokenizer, "the five continents are asia, europe, afri", 7
    )

    attributor.print_attributions(
        word_list=[tokenizer.decode(token_id) for token_id in token_ids],
        attr_scores=attr_scores,
        token_ids=token_ids,
        generation_length=7,
    )
