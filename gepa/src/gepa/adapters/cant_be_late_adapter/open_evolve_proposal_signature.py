# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

from gepa.proposer.reflective_mutation.base import Signature
from typing import Optional
import re

class OpenEvolveProposalSignature(Signature):
    # Adapt from full_rewrite_prompt_template in openEvolve
    prompt_template = """I provided an assistant with the following instructions to perform a task for me:
```
<curr_instructions>
```

The following are examples of task inputs provided to the assistant along with the assistant's response for each of them, and some feedback on how the assistant's response could be better:
```
<inputs_outputs_feedback>
```

Your task is to rewrite the program to improve its performance on the specified metrics.
Provide the complete new program code.

IMPORTANT: Make sure your rewritten program maintains the same inputsf and outputs
as the original program, but with improved internal implementation.

Provide the new rewritten program within 
```{language}
# Your rewritten program here
```
blocks.
"""

    input_keys = ["current_instruction_doc", "dataset_with_feedback"]
    output_keys = ["new_program"]

    @classmethod
    def prompt_renderer(cls, input_dict: dict[str, str]) -> str:
        # TODO(shu): we don't need many datasets with feedback here 
        # TODO(shu): we just need to provide the original program instructions + execution feedback trace 
        def format_samples(samples):
            formatted = []
            for i, sample in enumerate(samples, 1):
                s = f"Example {i}:\n"
                for key, val in sample.items():
                    s += f"- {key}: {val}\n"
                formatted.append(s.strip())
            return "\n\n".join(formatted)

        prompt = cls.prompt_template
        prompt = prompt.replace("<curr_instructions>", input_dict["current_instruction_doc"])
        prompt = prompt.replace("<inputs_outputs_feedback>", format_samples(input_dict["dataset_with_feedback"]))
        return prompt


    @classmethod
    def _parse_full_rewrite(cls, llm_response: str, language: str = "python") -> Optional[str]:
        """
        Extract a full rewrite from an LLM response

        Args:
            llm_response: Response from the LLM
            language: Programming language

        Returns:
            Extracted code or None if not found
        """
        code_block_pattern = r"```" + language + r"\n(.*?)```"
        matches = re.findall(code_block_pattern, llm_response, re.DOTALL)

        if matches:
            return matches[0].strip()

        # Fallback to any code block
        code_block_pattern = r"```(.*?)```"
        matches = re.findall(code_block_pattern, llm_response, re.DOTALL)

        if matches:
            return matches[0].strip()

        # Fallback to plain text
        return llm_response

    @classmethod
    def output_extractor(cls, lm_out: str) -> dict[str, str]:
        # TODO(shu): just add some config.language for other languages
        new_program = cls._parse_full_rewrite(lm_out, language="python")

        return {"new_program": new_program}
