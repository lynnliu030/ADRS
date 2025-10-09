from __future__ import annotations

from typing import Any, Callable

import os
import shutil
import tempfile

from gepa import EvaluationBatch, GEPAAdapter

from gepa.adapters.cant_be_late_adapter.openevolve.evaluator import (
    FAILED_SCORE,
    evaluate_stage1,
    evaluate_stage2,
)


class CantBeLateAdapter(GEPAAdapter[Any, Any, Any]):
    """Minimal adapter that wires OpenEvolve evaluator into GEPA."""
    def __init__(
        self,
        model: str | Callable,
        failure_score: float = FAILED_SCORE,
        max_litellm_workers: int = 10,
    ):
        if isinstance(model, str):
            import litellm  # type: ignore

            self.litellm = litellm
            model_name = model

            def _call_lm(prompt: str) -> str:
                completion = self.litellm.completion(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                )
                return completion.choices[0].message.content or ""

            self.reflection_lm = _call_lm
        else:
            self.reflection_lm = model
        self.failure_score = failure_score
        self.max_litellm_workers = max_litellm_workers
        self._last_tmpdir: str | None = None

    def _extract_trace_files(self, batch) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []

        for item in batch:
            if item is None:
                continue

            candidates: list[str] = []

            if isinstance(item, str):
                candidates.append(item)
            elif isinstance(item, dict):
                if "trace_file" in item and item["trace_file"]:
                    candidates.append(item["trace_file"])
                if "trace_files" in item and item["trace_files"]:
                    candidates.extend(item["trace_files"])
            else:
                trace_single = getattr(item, "trace_file", None)
                if trace_single:
                    candidates.append(trace_single)
                trace_list = getattr(item, "trace_files", None)
                if trace_list:
                    candidates.extend(trace_list)
                inputs_method = getattr(item, "inputs", None)
                if callable(inputs_method):
                    try:
                        inputs_dict = inputs_method()
                        if isinstance(inputs_dict, dict):
                            if "trace_file" in inputs_dict and inputs_dict["trace_file"]:
                                candidates.append(inputs_dict["trace_file"])
                            if "trace_files" in inputs_dict and inputs_dict["trace_files"]:
                                candidates.extend(inputs_dict["trace_files"])
                    except Exception:
                        pass

            for candidate in candidates:
                if not isinstance(candidate, str):
                    continue
                if candidate not in seen:
                    seen.add(candidate)
                    ordered.append(candidate)

        return ordered

    def build_program(
        self, candidate: dict[str, str]
    ) -> tuple[dict[str, str] | None, str | None]:
        """Write candidate code to a temp file and run stage 1 validation."""

        code = candidate["program"]
        tmpdir = tempfile.mkdtemp(prefix="cant_be_late_")
        program_path = os.path.join(tmpdir, "strategy.py")

        with open(program_path, "w", encoding="utf-8") as f:
            f.write(code)

        stage1 = evaluate_stage1(program_path)
        if stage1.get("runs_successfully", 0.0) < 1.0:
            stage1.setdefault("score", self.failure_score)
            stage1.setdefault("combined_score", self.failure_score)
            stage1.setdefault("trace_files", [])
            stage1.setdefault("stage", "stage1")
            shutil.rmtree(tmpdir, ignore_errors=True)
            return None, stage1

        self._last_tmpdir = tmpdir
        return {"program_path": program_path, "tmpdir": tmpdir}, None

    def evaluate(self, batch, candidate, capture_traces=False):
        """
        - If build passed 
            - Write the stage 2 check (e.g. evaluate_stage2) to ensure it runs successfully
            - Extract `score` 
            - Return `EvaluationBatch` with 
                - `outputs`: can be raw cost results or JSON dict 
                - `scores`: list of floats (or just replicating across examples? don't have per-example scores)
                - `trajectories`: can be error message or trace logs if you want reflective dataset 
        - If build failed
            - Return `EvaluationBatch` with 
                - `outputs`: None
                - `scores`: list of floats (or just replicating across examples? don't have per-example scores)
                - `trajectories`: feedback string
        """
        build_result, feedback = self.build_program(candidate)
        batch_size = len(batch)

        if build_result is None:
            return EvaluationBatch(
                outputs=[feedback for _ in range(batch_size)],
                scores=[self.failure_score for _ in range(batch_size)],
                trajectories=feedback,
            )

        program_path = build_result["program_path"]
        tmpdir = build_result["tmpdir"]

        trace_files = self._extract_trace_files(batch)

        try:
            result = evaluate_stage2(program_path, trace_files=trace_files or None)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
            if self._last_tmpdir == tmpdir:
                self._last_tmpdir = None

        runs_successfully = result.get("runs_successfully", 0.0) >= 1.0
        if trace_files and "trace_files" not in result:
            result["trace_files"] = trace_files

        scores = (
            [result.get("score", self.failure_score) for _ in range(batch_size)]
            if runs_successfully
            else [self.failure_score for _ in range(batch_size)]
        )

        return EvaluationBatch(
            outputs=[result for _ in range(batch_size)],
            scores=scores,
            trajectories=result,
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch,
        components_to_update: list[str],
    ):
        if "program" not in components_to_update:
            return {}

        dataset: list[dict[str, Any]] = []

        trajectories = eval_batch.trajectories
        if isinstance(trajectories, list):
            iterable = trajectories
        else:
            iterable = [trajectories]

        for score, trajectory in zip(eval_batch.scores, iterable, strict=False):
            if isinstance(trajectory, dict):
                success = trajectory.get("runs_successfully", 0.0) >= 1.0
                feedback_text = (
                    f"Stage 2 succeeded with score {score:.4f}."
                    if success
                    else trajectory.get("error", "Stage 2 reported failure without message.")
                )
                sample = {
                    "Score": score,
                    "Runs Successfully": success,
                    "Feedback": feedback_text,
                }
                if "avg_cost" in trajectory:
                    sample["Average Cost"] = trajectory["avg_cost"]
                if "combined_score" in trajectory:
                    sample["Combined Score"] = trajectory["combined_score"]
                if "trace_files" in trajectory:
                    sample["Trace Files"] = trajectory["trace_files"]
                dataset.append(sample)
            elif isinstance(trajectory, str):
                dataset.append(
                    {
                        "Score": score,
                        "Runs Successfully": False,
                        "Feedback": trajectory,
                    }
                )

        if not dataset:
            dataset.append(
                {
                    "Score": self.failure_score,
                    "Runs Successfully": False,
                    "Feedback": "No usable trajectory information was produced.",
                }
            )

        return {"program": dataset}


    def propose_new_texts(
        self,
        candidate: dict[str, str],
        reflective_dataset: dict[str, list[dict[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        """
        Use reflection LM and feedback dataset to rewrite the candidate "program" 
        
        Prompt already has the domain-specific instructions (Strategy API, SPOT/ON_DEMAND, etc.)
        """
        from gepa.adapters.cant_be_late_adapter.open_evolve_proposal_signature import (
            OpenEvolveProposalSignature,
        )

        new_texts: dict[str, str] = {}
        for name in components_to_update:
            base_instruction = candidate[name]
            dataset_with_feedback = reflective_dataset.get(name, [])

            new_texts[name] = OpenEvolveProposalSignature.run(
                lm=self.reflection_lm,
                input_dict={
                    "current_instruction_doc": base_instruction,
                    "dataset_with_feedback": dataset_with_feedback,
                },
            )["new_program"]

        return new_texts
