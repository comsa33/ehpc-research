import logging
from typing import Any, Dict, List

import evaluate
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from ..core.prompt_compressor import EHPCCompressor


class EHPCBenchmark:
    """EHPC 성능 벤치마크 클래스"""

    def __init__(self, compressor: EHPCCompressor):
        self.compressor = compressor
        self.rouge = evaluate.load("rouge")

    def run_qa_benchmark(
        self,
        dataset_name: str = "squad",
        num_samples: int = 100,
        compression_ratios: List[float] = [0.2, 0.3, 0.5, 0.7],
    ) -> Dict[str, Any]:
        """QA 태스크 벤치마크"""
        logging.info(f"Running QA benchmark on {dataset_name}")

        # 데이터셋 로드
        if dataset_name == "squad":
            dataset = load_dataset("squad", split=f"validation[:{num_samples}]")
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        results = {
            ratio: {"exact_match": [], "f1": [], "rouge": []}
            for ratio in compression_ratios
        }

        for example in tqdm(dataset, desc="Processing examples"):
            context = example["context"]
            question = example["question"]
            ground_truth = (
                example["answers"]["text"][0] if example["answers"]["text"] else ""
            )

            # 프롬프트 구성
            prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"

            for ratio in compression_ratios:
                try:
                    # 압축 및 생성
                    result = self.compressor.compress_and_generate(
                        prompt, compression_ratio=ratio, max_new_tokens=50
                    )

                    compressed_answer = self._extract_answer(
                        result["compressed_response"]
                    )

                    # 평가 지표 계산
                    exact_match = self._calculate_exact_match(
                        compressed_answer, ground_truth
                    )
                    f1_score = self._calculate_f1(compressed_answer, ground_truth)
                    rouge_score = self.rouge.compute(
                        predictions=[compressed_answer], references=[ground_truth]
                    )["rougeL"]

                    results[ratio]["exact_match"].append(exact_match)
                    results[ratio]["f1"].append(f1_score)
                    results[ratio]["rouge"].append(rouge_score)

                except Exception as e:
                    logging.warning(f"Error processing example with ratio {ratio}: {e}")
                    continue

        # 평균 계산
        summary_results = {}
        for ratio in compression_ratios:
            if results[ratio]["exact_match"]:
                summary_results[ratio] = {
                    "exact_match": np.mean(results[ratio]["exact_match"]),
                    "f1": np.mean(results[ratio]["f1"]),
                    "rouge": np.mean(results[ratio]["rouge"]),
                    "samples": len(results[ratio]["exact_match"]),
                }

        return summary_results

    def _extract_answer(self, generated_text: str) -> str:
        """생성된 텍스트에서 답변 추출"""
        # "Answer:" 이후 부분만 추출
        if "Answer:" in generated_text:
            answer = generated_text.split("Answer:")[-1].strip()
        else:
            answer = generated_text.strip()

        # 첫 번째 문장만 사용
        sentences = answer.split(".")
        return sentences[0].strip()

    def _calculate_exact_match(self, pred: str, truth: str) -> int:
        """정확한 일치 계산"""
        return int(pred.lower().strip() == truth.lower().strip())

    def _calculate_f1(self, pred: str, truth: str) -> float:
        """토큰 레벨 F1 점수 계산"""
        pred_tokens = set(pred.lower().split())
        truth_tokens = set(truth.lower().split())

        if not truth_tokens:
            return int(not pred_tokens)

        intersection = pred_tokens & truth_tokens
        precision = len(intersection) / len(pred_tokens) if pred_tokens else 0
        recall = len(intersection) / len(truth_tokens)

        if precision + recall == 0:
            return 0

        return 2 * precision * recall / (precision + recall)

    def run_summarization_benchmark(
        self, num_samples: int = 50, compression_ratios: List[float] = [0.2, 0.3, 0.5]
    ) -> Dict[str, Any]:
        """요약 태스크 벤치마크"""
        logging.info("Running summarization benchmark")

        # CNN/DailyMail 데이터셋 로드
        dataset = load_dataset("cnn_dailymail", "3.0.0", split=f"test[:{num_samples}]")

        results = {
            ratio: {"rouge1": [], "rouge2": [], "rougeL": []}
            for ratio in compression_ratios
        }

        for example in tqdm(dataset, desc="Processing summaries"):
            article = example["article"]
            reference_summary = example["highlights"]

            # 요약 프롬프트 구성
            prompt = f"Summarize the following article:\n{article}\nSummary:"

            for ratio in compression_ratios:
                try:
                    result = self.compressor.compress_and_generate(
                        prompt, compression_ratio=ratio, max_new_tokens=100
                    )

                    generated_summary = self._extract_summary(
                        result["compressed_response"]
                    )

                    # ROUGE 점수 계산
                    rouge_scores = self.rouge.compute(
                        predictions=[generated_summary], references=[reference_summary]
                    )

                    results[ratio]["rouge1"].append(rouge_scores["rouge1"])
                    results[ratio]["rouge2"].append(rouge_scores["rouge2"])
                    results[ratio]["rougeL"].append(rouge_scores["rougeL"])

                except Exception as e:
                    logging.warning(f"Error processing summary with ratio {ratio}: {e}")
                    continue

        # 평균 계산
        summary_results = {}
        for ratio in compression_ratios:
            if results[ratio]["rouge1"]:
                summary_results[ratio] = {
                    "rouge1": np.mean(results[ratio]["rouge1"]),
                    "rouge2": np.mean(results[ratio]["rouge2"]),
                    "rougeL": np.mean(results[ratio]["rougeL"]),
                    "samples": len(results[ratio]["rouge1"]),
                }

        return summary_results

    def _extract_summary(self, generated_text: str) -> str:
        """생성된 텍스트에서 요약 추출"""
        if "Summary:" in generated_text:
            summary = generated_text.split("Summary:")[-1].strip()
        else:
            summary = generated_text.strip()

        # 너무 길면 첫 3문장만 사용
        sentences = summary.split(".")[:3]
        return ".".join(sentences).strip()
