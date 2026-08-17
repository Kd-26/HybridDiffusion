"""
MATH-500 evaluation across one or more SGLang servers.

Uses math_verify for answer verification (same as OpenCompass MATHVerifyEvaluator).
Strips <think>...</think> before extracting \boxed{} answers.

Normally invoked through ``scripts/evaluate.sh``. Direct usage:
  python benchmark_clients/quality/eval_math500.py --ports 30000
"""
import argparse
import json
import re
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from latex2sympy2_extended import NormalizationConfig
from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify

MODEL_NAME = "default"


MATH_HF_DATASET = "HuggingFaceH4/MATH-500"


def strip_thinking(text):
    if not text:
        return ""
    return re.sub(r'^.*</think>\s*', '', text, count=1, flags=re.DOTALL)


def verify_answer(prediction, reference):
    """Verify prediction against reference using math_verify (OpenCompass-compatible)."""
    prediction = strip_thinking(prediction)

    ref_with_env = f'${reference}$'
    gold_parsed = parse(
        ref_with_env,
        extraction_mode='first_match',
        extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()],
    )

    if len(gold_parsed) == 0:
        return False

    pred_parsed = parse(
        prediction,
        extraction_config=[
            LatexExtractionConfig(
                boxed_match_priority=0,
            ),
            ExprExtractionConfig(),
        ],
    )

    if len(pred_parsed) == 0:
        return False

    try:
        return verify(gold_parsed, pred_parsed)
    except Exception:
        return False


def run_one(args):
    i, problem, port, max_tokens, timeout, temperature, top_p, top_k, presence_penalty, enable_thinking = args
    prompt = f"{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
    try:
        response = requests.post(
            f"http://localhost:{port}/v1/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "presence_penalty": presence_penalty,
                "chat_template_kwargs": {"enable_thinking": enable_thinking},
            },
            timeout=timeout,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise RuntimeError(
                f"{exc}: {response.text[:500]}"
            ) from exc
        payload = response.json()
        pred = payload["choices"][0]["message"]["content"]
        comp = payload["usage"]["completion_tokens"]
        return i, pred, comp, None
    except Exception as e:
        return i, "", 0, str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="default")
    parser.add_argument("--num-problems", type=int, default=None)
    parser.add_argument("--ports", type=int, nargs="+", default=[30000 + i for i in range(8)])
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--presence-penalty", type=float, default=1.5)
    parser.add_argument("--disable-thinking", dest="enable_thinking", action="store_false")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Optional OpenCompass-style local JSON; defaults to the public HF dataset",
    )
    args = parser.parse_args()
    global MODEL_NAME
    MODEL_NAME = args.model

    import os
    if args.data_path:
        if not os.path.isfile(args.data_path):
            raise FileNotFoundError(f"MATH-500 data file not found: {args.data_path}")
        with open(args.data_path) as f:
            data = json.load(f)
        problems = [(data[str(i)]["problem"], data[str(i)]["solution"]) for i in range(len(data))]
    else:
        from datasets import load_dataset
        ds = load_dataset(MATH_HF_DATASET, split="test")
        problems = [(item["problem"], item["answer"]) for item in ds]
    N = min(args.num_problems, len(problems)) if args.num_problems else len(problems)
    problems = problems[:N]
    ports = args.ports
    max_workers = args.max_workers or N

    print(
        f"MATH-500: {N} problems, {len(ports)} servers, "
        f"max_tokens={args.max_tokens}, temp={args.temperature}, "
        f"top_p={args.top_p}, top_k={args.top_k}, "
        f"presence_penalty={args.presence_penalty}, enable_thinking={args.enable_thinking}"
    )

    tasks = [
        (
            i,
            prob,
            ports[i % len(ports)],
            args.max_tokens,
            args.timeout,
            args.temperature,
            args.top_p,
            args.top_k,
            args.presence_penalty,
            args.enable_thinking,
        )
        for i, (prob, _) in enumerate(problems)
    ]

    print(f"Running {N} problems, {max_workers} concurrent workers...")
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(run_one, t) for t in tasks]
        done = 0
        for f in as_completed(futures):
            done += 1
            if done % max(N // 5, 1) == 0:
                print(f"  {done}/{N} done ({time.time() - t0:.0f}s)")
        results = [f.result() for f in futures]

    elapsed = time.time() - t0

    # Verify
    predictions = [""] * N
    request_errors = [None] * N
    total_tok = 0
    errors = 0
    for i, pred, comp, err in results:
        predictions[i] = pred
        request_errors[i] = err
        total_tok += comp
        if err:
            errors += 1

    correct = 0
    details = []
    for i, (pred, (prob, sol)) in enumerate(zip(predictions, problems)):
        is_correct = verify_answer(pred, sol)
        correct += is_correct
        details.append({
            "idx": i,
            "problem": prob,
            "solution": sol,
            "prediction": pred,
            "correct": is_correct,
            "error": request_errors[i],
        })

    score = correct / N * 100
    throughput = total_tok / elapsed
    print(f"\n{'=' * 55}")
    print(f"MATH-500 {N} problems, {len(ports)} servers")
    print(f"{'=' * 55}")
    print(f"Accuracy:       {score:.1f}% ({correct}/{N})")
    print(f"Total tokens:   {total_tok:,}")
    print(f"Wall time:      {elapsed:.1f}s")
    print(f"Throughput:     {throughput:.1f} tok/s")
    print(f"Errors:         {errors}")

    if args.output_dir:
        import os
        os.makedirs(args.output_dir, exist_ok=True)
        tag = f"_{args.tag}" if args.tag else ""
        with open(os.path.join(args.output_dir, f"math500_summary{tag}.json"), "w") as f:
            json.dump(
                {
                    "score": score,
                    "correct": correct,
                    "total": N,
                    "errors": errors,
                    "total_tokens": total_tok,
                    "wall_time_seconds": elapsed,
                    "throughput_tokens_per_second": throughput,
                    "max_tokens": args.max_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "top_k": args.top_k,
                    "presence_penalty": args.presence_penalty,
                    "enable_thinking": args.enable_thinking,
                },
                f,
                indent=2,
            )
        with open(os.path.join(args.output_dir, f"math500_details{tag}.json"), "w") as f:
            json.dump(details, f, indent=2, ensure_ascii=False)
        print(f"Saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
