import os

import torch

from torchtitan.components.metrics import WandBLogger
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.components.training_data_logger import TrainingDataLogger
from torchtitan.config import JobConfig


class _FakeTokenizer(BaseTokenizer):
    def __init__(self):
        super().__init__()
        self.bos_id = 0
        self.eos_id = 1
        self.pad_id = 2
        self._pieces = {
            0: "<|bos|>",
            1: "<|eos|>",
            2: "<|pad|>",
            7: "<|assistant|>",
            9: "Ċ",
            10: "hello",
            11: "world",
            12: "answer",
            13: "token",
            14: "more",
            99: "<|MASK|>",
        }

    def encode(self, text: str, *args, **kwargs) -> list[int]:
        return [int(part) for part in text.split()] if text else []

    def decode(self, token_ids: list[int], *args, **kwargs) -> str:
        return " ".join(self._pieces.get(int(token_id), str(int(token_id))) for token_id in token_ids)

    def get_vocab_size(self) -> int:
        return 1024

    def id_to_token(self, token_id: int) -> str:
        return self._pieces.get(int(token_id), str(int(token_id)))


class _FakeWandB:
    def __init__(self):
        self.logged = []
        self.run = object()

    @staticmethod
    def Html(content: str, inject: bool = False):
        return {"html": content, "inject": inject}

    def log(self, payload, step):
        self.logged.append((payload, step))


def _build_job_config(tmp_path, *, async_worker: bool = False) -> JobConfig:
    job_config = JobConfig()
    job_config.job.dump_folder = str(tmp_path)
    job_config.job.config_file = (
        "/tmp/qwen3_1.7b_dllm_nemotron_post_training_v2_sft_correct_token.toml"
    )
    job_config.checkpoint.folder = "checkpoints"
    job_config.training_data_visualization.enable = True
    job_config.training_data_visualization.freq = 1
    job_config.training_data_visualization.num_samples = 1
    job_config.training_data_visualization.async_worker = async_worker
    job_config.training_data_visualization.upload_to_s3 = False
    job_config.dllm.block_size = 4
    job_config.dllm.mask_token_id = 99
    job_config.dllm.pad_token_id = 2
    return job_config


def test_training_data_logger_writes_interactive_rich_html(tmp_path, monkeypatch):
    monkeypatch.setenv("USER", "Test User")

    job_config = _build_job_config(tmp_path)
    logger = TrainingDataLogger(job_config, _FakeTokenizer())

    input_ids = torch.tensor([[0, 7, 10, 11, 9, 12, 99, 1]])
    labels = torch.tensor([[-100, -100, -100, 11, 9, 12, 99, 1]])
    viz_payload = {
        "masked_indices": torch.tensor(
            [
                [False, False, False, True, False, False, False, False],
                [False, False, False, False, True, False, True, False],
            ]
        )
    }

    samples = logger.log_batch_samples(
        input_ids=input_ids,
        labels=labels,
        step=10,
        viz_payload=viz_payload,
    )

    assert len(samples) == 1
    html_path = samples[0]["html_path"]
    assert os.path.exists(html_path)
    assert os.path.exists(samples[0]["json_path"])

    with open(html_path, "r", encoding="utf-8") as handle:
        html_content = handle.read()

    assert "Training Data Visualization Step 10" in html_content
    assert "DLLM Branch Assignment Mask (full)" in html_content
    assert "DLLM Block Mask" in html_content
    assert "DLLM Attention" in html_content
    assert "Aligned x0/xt Attention Window" in html_content
    assert "Tokenized + Labeled Table" in html_content
    assert "Primary masked" in html_content
    assert "Complementary masked" in html_content
    assert "canvas-popup" in html_content
    assert "canvas-host" in html_content
    assert "branch_assignment" in html_content
    assert "masked token in primary branch" in html_content
    assert "chat-template special token" in html_content


def test_training_data_logger_honors_job_run_name(tmp_path, monkeypatch):
    monkeypatch.setenv("USER", "viz_user")

    job_config = _build_job_config(tmp_path)
    job_config.job.run_name = "custom_stage_run"
    job_config.job.trial_name = "should_not_win"

    logger = TrainingDataLogger(job_config, _FakeTokenizer())
    samples = logger.log_batch_samples(
        input_ids=torch.tensor([[10, 11, 12, 13]]),
        labels=torch.tensor([[-100, 11, 12, 13]]),
        step=1,
        viz_payload=None,
    )

    assert len(samples) == 1
    assert "custom_stage_run" in samples[0]["html_path"]
    assert "should_not_win" not in samples[0]["html_path"]


def test_training_data_logger_async_worker_keeps_inline_html(tmp_path, monkeypatch):
    monkeypatch.setenv("USER", "async_user")

    job_config = _build_job_config(tmp_path, async_worker=True)
    job_config.training_data_visualization.log_to_wandb = True
    logger = TrainingDataLogger(job_config, _FakeTokenizer())

    samples = logger.log_batch_samples(
        input_ids=torch.tensor([[10, 11, 12, 13, 14, 1]]),
        labels=torch.tensor([[-100, -100, 12, 13, 14, 1]]),
        step=3,
        viz_payload={
            "masked_indices": torch.tensor(
                [[False, False, True, False, False, False]]
            )
        },
    )

    assert len(samples) == 1
    assert samples[0]["html_content"] is not None
    assert samples[0]["json_path"].endswith("step_00000003.json")

    logger.close()
    assert os.path.exists(samples[0]["html_path"])


def test_training_data_logger_handles_missing_viz_payload(tmp_path, monkeypatch):
    monkeypatch.setenv("USER", "plain_user")

    job_config = _build_job_config(tmp_path)
    logger = TrainingDataLogger(job_config, _FakeTokenizer())

    samples = logger.log_batch_samples(
        input_ids=torch.tensor([[10, 11, 12, 1]]),
        labels=torch.tensor([[-100, 11, 12, 1]]),
        step=2,
        viz_payload=None,
    )

    assert len(samples) == 1
    # Inline HTML is only retained when W&B logging is enabled. The local HTML
    # artifact is always written when visualization itself is enabled.
    assert samples[0]["html_content"] is None
    with open(samples[0]["html_path"], "r", encoding="utf-8") as handle:
        html_content = handle.read()
    assert "DLLM Branch Assignment Mask" in html_content


def test_wandb_logger_uses_qwen35_style_rich_payload_keys():
    fake_wandb = _FakeWandB()
    logger = WandBLogger.__new__(WandBLogger)
    logger.enabled = True
    logger.is_rank_zero = True
    logger.wandb = fake_wandb
    logger.tag = None
    logger._initialized = True
    logger.ensure_initialized = lambda: None

    logger.log_training_data_samples(
        [
            {
                "html_content": "<html><body>rich</body></html>",
                "html_path": None,
                "html_url": "https://example.com/viz.html",
                "wandb_log_url_only": False,
            }
        ],
        step=7,
    )

    assert len(fake_wandb.logged) == 1
    payload, step = fake_wandb.logged[0]
    assert step == 7
    assert "vis_batch" in payload
    assert "training_data/url" in payload


def test_wandb_logger_fallback_html_keeps_qwen35_style_keys():
    fake_wandb = _FakeWandB()
    logger = WandBLogger.__new__(WandBLogger)
    logger.enabled = True
    logger.is_rank_zero = True
    logger.wandb = fake_wandb
    logger.tag = None
    logger._initialized = True
    logger.ensure_initialized = lambda: None

    logger.log_training_data_samples(
        [
            {
                "sample_id": 0,
                "input_text": "hello",
                "label_text": "world",
                "html_content": None,
                "html_path": None,
                "html_url": "https://example.com/fallback.html",
                "wandb_log_url_only": False,
            }
        ],
        step=11,
    )

    assert len(fake_wandb.logged) == 1
    payload, step = fake_wandb.logged[0]
    assert step == 11
    assert "vis_batch" in payload
    assert "training_data/url" in payload
