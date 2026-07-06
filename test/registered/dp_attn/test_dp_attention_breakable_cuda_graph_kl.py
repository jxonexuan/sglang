import inspect
import unittest

from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kl_test_utils import (
    _extract_output_logprobs,
    _flush_cache,
    _generate,
    _get_input_logprobs,
    compare_kl_divergence,
)
from sglang.test.test_utils import (
    DEFAULT_TARGET_MODEL_EAGLE_DP_ATTN,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=300, stage="base-b", runner_config="2-gpu-large")


MODEL = DEFAULT_TARGET_MODEL_EAGLE_DP_ATTN
KL_THRESHOLD = 0.05
MAX_NEW_TOKENS = 8
PROMPTS = [
    "A concise definition of machine learning is",
    "The capital city of France is",
    "In Python, a list comprehension can",
    "The first step in debugging a program is",
]


class TestDPAttentionBreakableCudaGraphKL(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp",
                "2",
                "--dp",
                "2",
                "--enable-dp-attention",
                "--trust-remote-code",
                "--attention-backend",
                "triton",
                "--cuda-graph-backend-prefill=breakable",
                "--moe-runner-backend",
                "triton",
                "--cuda-graph-bs-prefill",
                "4",
                "8",
                "16",
                "32",
                "64",
                "--cuda-graph-max-bs-decode",
                "16",
                "--chunked-prefill-size",
                "128",
                "--max-running-requests",
                "16",
                "--mem-fraction-static",
                "0.70",
            ],
        )
        tokenizer = get_tokenizer(cls.model)
        cls.input_ids = [
            tokenizer.encode(prompt, add_special_tokens=False) for prompt in PROMPTS
        ]

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def _assert_results(self, results, expected_len):
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), expected_len)
        for result in results:
            self.assertIn("meta_info", result)
            self.assertIn("output_ids", result)

    def _compare_prefill_and_decode_logprobs(
        self, new_input_ids, output_logprobs, test_name
    ):
        input_logprobs = _get_input_logprobs(
            self.base_url,
            new_input_ids,
            output_logprobs,
        )
        compare_kl_divergence(
            input_logprobs,
            output_logprobs,
            {self.model: {"kl_div": KL_THRESHOLD}},
            self.model,
            test_name,
        )

    def test_decode_logprobs_match_prefill(self):
        _flush_cache(self.base_url)
        results = _generate(
            self.base_url,
            self.input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            return_logprob=True,
        )
        self._assert_results(results, len(self.input_ids))

        new_input_ids = []
        output_logprobs = []
        for prompt_ids, result in zip(self.input_ids, results):
            new_input_ids.append(prompt_ids + result["output_ids"])
            output_logprobs.append(_extract_output_logprobs(result))

        self.assertEqual(len(new_input_ids), len(self.input_ids))
        self._compare_prefill_and_decode_logprobs(
            new_input_ids,
            output_logprobs,
            inspect.currentframe().f_code.co_name,
        )


if __name__ == "__main__":
    unittest.main()
