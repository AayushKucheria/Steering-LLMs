# Imports
from steerllm import main
import unittest
import transformer_lens

class MainTest(unittest.TestCase):

    def test_load(self):
        prompt_dict = main.csv_to_dictionary(main.PROMPT_FILE_PATH)
        keys = set([main.PROMPT_COLUMN, main.ETHICAL_AREA_COLUMN, main.POS_COLUMN])
        self.assertEqual(set(prompt_dict.keys()), keys)

        prompt_list = prompt_dict[main.PROMPT_COLUMN]
        self.assertTrue(all(isinstance(prompt, str) for prompt in prompt_list))
        self.assertTrue(len(prompt_list) > 0)

    def test_compute(self):
        # Check that it runs successfully
        model = transformer_lens.HookedTransformer.from_pretrained("gpt2-small")
        activations_cache = main.compute_activations(model, "test")
        self.assertTrue(True)

# Running the tests
if __name__ == '__main__':
    unittest.main()
