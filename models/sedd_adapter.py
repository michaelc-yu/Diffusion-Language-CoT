from models.sedd_core.utils import load_sedd
from models.sedd_core.sample import sample
from transformers import AutoTokenizer

class SEDDAdapter:
    def __init__(self, model_id="louaaron/sedd-small"):
        self.model, self.graph, self.noise = load_sedd(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

    def generate(self, prompt="", steps=64, max_length=128):
        # (you can implement prefix/suffix or conditional logic later)
        return sample(self.model, self.tokenizer, steps=steps, max_length=max_length)



model = SEDDAdapter("louaaron/sedd-small")
output = model.generate(prompt="Once upon a time,", steps=32, max_length=64)
print("\n--- Generated Text ---\n")
print(output)

