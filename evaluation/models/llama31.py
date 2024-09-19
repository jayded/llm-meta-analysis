from .model import Model
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class Llama31(Model):
    def __init__(self) -> None:
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.__load_model()
        self.tokenizer = self.__load_tokenizer()

    def get_context_length(self) -> int:
        return 128000
    
    def encode_text(self, text: str) -> str:
        """
        This method encodes the text

        :param text: text to encode

        :return encoded text
        """
        # 30,000 is some arbitrary number that is larger than the maximum context length of the model
        return self.tokenizer.encode(text, max_length = 128000)

    def __load_model(self):
        # fine-tuned on biomedical texts but only to context of 2048 tokens
        # also have only been evaluated on biomedical tasks that are multiple choice questions      
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", device_map="auto")
        return model

    def __load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
        return tokenizer

    def generate_output(self, input: str, max_new_tokens: int) -> str:
        """
        This method generates the output given the input

        :param input: input to the model
        :param max_new_tokens: maximum number of tokens to generate

        :return output of the model
        """
        try:
            chat = [
                {"role": "user", "content": input},
            ]
            encoded = self.tokenizer.apply_chat_template(chat, return_tensors="pt")
            model_inputs = encoded.to(self.device)
            with torch.no_grad():
                # https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct/blob/main/generation_config.json
                # these seem to be the same as the default decoding parameters from hf model info
                result = self.model.generate(model_inputs, max_new_tokens=max_new_tokens + 10, do_sample=True, top_p=.9, temperature=.6)
            decoded = self.tokenizer.decode(result[0, model_inputs.shape[1]:], skip_special_tokens=True)
            # the chat template
            decoded = decoded.replace('assistant\n\n', '')
            return decoded
        except Exception as e:
            print("[ERROR]", e)
