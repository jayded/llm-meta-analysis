from .model import Model
from openai import OpenAI

class GPT4(Model):
    def __init__(self) -> None:
        super().__init__()
        self.client = OpenAI(organization='org-amDbJ4wMNLPWA2hhgt3UdF7k',)

    def generate_output(self, input: str, max_new_tokens: int, temperature: str = 1) -> str:
        """
        This method generates the output given the input

        :param input: input to the model

        :return output of the model
        """
        try:
            completion = self.client.chat.completions.create(
                model="gpt-4-0125-preview", # 128,000 tokens (https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo)
                messages=[
                    {"role": "system", "content": "You are a helpful assistant for conducting meta-analyses of randomized controlled trials."},
                    {"role": "user", "content": input}
                ],
                # TODO: currently set as default but should figure out temperature/top_p parameters
                # https://community.openai.com/t/cheat-sheet-mastering-temperature-and-top-p-in-chatgpt-api/172683
                temperature=temperature,
                top_p=1,
                max_tokens=max_new_tokens,
            )
        except Exception as e:
            print(e)
            return "Error: GPT-4 API call failed."

        return completion.choices[0].message.content