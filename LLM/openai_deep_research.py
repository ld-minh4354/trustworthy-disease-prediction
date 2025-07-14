from openai import OpenAI
import os, warnings



# Class to call the ChatGPT API
class APICaller:
    def __init__(self):
        f = open(os.path.join("LLM", "uwu.txt"), "r")
        api_key = f.read()
        self.__client = OpenAI(api_key=api_key)

    
    # Function to call the API
    def call(self, input_text: str, mini_model: bool = True):
        if mini_model:
            model = "o4-mini-deep-research-2025-06-26"
        else:
            model = "o3-deep-research-2025-06-26"

        response = self.__client.responses.parse(
            model = model,
            input = input_text,
            tools = [{"type": "web_search_preview"}],
            max_tool_calls = 15
        )

        return response.output_text
    


class LitReview:
    def __init__(self):
        self.api_caller = APICaller()

        self.input_template = (
            "Do literature review to determine whether the factor: {}, causes the disease: {}.\n"

            "If there is no or weak causation, only output 'no or weak causation' and nothing else.\n"

            "If there is strong causation, but the disease causes the factor, "
            "only output 'disease causes factor' and nothing else.\n"

            "If there is strong causation, only output the following two items and nothing else: "
            "(1) IEEE citation of a quality peer-reviewed research paper which shows that {} causes {}, and "
            "(2) A 50-word summary of how this paper shows that {} causes {}. "
        )


    def get_input_text(self, factor, disease):
        return self.input_template.format(factor, disease, factor, disease, factor, disease)


    def process_factor(self, factor, disease):
        input_text = self.get_input_text(factor, disease)
        result = self.api_caller.call(input_text, True)
        print(result)



if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    lit_review = LitReview()
    lit_review.process_factor("weak physical health", "asthma")