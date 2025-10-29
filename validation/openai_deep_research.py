from openai import OpenAI
import pandas as pd
import os, warnings



# Class to call the ChatGPT API
class APICaller:
    def __init__(self):
        f = open(os.path.join("validation", "uwu.txt"), "r")
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
            "Find a peer-reviewed research paper (journal or conference) which shows that the factor: {}, "
            "either increases or reduces or affects the risk of having the disease: {}.\n"

            "Output the following two items and nothing else: "
            "(1) IEEE citation of this research paper, and "
            "(2) A 50-word summary of this paper.\n"

            "If you cannot find any paper, output 'no paper'."
        )


    def get_input_text(self, factor, disease):
        return self.input_template.format(factor, disease)


    def process_factor(self, factor, disease):
        input_text = self.get_input_text(factor, disease)
        result = self.api_caller.call(input_text, True)
        print(result)
        return result


class OpenAIResearch:
    def __init__(self):
        self.df = pd.read_csv(os.path.join("data", "final", "factor_list_openai.csv"))
        if "openai_answer" not in self.df.columns:
            self.df["openai_answer"] = None

        self.lit_review = LitReview()


    def deep_research(self):
        for idx, row in self.df.iterrows():
            disease = row["disease"]
            factor = row["factor"]

            print(f"Researching {factor} causing {disease}. Result:\n")

            if pd.isna(row["openai_answer"]):
                result = self.lit_review.process_factor(factor, disease)
                self.df.at[idx, "openai_answer"] = result
                self.save_df()
            else:
                print("Already done")
            
            print("==================================")



    
    def save_df(self):
        self.df.to_csv(os.path.join("data", "final", "factor_list_openai.csv"), index=False)




if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    research = OpenAIResearch()
    research.deep_research()

    # lit_review = LitReview()
    # lit_review.process_factor("weight", "high blood pressure")