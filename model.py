import pandas as pd
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
from langchain.chains import ConversationChain
from vllm import LLM, SamplingParams
import pandas as pd
from template import STAGE_1_TEMPLATE, STAGE_2_TEMPLATE, STAGE_3_TEMPLATE

# Function for error-handling non-strings
def safe_get(item, key):
    value = item.get(key, "")
    if pd.isna(value):
        return ""
    return str(value)

class LangchainCOTProcessor:
    def __init__(self, model_name):
        self.model_name = model_name
        self.gen_llm = LLM(model=self.model_name, task="generate")
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=7168 # NOTE: Max Token 갯수는 Naver Translate 파파고와의 연동성 고려하여 지정. 
        )

        self.stage1_memory = ConversationBufferMemory()
        self.stage2_memory = ConversationBufferMemory()
        self.stage3_memory = ConversationBufferMemory()

    def process_single_prompt(self, prompt, stage_memory):
        
        memory_variables = stage_memory.load_memory_variables({})
        history = memory_variables.get("history", "")

        if history:
            full_prompt = f"""Previous Context:
                                          {history}

                                          Current Input:
                                          {prompt}"""
        else:
            full_prompt = prompt

        text_output = self.gen_llm.generate([full_prompt], self.sampling_params)[0]
        generated_text = text_output.outputs[0].text

        stage_memory.save_context(
            {"input": prompt},
            {"output": generated_text}
        )

        return generated_text

    def clear_memories(self):
        """각 단계를 시작할 때마다 메모리 초기화"""
        self.stage1_memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")
        self.stage2_memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")
        self.stage3_memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")

    # NOTE: 아래 Function은 실제로 활용되지 않음.(Testing 용도)
    def process_cot(self, input_data):
        # Stage 1
        stage1_prompt = STAGE_1_TEMPLATE.format(
            disease_classification=input_data['disease_classification'],
            departments=input_data['departments'],
            systems=input_data['systems'],
            symptoms_complications=input_data['symptoms_complications'],
            gender=input_data['gender'],
            age=input_data['age'],
            bmi=input_data['bmi']
        )

        stage3_prompt = STAGE_3_TEMPLATE.format(top_item=input_data['top_item'])

        stage1_output, stage1_embed = self.process_single_prompt(stage1_prompt)

        # Stage 2
        stage2_prompt = f"{stage1_prompt}\n{stage1_output}\n\n{STAGE_2_TEMPLATE}"
        stage2_output, stage2_embed = self.process_single_prompt(stage2_prompt)

        # Stage 3
        stage3_prompt = f"{stage1_prompt}\n{stage1_output}\n{stage2_prompt}\n{stage2_output}\n\n{STAGE_3_TEMPLATE}"
        stage3_output, stage3_embed = self.process_single_prompt(stage3_prompt)

        return (stage1_embed, stage2_embed, stage3_embed)