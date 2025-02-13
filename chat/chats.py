import os
import pandas as pd
from langchain.chains import LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq


def create_tenbagger_prompts():
    return [
        f"""\
英語で思考し、日本語で答えてください。
企業名: フィットイージー
この会社はテンバガーになるポテンシャルがあるかを教えて欲しいです。 
""",
        f"""\
代表者名は 國江仙嗣ですが、
上場当時（2024年）に、この経営者がこの企業を成長させる強い意志を持っていたか教えて。
を教えてください。
""",
        f"""\
- この経営者が目標を共有する優秀な部下がいたか
- 同じ業界内の競合に押し潰されなかった理由は何か
- この経営者の言動が一致しているかどうか
"""
    ]

class Chats:
    def __init__(self):
        self.groq = ChatGroq(model_name="deepseek-r1-distill-llama-70b", groq_api_key=os.environ["GROQ_API_KEY"])        
        system_prompt = "あなたは有能な投資分析アシスタントです。"
        self.memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history", return_messages=True)
        
        self.prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{human_input}"),
        ])

    def run(self):
        prompts = create_tenbagger_prompts()
        conversation = LLMChain(
            llm=self.groq,
            prompt=self.prompt_template,
            verbose=False,
            memory=self.memory,
        )
        file_counter = 1
        for prompt in prompts:
            response = conversation.predict(human_input=prompt)
            print(f"Prompt: {prompt}\nResponse: {response}\n{'='*80}")
            
            with open(f"result_{file_counter}.md", "w") as f:
                print(f"## Prompt:\n{prompt}\n\n### Response:\n{response}\n\n{'='*80}\n\n")
                f.write(response)

            file_counter += 1


if __name__ == '__main__':
    chats = Chats()
    chats.run()
