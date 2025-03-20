import gradio as gr
import random
import json
import sys
import ollama
import chromadb
import onnxruntime
import numpy as np
import rag
from ollama import AsyncClient
import asyncio
from ollama import chat
from ollama import ChatResponse
import time

# Configuração do cliente Ollama
client = AsyncClient()

def ollama_stream_response(message, history):
    """
    Função geradora que envia a mensagem para o Ollama
    e retorna a resposta em tempo real via streaming.
    """
    resp_rag = rag.answer_question(message)
    if not resp_rag:
        resp_rag = "Nenhuma informação encontrada."
    else:
        resp_rag = resp_rag[0]

    # Retirei para testes: 4. Retorne os **3 registros mais relevantes**, incluindo datas/horários associados a cada um.
    prompt = f"""
    Com base nos embeddings gerados sobre a minha pergunta:  
    Embeddings : {resp_rag}

    - **Pergunta**: {message}  
    - **Critérios**:  
    1. Priorize embeddings que tenham **similaridade semântica** com palavras-chave ou contextos da pergunta.  
    2. Calcule a **similaridade de cosseno** entre a pergunta e os embeddings para ranqueamento.  
    3. Gere uma **resposta direta** à pergunta, combinando informações dos embeddings mais relevantes.  
    4. Associe a pergunta à sua resposta, ignorando os embeddings, no formato json a seguir:
        "date": "date of the question",
        "time": "time of the question",
        "period": "period of time of the question",
        "day_of_week": "day_of_week of the question",
        "season": "Season of the year",
        "session": "event",
        "description": "answer"
    """
    
    # Mantém apenas as últimas 3 interações (cada interação tem [mensagem do usuário, resposta do assistente])
    last_messages = history[-3:] if len(history) >= 3 else history

    # Formata o histórico para a API do modelo
    messages = []
    #print(history)

    messages = []
    # for i in range(0, len(last_messages), 2):  # Pega de 2 em 2
    #     if i + 1 < len(last_messages):  # Garante que há um par
    #         user_msg = last_messages[i]["content"]
    #         bot_reply = last_messages[i + 1]["content"].split("</think>")[1].strip()
    #         print(user_msg)
    #         print(bot_reply)
            
    #         messages.append({"role": "user", "content": user_msg})
    #         messages.append({"role": "assistant", "content": bot_reply})


    messages.append({'role': 'user', 'content': prompt})

    response_text = "Pensando"
    
    stream = chat(
        model="deepseek-r1:14b",  # Altere para o modelo desejado
        messages=messages,
        stream=True
    )
    for chunk in stream:
        response_text += chunk['message']['content']
        yield response_text
        # yield response_text

def random_response(message, history):
    return rag.answer_question(message)[0]


demo = gr.ChatInterface(
    fn=ollama_stream_response,
    type="messages",
    theme='gstaff/sketch'   
)

if __name__ == "__main__": 
    rag.init_chroma()
    demo.launch(share=False)  # Altere para share=True se quiser um link público
