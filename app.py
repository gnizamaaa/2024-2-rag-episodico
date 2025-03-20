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

def criaJson(message):
    # Esse daqui parece precisar falar a data, pq acho que o deepseek no terminal fica preso no tempo, mas se falar ele acerta, mesma coisa com horário
    # Seria interessante testar trocar o boolean para analisar se foi um relato do dia ou uma pergunta, a ideia seria usar ele para ver se usaria embendings ou não
    prompt = f"""Você é um sistema de processamento de linguagem natural. 
    Sua tarefa é analisar a entrada do usuário, extrair as seguintes informações e retornar um JSON com a estrutura:

    {{
        "date": "data inferida (formato AAAA-MM-DD)",
        "time": "hora inferida (formato HH:MM)",
        "period": "período do dia (manhã/tarde/noite)",
        "day_of_week": "dia da semana",
        "season": "estação do ano",
        "session": "evento principal",
        "description": "resumo da atividade"
    }}

    Além disso, retorne um booleano indicando se a data é passada.

    Instruções:
    1. Infira data/hora com base no contexto ou use a data atual se não especificado
    2. Determine o período do dia com base na hora
    3. Calcule dia da semana e estação do ano com base na data
    4. Identifique o evento principal (ex: "Almoço", "Trabalho", "Lazer")
    5. Para datas passadas, considere a data atual do sistema

    Exemplo de entrada: "Hoje meu dia foi fenomenal, almocei macarrão"
    Exemplo de saída: 
    (
        {{
            "date": "2024-05-15",
            "time": "12:00",
            "period": "tarde",
            "day_of_week": "Quarta-feira",
            "season": "Outono",
            "session": "Almoço",
            "description": "Dia fenomenal com almoço de macarrão"
        }},
        <boolean>
        False
        <boolean>

    Entrada: {message}
    )"""
    messages = []
    messages.append({'role': 'user', 'content': prompt})
    response: ChatResponse = chat(model='deepseek-r1:14b', messages = messages)
    print(response['message']['content'])

def ollama_stream_response(message, history):
    
    criaJson(message)
    """
    Função geradora que envia a mensagem para o Ollama
    e retorna a resposta em tempo real via streaming.
    """
    resp_rag = rag.answer_question(message)
    if not resp_rag:
        resp_rag = "Nenhuma informação encontrada."
    else:
        resp_rag = resp_rag[0]

    '''
    Tinha tentado com essa
    4. Associe a pergunta à sua resposta, ignorando os embeddings, no formato json a seguir:
        "date": "date of the question",
        "time": "time of the question",
        "period": "period of time of the question",
        "day_of_week": "day_of_week of the question",
        "season": "Season of the year",
        "session": "event",
        "description": "answer"
    '''
    # Retirei para testes: 4. Retorne os **3 registros mais relevantes**, incluindo datas/horários associados a cada um.
    prompt = f"""
    Com base nos embeddings gerados sobre a minha pergunta:  
    Embeddings : {resp_rag}

    - **Pergunta**: {message}  
    - **Critérios**:  
    1. Priorize embeddings que tenham **similaridade semântica** com palavras-chave ou contextos da pergunta.  
    2. Calcule a **similaridade de cosseno** entre a pergunta e os embeddings para ranqueamento.  
    3. Gere uma **resposta direta** à pergunta, combinando informações dos embeddings mais relevantes.
    4. Retorne os **3 registros mais relevantes**, incluindo datas/horários associados a cada um.
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
