import re
import gradio as gr
import random
import json
import sys
import ollama
import chromadb
import onnxruntime
import numpy as np
import rag
import asyncio
from ollama import chat
from ollama import ChatResponse
import time

MODEL_NAME = 'deepseek-r1:14b'
MODEL_PRE_NAME = 'deepseek-r1:14b'


def jsonAtual(session, description):
    date = time.strftime("%Y-%m-%d")
    current_time = time.strftime("%H:%M")
    period = "manhã" if current_time < "12:00" else "tarde" if current_time < "18:00" else "noite"
    day_of_week = time.strftime("%A")

    day_of_week_mapping = {
        "Monday": "Segunda-feira",
        "Tuesday": "Terça-feira",
        "Wednesday": "Quarta-feira",
        "Thursday": "Quinta-feira",
        "Friday": "Sexta-feira",
        "Saturday": "Sábado",
        "Sunday": "Domingo"
    }
    day_of_week = day_of_week_mapping.get(
        time.strftime("%A"), "Dia desconhecido")

    month = int(time.strftime("%m"))

    season = "Outono" if 3 <= month <= 5 else "Inverno" if 6 <= month <= 8 else "Primavera" if 9 <= month <= 11 else "Verão"

    json_today = {
        "date": date,
        "time": current_time,
        "period": period,
        "day_of_week": day_of_week,
        "season": season,
        "session": session,
        "description": description
    }
    return json.dumps(json_today)


def criaJson(message, model='deepseek-r1:14b'):
    # Esse daqui parece precisar falar a data, pq acho que o deepseek no terminal fica preso no tempo, mas se falar ele acerta, mesma coisa com horário
    # Seria interessante testar trocar o boolean para analisar se foi um relato do dia ou uma pergunta, a ideia seria usar ele para ver se usaria embendings ou não
    date = time.strftime("%Y-%m-%d")
    current_time = time.strftime("%H:%M")
    period = "manhã" if current_time < "12:00" else "tarde" if current_time < "18:00" else "noite"

    day_of_week_mapping = {
        "Monday": "Segunda-feira",
        "Tuesday": "Terça-feira",
        "Wednesday": "Quarta-feira",
        "Thursday": "Quinta-feira",
        "Friday": "Sexta-feira",
        "Saturday": "Sábado",
        "Sunday": "Domingo"
    }
    day_of_week = day_of_week_mapping.get(
        time.strftime("%A"), "Dia desconhecido")

    month = int(time.strftime("%m"))

    season = "Outono" if 3 <= month <= 5 else "Inverno" if 6 <= month <= 8 else "Primavera" if 9 <= month <= 11 else "Verão"

    json_today = {
        "date": date,
        "time": current_time,
        "period": period,
        "day_of_week": day_of_week,
        "season": season,
        "description": message
    }

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
    
    Sabendo que o momento atual é {current_time}, dia {date}, em um(a) {day_of_week}, {season}, você deve inferir as informações restantes com base no contexto da entrada do usuário.

    Instruções:
    1. Infira a data do evento referenciado na entrada com base no contexto ou use a data atual se não especificado na entrada
    2. Infira um horário aproximado com base no evento referenciado na entrada, ou use o horário atual se for um evento atual
    3. Determine o período do dia com base na hora (manhã, tarde, noite)
    4. Identifique o evento principal (ex: "Almoço", "Trabalho", "Lazer")
    5. Formate a saída para que siga a estrutura do JSON

    Exemplo de entrada: "Hoje meu dia foi fenomenal, almocei macarrão"
    Exemplo de saída: 
    (
        {{
            "date": {current_time},
            "time": {date},
            "period": {period},
            "day_of_week": {day_of_week},
            "season": {season},
            "session": "Almoço",
            "description": "Dia fenomenal com almoço de macarrão"
        }}

    Entrada: {message}
    )"""
    response: ChatResponse = chat(
        model=model, messages=[{'role': 'user', 'content': prompt}], stream=False)

    response = response['message']['content']
    response = response.split("</think>")[-1]

    try:
        jsonResp = re.sub(r'`(?:json)?\n?|`', '', response).strip()
        jsonResp = json.loads(jsonResp)
        mes = jsonResp['date'].split('-')[1]
        jsonResp['season'] = "Outono" if 3 <= int(mes) <= 5 else "Inverno" if 6 <= int(
            mes) <= 8 else "Primavera" if 9 <= int(mes) <= 11 else "Verão"
        jsonResp['description'] = message
    except:
        print("Erro ao converter JSON")

    return response


def ollama_stream_response(message, history):

    # Baixa os modelos (se não existirem)
    ollama.pull(MODEL_NAME)
    ollama.pull(MODEL_PRE_NAME)

    print("Modelos baixados")
    # Inicializa o Chroma
    ragClient = rag.ChromaManager()

    strBusca = criaJson(message, model=MODEL_PRE_NAME)
    resp_rag = ragClient.answer_question(strBusca)
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
    prompt_sistema = f"""Você é um assistente de IA. Use o histórico de conversa (enviado automaticamente após este prompt) apenas se for relevante para responder à pergunta."""
    prompt = f"""
    ### CONTEXTOS RECUPERADOS (RAG)
    {resp_rag}

    ### PERGUNTA
    {message}

    ### TAREFA
    1️⃣ Se a última mensagem **não for uma pergunta**, responda normalmente, ignorando contextos.
    2️⃣ Se for pergunta, selecione até **3 contextos relevantes**, priorizando similaridade semântica e incluindo datas/horários.
    3️⃣ Gere uma resposta **direta, coesa e concisa**, baseada somente nesses contextos.
    4️⃣ Se **nenhum** contexto for relevante, responda: “Não encontrei informações suficientes para responder a essa pergunta.”

    """

    response_text = "Pensando"
    saida: str = ""

    ollama_history = [{'role': 'system', 'content': prompt_sistema}]

    for entry in history:
        ollama_entry = {
            'role': entry['role'],
            'content': entry['content']
        }
        ollama_history.append(ollama_entry)

    ollama_history.append({'role': 'user', 'content': prompt})
    print(ollama_history)

    stream = chat(
        model=MODEL_NAME,  # Altere para o modelo desejado
        messages=ollama_history,
        stream=True
    )
    for chunk in stream:
        response_text += chunk['message']['content']
        if (response_text.find("</think>") != -1):
            saida += chunk['message']['content']
            yield saida

    print("Pensamento: " + response_text.split("</think>")[0])

    if (history):
        # Essa linha ainda precisa existir?
        ollama_history.append({'role': 'assistant', 'content': saida})

        promptClass = f"""Você é um sistema de processamento de linguagem natural.
        Sua tarefa é analisar a entrada do usuário em conjunto com a resposta do assistente e classificar se essa é uma memória digna de ser armazenada.
        Caso seja, você deve fazer uma descrição da memória e 2 keywords que representem a memória.
        Deverá então formatar sua resposta em um JSON com a seguinte estrutura:
        
        {{
            "session": "keywords",
            "description": "resumo da atividade"
        }}
        
        Se não for uma memória relevante, você deve informar que a memória não é relevante no campo sessions.
        Forme apenas uma memória, não é necessário formar várias memórias, caso tenha mais de um evento, faça uma memória mais longa contendo os eventos que julgar relevantes.
        
        São consideradas memórias relevantes eventos que sejam marcantes, importantes ou que possam ser úteis para o usuário em um futuro próximo, como por exemplo, um compromisso, uma tarefa, uma reunião, um jantar, uma refeição e outros.
        
        Entrada: {history}
        """
        print(history)

        response: ChatResponse = chat(
            model=MODEL_PRE_NAME, messages=[{'role': 'user', 'content': promptClass}], stream=False)

        response = response['message']['content']
        response = response.split("</think>")[-1]
        try:
            print("Resposta:" + response)
            jsonResp = re.sub(r'`(?:json)?\n?|`', '', response).strip()
            print("Resposta cortada:" + jsonResp)
            jsonMemoria = json.loads(jsonResp)

            relevante = True

            if jsonMemoria['session'].find("relevante") != -1:
                relevante = False

            if relevante == False:
                print("Memória não relevante")
            else:
                print(jsonMemoria)
                memoria = jsonAtual(
                    jsonMemoria['session'], jsonMemoria['description'])
                ragClient.add_memory(memoria)
                gr.Info("Memória salva")
        except (Exception) as e:
            print("Erro ao converter JSON, memória provavelmente não relevante")
            print(e)
            gr.Warning("Memória não salva")


demo = gr.ChatInterface(
    fn=ollama_stream_response,
    type="messages"
)

if __name__ == "__main__":
    # rag.init_chroma()
    # Altere para share=True se quiser um link público
    demo.launch(share=False)
