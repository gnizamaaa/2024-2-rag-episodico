import json
import sys
import ollama
import chromadb
import numpy as np

collection = None

def init_chroma():
    teste = open('Dataset.txt', 'r')
    dataset = json.load(teste)

    client = chromadb.PersistentClient(path="chroma/")
    collection = client.get_or_create_collection(name="temporal")

    ollama.pull("granite-embedding")

    for d, i in enumerate(dataset):
        a = f"""{{
        "date":"{i['date']}",
        "time":"{i['time']}",
        "period":"{i['period']}",
        "day_of_week":"{i['day_of_week']}",
        "season":"{i['season']}",
        "session":"{i['session']}"
        }}
        """
        
        description = i["description"]

        embedding = np.array(ollama.embed(model="granite-embedding:278m", input=a)["embeddings"])
        description_embedding = np.array(ollama.embed(model="granite-embedding:278m", input=description)["embeddings"])

        # Reduzir peso da description em: reduzir 80%
        description_embedding *= 0.2

        # Combinar embeddings
        final_embedding = embedding + description_embedding

        collection.add(
            ids=[str(d)],
            embeddings=final_embedding.tolist(),
            documents=[f"{a}\n\n {description}"]
        )


def answer_question(question):
    embeddingPergunta = ollama.embed(model="granite-embedding:278m", input=question)["embeddings"]
    memorias = collection.query(query_embeddings=embeddingPergunta, n_results=6)["documents"]
    return memorias