import json
import ollama
import chromadb
import numpy as np


class ChromaManager:
    def __init__(self, chroma_path="chroma/", model_name="granite-embedding:278m"):
        self.chroma_path = chroma_path
        self.model_name = model_name
        self.client = chromadb.PersistentClient(path=self.chroma_path)
        self.collection = self.client.get_or_create_collection(name="temporal")
        ollama.pull(self.model_name)

    def populateChroma(self, dataset_path='Dataset.txt'):
        with open(dataset_path, 'r') as teste:
            dataset = json.load(teste)

        for d, i in enumerate(dataset):
            
            final_embedding, the_rest, description = self.embed_weighted(i)

            self.collection.add(
                ids=[str(d)],
                embeddings=final_embedding.tolist(),
                documents=[f"{the_rest}\n\n {description}"]
            )

    def answer_question(self, question):

        embedding_pergunta, *_ = self.embed_weighted(question)

        memorias = self.collection.query(
            query_embeddings=embedding_pergunta, n_results=3)["documents"]
        return memorias

    def add_memory(self, memory: str):

        print("Adding memory")

        try:
            m = json.loads(memory)
        except (Exception) as e:
            print("Memória relevante, porém erro ao converter seu JSON")
            print(memory)
            print(e)
        
        embedding, *_ = self.embed_weighted(m)

        try:
            self.collection.add(
                ids=[str(self.collection.count())],
                embeddings=embedding,
                documents=[memory]
            )
        except (Exception) as e:
            print("Erro ao adicionar memória")
            print(e)

    def embed_weighted(self, m: dict):
        the_rest = f"""{{
            "date":"{m['date']}",
            "time":"{m['time']}",
            "period":"{m['period']}",
            "day_of_week":"{m['day_of_week']}",
            "season":"{m['season']}",
            "session":"{m['session']}"
            }}
            """

        description = m["description"]

        embedding = np.array(ollama.embed(
            model=self.model_name, input=the_rest)["embeddings"])
        description_embedding = np.array(ollama.embed(
            model=self.model_name, input=description)["embeddings"])

        # Reduce weight of description by 80%
        description_embedding *= 0.2

        # Combine embeddings
        final_embedding = embedding + description_embedding
        return final_embedding, the_rest, description