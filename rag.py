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

            embedding = np.array(ollama.embed(
                model=self.model_name, input=a)["embeddings"])
            description_embedding = np.array(ollama.embed(
                model=self.model_name, input=description)["embeddings"])

            # Reduce weight of description by 80%
            description_embedding *= 0.2

            # Combine embeddings
            final_embedding = embedding + description_embedding

            self.collection.add(
                ids=[str(d)],
                embeddings=final_embedding.tolist(),
                documents=[f"{a}\n\n {description}"]
            )

    def answer_question(self, question):
        
        # TODO: Adicionar peso nos embeddings aqui.

        embeddingPergunta = ollama.embed(
            model=self.model_name, input=question)["embeddings"]
        memorias = self.collection.query(
            query_embeddings=embeddingPergunta, n_results=3)["documents"]
        return memorias

    def add_memory(self, memory: str):

        print("Adding memory")

        try:
            m = json.loads(memory)
        except (Exception) as e:
            print("Memória relevante, porém erro ao converter seu JSON")
            print(memory)
            print(e)

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

        try:
            self.collection.add(
                ids=[str(self.collection.count())],
                embeddings=final_embedding,
                documents=[memory]
            )
        except (Exception) as e:
            print("Erro ao adicionar memória")
            print(e)
