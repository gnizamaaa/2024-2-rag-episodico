# 2024-2-rag-episodico
Antonio Lucio Braga Secchin, Bruno Lopes Altoé, Rhuan Garcia de Assis Teixeira

"Explorando Retrieval-Augmented-Generation como solucão para Memória Episódica"

## Descrição

Desenvolvimento de uma memória episódica para sistemas GPT, armazenando experiências em um banco de dados de embeddings com marcações temporais para recuperação e uso contextual futuro. Explora alternativas como fine-tuning periódico para aprimorar a recuperação da memória.

## Configuração para Desenvolvimento

Para criar o ambiente de desenvolvimento primeiro instale as dependências necessárias:

Entre no [site do *Ollama*](https://ollama.com/download) e instale-o.

Agora, certifique-se que está utilizando uma versão de Python 3.10 ou superior, instale as seguintes dependências:

```sh
pip install --upgrade gradio
```
```sh
pip install chromadb ollama numpy
``` 

Agora com as dependências instaladas, clone o repositório:

```sh
git clone https://github.com/gnizamaaa/2024-2-rag-episodico.git
```

Se tudo ocorreu certo, será possível executar o aplicativo.

### Baixando os modelos

Este passo não é necessário, porém se não for executado a primeira requisição do agente demorará um longo período de tempo já que baixará os modelos.

Neste trabalho utilizamos de dois modelos: o deepseek-r1:14b e o granite-embedding:278m

```sh
ollama pull deepseek-r1:14b
```

```sh
ollama pull granite-embedding:278m
```

### Baixando a base de dados

A base de dados já está inclusa no github do aplicativo, então ao clonar o repositório, ela já está disponível.

## Executando o aplicativo

Em um terminal a parte, inicie o ollama:

```sh
 ollama serve
```

Caso queira utilizar nosso dataset de memórias usado nos testes da aplicação, basta executar o aplicativo utilizando o comando abaixo, que irá preencher o banco de dados com nossas memórias
```sh
 python app.py test  
```

Esse comando irá preencher a base de conhecimento com nossas memórias e após isso aparecerá um endereço no localhost ao acessá-lo será possível interagir com o modelo. 

Para apenas executar o aplicativo, utilize o seguinte comando na pasta que contém o arquivo app.py:
```sh
 python .\app.py  
```

Após um curto período de tempo, aparecerá um endereço no localhost ao acessá-lo será possível interagir com o modelo. 


## Exemplos de entrada

Aqui estão algumas entradas para serem perguntadas com a base de dados já disponível:

Q: "O que eu almocei no dia 26/02 as 12:00?"

R: "No dia 26 de fevereiro, às 12:20, você almoçou em um pequeno restaurante italiano no centro da cidade. A comida incluiu massa fresca e cozida al dente acompanhada por vinho tinto encorpado, além de uma conversa agradável com um velho amigo."

Q: "Como foi a manhã do dia 2025-02-27?"

R: "No período da manhã do dia 2025-02-27, o despertador tocou com uma melodia suave, e o corpo se moveu com energia renovada após dormir bem. O café da manhã foi preparado com calma, incluindo ovos mexidos, torradas integrais e uma fruta fresca. A leitura do jornal acompanhou os primeiros raios de sol, misturando notícias ao aroma do café. A reunião de trabalho seguiu em um ambiente concentrado, com discussões detalhadas sobre o projeto."

## Explorando Retrieval-Augmented-Generation como solucão para Memória Episódica

É disponibilizado no repositório github um artigo chamado "Explorando Retrieval-Augmented-Generation como solucão para Memória Episódica", tanto em formato LaTeX como em PDF, apresentando e explicando a abordagem utilizada para a criação desse aplicativo, junto com trabalhos correlatos, os testes realizados e os resultados obtidos.
