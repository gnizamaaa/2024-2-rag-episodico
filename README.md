# 2024-2-rag-episodico

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

Para executar o aplicativo, utilize o seguinte comando na pasta que contém o arquivo app.py:

```sh
 python .\app.py  
```

Após um curto período de tempo, aparecerá um endereço no localhost ao acessá-lo será possível interagir com o modelo. 


## Testes com a base de dados

Aqui estão alguns testes para serem realizados com a base de dados já disponível:

Q: "O que eu almocei no dia 26/02 as 12:00?"

R: "No dia 26 de fevereiro, às 12:20, você almoçou em um pequeno restaurante italiano no centro da cidade. A comida incluiu massa fresca e cozida al dente acompanhada por vinho tinto encorpado, além de uma conversa agradável com um velho amigo."

