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
pip install chromadb
```
```sh
pip install ollama
```
```sh
pip install numpy
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

