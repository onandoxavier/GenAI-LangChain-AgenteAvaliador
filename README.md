# GenAI-LangChain-AgenteAvaliador

Agente que busca avaliações na API do exercício e classifica o sentimento de
cada uma como `POSITIVA` ou `NEGATIVA`.

## Configuração

1. Use Python 3.10 ou superior.
2. Instale o [Ollama](https://ollama.com/) e inicie o serviço local.
3. Baixe o modelo padrão:

   ```bash
   ollama pull qwen2.5:3b
   ```

4. Copie `.env.example` para `.env`. Se necessário, altere `OLLAMA_MODEL`
   para o nome de outro modelo que já tenha baixado no Ollama.
5. Instale as dependências:

   ```bash
   python -m pip install -r requirements.txt
   ```

O agente não precisa de chave de API: ele usa o Ollama localmente. O modelo
padrão é `qwen2.5:3b`.

## Execução

```bash
python agente_avaliacoes.py
```

Para cada avaliação retornada pela API, o programa imprime a classificação.

## Chain LangChain

O fluxo segue os componentes trabalhados nas aulas:

1. `ChatPromptTemplate` define as instruções de classificação e recebe a
   avaliação em `{review}`.
2. `ChatOllama` consulta o modelo local configurado.
3. `StrOutputParser` converte a resposta do modelo em texto.

Os três componentes são ligados com `|` (`prompt | model | output_parser`) e
executados com `chain.invoke({"review": review})`.
