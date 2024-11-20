# Assistente Virtual para Análise de Documentos

Este projeto fornece um assistente virtual capaz de responder perguntas com base em documentos PDF enviados. Ele utiliza técnicas de Recuperação Aumentada por Recuperação (RAG) para gerar respostas contextuais.

## Requisitos

- Python 3.8 ou superior
- Poetry para gerenciamento de dependências e ambientes virtuais

## Instalação

1. Clone este repositório:
   ```bash
   git clone https://github.com/seu-usuario/seu-repositorio.git
   cd seu-repositorio
   ```

2. Instale o Poetry (se ainda não tiver instalado):
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

3. Instale as dependências do projeto:
   ```bash
   poetry install
   ```

4. Configure as variáveis de ambiente no arquivo `.env`:
   - **HUGGINGFACE_API_KEY**: A chave da API Hugging Face para acessar o modelo.
   
   Exemplo de conteúdo do arquivo `.env`:
   ```env
   HUGGINGFACE_API_KEY=seu_token_aqui
   ```

5. Para rodar o aplicativo localmente, execute:
   ```bash
   poetry run streamlit run main.py
   ```

   Isso iniciará a aplicação do Streamlit no seu navegador.

## Como Usar

1. Na barra lateral, envie arquivos PDF para análise.
2. Após o envio dos arquivos, digite uma pergunta no campo de entrada.
3. O assistente virtual irá processar a pergunta com base nos documentos enviados e fornecer uma resposta contextualizada.

## Estrutura do Projeto

- `main.py`: Interface de usuário e lógica de fluxo de documentos e perguntas.
- `model.py`: Definição do modelo de linguagem (HuggingFace, OpenAI, Ollama).
- `chain.py`: Configuração da cadeia de RAG (Recuperação Aumentada por Recuperação).
- `retriever.py`: Funções para carregar e dividir documentos PDF, criando o banco de vetores.

## Seção de Imagens

Aqui estão algumas imagens representando a interação do assistente:

1. **Tela inicial**:
   - Imagem de como a interface se apresenta ao usuário ao iniciar a aplicação.
   ![Tela Inicial](imagens/tela_inicial.png)

2. **Exemplo de Resposta**:
   - Imagem do resultado gerado após o envio de um documento e a pergunta ao assistente.
   ![Exemplo de Resposta](imagens/exemplo_resposta.png)

## Contribuindo

Sinta-se à vontade para abrir um pull request com melhorias ou correções de bugs. 
