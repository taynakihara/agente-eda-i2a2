# Instruções de Deploy - Agente EDA

## Opção 1: Streamlit Community Cloud (Recomendado)

### Passo 1: Preparar Repositório GitHub
1. Crie um novo repositório no GitHub
2. Faça upload dos seguintes arquivos:
   - `app.py`
   - `data_analysis_agent.py`
   - `requirements.txt`
   - `README.md`

### Passo 2: Deploy no Streamlit Cloud
1. Acesse [share.streamlit.io](https://share.streamlit.io)
2. Conecte sua conta GitHub
3. Selecione o repositório criado
4. Configure:
   - **Main file path**: `app.py`
   - **Python version**: 3.11
5. Adicione secrets (variáveis de ambiente):
   - `OPENAI_API_KEY`: sua chave da OpenAI

### Passo 3: Configurar Secrets
No painel do Streamlit Cloud, adicione:
```toml
OPENAI_API_KEY = "sua-chave-aqui"
```

## Opção 2: Hugging Face Spaces

### Arquivos Necessários
Todos os arquivos já estão prontos na pasta `/home/ubuntu/`:
- `app.py` - Interface Streamlit
- `data_analysis_agent.py` - Lógica do agente
- `requirements.txt` - Dependências
- `data/creditcard.csv` - Dataset de exemplo

### Configuração do Space
1. Crie novo Space no Hugging Face
2. Selecione SDK: **Streamlit**
3. Faça upload dos arquivos
4. Configure secrets:
   - `OPENAI_API_KEY`: sua chave da OpenAI

## Opção 3: Railway (Deploy Local)

### Comandos para Deploy
```bash
# Instalar Railway CLI
npm install -g @railway/cli

# Login
railway login

# Inicializar projeto
railway init

# Deploy
railway up
```

## URL de Exemplo
Após o deploy, a URL será algo como:
- Streamlit Cloud: `https://agente-eda-[username].streamlit.app`
- Hugging Face: `https://huggingface.co/spaces/[username]/agente-eda`
- Railway: `https://agente-eda-production.up.railway.app`

## Teste do Deploy
Para testar se o deploy funcionou:
1. Acesse a URL
2. Configure sua OpenAI API Key
3. Carregue um arquivo CSV ou use o dataset de exemplo
4. Faça uma pergunta: "Quantas linhas tem este dataset?"
5. Verifique se recebe uma resposta do agente
