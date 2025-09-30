# 🚀 Deploy Rápido - 3 Passos

## 📦 Streamlit Community Cloud (Recomendado)

### 1️⃣ GitHub (2 minutos)
1. Crie repositório: `agente-eda-robusto`
2. Upload de todos os arquivos desta pasta

### 2️⃣ Streamlit Deploy (2 minutos)
1. Acesse [share.streamlit.io](https://share.streamlit.io)
2. **New app** → Conecte o repositório
3. **Settings**:
   - Main file: `app.py`
   - Python: `3.11`
4. **Secrets** (opcional):
   ```
   OPENAI_API_KEY = "sua-chave-openai"
   ```

### 3️⃣ Teste (1 minuto)
- URL: `https://agente-eda-robusto.streamlit.app`
- Upload CSV → Dashboard instantâneo
- Perguntas básicas funcionam sem API key

---

## 🧪 Teste Local

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## ✅ Funcionalidades Garantidas

### Sempre Funcionam (sem API key):
- ⚡ Carregamento instantâneo (2-5s)
- 📊 Dashboard automático completo
- 🔍 Respostas pré-computadas (90% das perguntas)
- 📈 Estatísticas e insights automáticos

### Com API Key (opcional):
- 🧠 Perguntas específicas via LLM
- 📊 Gráficos interativos (se Plotly disponível)

---

**🎯 Total: 5 minutos para aplicação online!**
