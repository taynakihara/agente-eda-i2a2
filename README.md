# 🤖 Agente EDA - Análise Exploratória de Dados

Um agente autônomo desenvolvido com **LangChain** e **LangGraph** para realizar análises exploratórias automatizadas em qualquer arquivo CSV.

## 🎯 Funcionalidades

- **Análise Genérica**: Funciona com qualquer arquivo CSV
- **Interface Intuitiva**: Chat em linguagem natural via Streamlit
- **Visualizações Automáticas**: Gera gráficos quando apropriado
- **Memória Persistente**: Mantém contexto das análises realizadas
- **Conclusões Inteligentes**: Gera insights baseados nas análises

## 🚀 Como Usar

1. **Configure** sua OpenAI API Key
2. **Carregue** um arquivo CSV ou use o dataset de exemplo
3. **Faça perguntas** em linguagem natural sobre os dados
4. **Receba** análises detalhadas com gráficos e insights
5. **Gere conclusões** baseadas nas análises realizadas

## 📊 Tipos de Análise Suportados

### Análise Descritiva
- Estatísticas básicas (média, mediana, desvio padrão)
- Distribuições de variáveis
- Tipos de dados e valores nulos
- Intervalos e variabilidade

### Análise Exploratória
- Correlações entre variáveis
- Detecção de outliers
- Padrões temporais
- Agrupamentos (clusters)

### Visualizações
- Histogramas e distribuições
- Gráficos de dispersão
- Mapas de calor (correlações)
- Box plots
- Séries temporais

## 🛠️ Tecnologias Utilizadas

- **LangChain**: Framework para agentes de IA
- **LangGraph**: Orquestração de workflows complexos
- **Streamlit**: Interface web interativa
- **Plotly**: Visualizações interativas
- **Pandas**: Manipulação de dados
- **OpenAI GPT**: Modelo de linguagem

## 📦 Instalação Local

```bash
# Clonar repositório
git clone <url-do-repositorio>
cd agente-eda

# Instalar dependências
pip install -r requirements.txt

# Executar aplicação
streamlit run app.py
```

## 🔧 Configuração

### Variáveis de Ambiente
```bash
OPENAI_API_KEY=sua-chave-da-openai
```

### Dependências Principais
- streamlit==1.50.0
- langchain==0.3.27
- langgraph==0.6.8
- pandas==2.2.3
- plotly==5.24.1

## 📝 Exemplos de Perguntas

- "Quais são as estatísticas descritivas básicas do dataset?"
- "Qual a distribuição da variável Class?"
- "Existem outliers nos dados?"
- "Como as variáveis estão correlacionadas?"
- "Existem padrões temporais nos dados?"
- "Quais são suas conclusões sobre este dataset?"

## 🎓 Desenvolvido para

**Atividade Obrigatória - Agentes Autônomos**  
Institut d'Intelligence Artificielle Appliquée

### Requisitos Atendidos
✅ Agente genérico para qualquer CSV  
✅ Interface para perguntas em linguagem natural  
✅ Geração de gráficos automática  
✅ Memória e contexto persistente  
✅ Análises de EDA completas  
✅ Geração de conclusões  

## 📄 Licença

Este projeto foi desenvolvido para fins educacionais como parte de uma atividade acadêmica.

## 🔗 Links Úteis

- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [OpenAI API](https://platform.openai.com/docs/)
