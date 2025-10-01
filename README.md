# 🤖 Agente de Análise Exploratória de Dados (E.D.A.) - Versão Groq

## 📋 Descrição
Esta aplicação Streamlit é um **agente inteligente** que permite análise exploratória completa de **qualquer arquivo CSV** de forma automática e interativa, powered by **Groq AI**. A ferramenta foi desenvolvida para atender aos requisitos da atividade obrigatória do Institut d'Intelligence Artificielle Appliquée.

## ⚡ Por que Groq?

**Groq** é uma plataforma de IA que oferece vantagens significativas sobre outras APIs:

- **🚀 Velocidade Extrema**: Até 10x mais rápida que OpenAI
- **💰 Muito Econômica**: Tier gratuito generoso com milhares de tokens
- **🧠 Modelos Avançados**: Llama 3 70B, Mixtral 8x7B, Gemma 7B
- **🔒 Confiável**: Infraestrutura robusta e estável
- **🌟 Fácil de Usar**: API simples e bem documentada

## 🚀 Funcionalidades Principais

### 📋 Visão Geral
- **Informações básicas** do dataset (linhas, colunas, tamanho)
- **Tipos de dados** e identificação automática
- **Estatísticas descritivas** completas
- **Detecção de valores nulos** e únicos

### 📊 Distribuições
- **Histogramas automáticos** para variáveis numéricas
- **Gráficos de barras** para variáveis categóricas
- **Visualizações com alto contraste** para excelente legibilidade
- **Filtragem automática de outliers** para melhor visualização

### 🔍 Correlações
- **Matriz de correlação** interativa com heatmap
- **Identificação automática** de correlações significativas
- **Classificação por força** da correlação (forte, moderada, fraca)
- **Análise de dependências** entre variáveis

### 📈 Tendências
- **Detecção automática** de colunas temporais
- **Análise de tendências temporais** interativa
- **Padrões em variáveis categóricas**
- **Valores mais e menos frequentes**

### ⚠️ Anomalias
- **Detecção automática de outliers** usando método IQR
- **Visualização com boxplots** de alta qualidade
- **Estatísticas detalhadas** de anomalias por variável
- **Percentuais e limites** claramente definidos

### 🤖 Consulta Inteligente com IA Groq
- **Múltiplos modelos disponíveis**:
  - 🦙 **Llama 3.3 70B** (Recomendado) - Mais inteligente
  - 🦙 **Llama 3.1 8B** - Mais rápido
  - 🧠 **GPT OSS 120B** - Mais poderoso
  - 🧠 **GPT OSS 20B** - Eficiente
- **Configurações avançadas** personalizáveis
- **Contexto automático** com estatísticas do dataset
- **Eficiência de custos** - API chamada apenas quando solicitado

## 🛠️ Como Executar Localmente

### Pré-requisitos
- Python 3.7+
- pip (gerenciador de pacotes Python)

### Instalação
1. Clone ou baixe este repositório
2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
3. Execute a aplicação:
   ```bash
   # Versão básica
   streamlit run app.py
   
   # Versão avançada com múltiplos modelos
   streamlit run app_groq_advanced.py
   ```
4. Acesse no navegador: `http://localhost:8501`

## ☁️ Deploy no Streamlit Cloud

### Passo a Passo
1. **Fork este repositório** no GitHub
2. **Acesse** [share.streamlit.io](https://share.streamlit.io)
3. **Conecte sua conta** do Streamlit Cloud ao GitHub
4. **Selecione este repositório** para deploy
5. **Configure** o arquivo principal como `app.py`
6. **Deploy automático** será realizado

### URL de Acesso
Após o deploy, sua aplicação estará disponível em:
`https://[nome-do-app]-[seu-usuario].streamlit.app`

## 🔑 Uso da API Groq

### Como Obter sua Chave
1. Acesse [console.groq.com](https://console.groq.com)
2. Faça login ou **crie uma conta gratuita**
3. Navegue até **API Keys**
4. **Crie uma nova chave** secreta
5. **Cole a chave** na interface da aplicação

### Características de Eficiência
- ✅ **Consultas sob demanda** - API chamada apenas quando solicitado
- ✅ **Contexto otimizado** - Envia apenas estatísticas relevantes
- ✅ **Controle de custos** - Usuário insere sua própria chave
- ✅ **Sem armazenamento** - Chave não é salva ou compartilhada
- ✅ **Tier gratuito generoso** - Milhares de tokens gratuitos por mês

## 📁 Estrutura dos Arquivos

```
streamlit_app/
├── app.py                    # Aplicação principal Streamlit (Groq básico)
├── app_groq_advanced.py      # Versão avançada com múltiplos modelos
├── requirements.txt          # Dependências Python (com groq)
├── README_GROQ.md           # Este arquivo
└── README.md                # README original
```

## 🎯 Casos de Uso

### Para Cientistas de Dados
- **Análise exploratória ultra-rápida** de novos datasets
- **Identificação automática** de padrões e anomalias
- **Geração de insights** com IA de última geração

### Para Analistas de Negócios
- **Compreensão intuitiva** de dados complexos
- **Visualizações profissionais** prontas para apresentação
- **Perguntas em linguagem natural** com respostas instantâneas

### Para Estudantes
- **Aprendizado prático** de análise de dados
- **Exemplos visuais** de conceitos estatísticos
- **Ferramenta educacional** com IA avançada

## 🔧 Tecnologias Utilizadas

- **Streamlit** - Framework web para Python
- **Pandas** - Manipulação e análise de dados
- **Matplotlib** - Visualizações estáticas
- **Seaborn** - Visualizações estatísticas avançadas
- **NumPy** - Computação numérica
- **Groq** - API de IA ultra-rápida

## 📊 Modelos Disponíveis

### 🦙 Llama 3.3 70B (Recomendado)
- **Melhor qualidade** de resposta
- **Raciocínio avançado** para análises complexas
- **Ideal para** insights profundos
- **Contexto**: 131K tokens

### 🦙 Llama 3.1 8B (Rápido)
- **Velocidade máxima** de resposta
- **Boa qualidade** para perguntas simples
- **Ideal para** consultas rápidas
- **Contexto**: 131K tokens

### 🧠 GPT OSS 120B (Poderoso)
- **Modelo mais poderoso** disponível
- **Excelente para** análises complexas
- **Ideal para** tarefas avançadas
- **Contexto**: 131K tokens

### 🧠 GPT OSS 20B (Eficiente)
- **Equilibrio** entre potência e eficiência
- **Boa para** diversos tipos de análise
- **Ideal para** uso geral
- **Contexto**: 131K tokens

## 🎨 Design e Usabilidade

### Alto Contraste
- **Fundo escuro** (#0E1117) para reduzir fadiga visual
- **Texto branco** para máxima legibilidade
- **Cores vibrantes** (cyan, coral) para destacar dados
- **Grid sutil** para orientação visual

### Interface Intuitiva
- **Abas organizadas** por tipo de análise
- **Upload simples** de arquivos CSV
- **Seleção de modelos** IA
- **Configurações avançadas** opcionais
- **Feedback visual** em tempo real
- **Responsivo** para diferentes dispositivos

## 🏆 Diferenciais da Versão Groq

1. **Ultra-Rápida** - Respostas em segundos
2. **Econômica** - Tier gratuito muito generoso
3. **Múltiplos Modelos** - Escolha o melhor para sua necessidade
4. **Configurável** - Ajuste temperatura, tokens, prompts
5. **Confiável** - Infraestrutura robusta
6. **Moderna** - Modelos de última geração

## 🆚 Comparação: Groq vs OpenAI

| Característica | Groq | OpenAI |
|---|---|---|
| **Velocidade** | ⚡ Até 10x mais rápida | 🐌 Padrão |
| **Custo** | 💰 Muito econômica | 💸 Mais cara |
| **Tier Gratuito** | 🎁 Muito generoso | 🎁 Limitado |
| **Modelos** | 🧠 Llama 3.3, GPT OSS | 🧠 GPT-3.5, GPT-4 |
| **Qualidade** | ⭐ Excelente | ⭐ Excelente |
| **Facilidade** | ✅ Muito fácil | ✅ Fácil |

## 📈 Performance

- **Tempo de resposta**: < 2 segundos (típico)
- **Throughput**: Milhares de tokens por segundo
- **Disponibilidade**: 99.9% uptime
- **Latência**: Ultra-baixa

---

**Desenvolvido com ❤️ e powered by ⚡ Groq AI para análise inteligente de dados**
