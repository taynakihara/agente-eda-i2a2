"""
Agente Inteligente Otimizado
Usa análise automática Python + LLM apenas para perguntas específicas
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import json
import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from auto_eda import AutoEDA

class SmartDataAgent:
    """Agente otimizado que combina análise automática com LLM inteligente"""
    
    def __init__(self, openai_api_key: str):
        self.llm = ChatOpenAI(
            model="gpt-4.1-mini",
            api_key=openai_api_key,
            temperature=0.1
        )
        self.auto_eda = AutoEDA()
        self.df = None
        self.eda_results = None
        self.chat_history = []
        
    def load_and_analyze_csv(self, file_path: str) -> Dict[str, Any]:
        """Carrega CSV e faz análise automática completa"""
        print("📁 Carregando arquivo CSV...")
        
        try:
            # Carregar dados
            self.df = pd.read_csv(file_path)
            print(f"✅ Arquivo carregado: {self.df.shape[0]} linhas, {self.df.shape[1]} colunas")
            
            # Análise automática completa
            print("🚀 Executando análise automática...")
            self.eda_results = self.auto_eda.analyze_dataset(self.df)
            
            return {
                'success': True,
                'message': f'Dataset carregado e analisado: {self.df.shape[0]} registros, {self.df.shape[1]} colunas',
                'eda_results': self.eda_results
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Erro ao carregar arquivo: {str(e)}',
                'eda_results': None
            }
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """Responde pergunta usando análise pré-computada + LLM quando necessário"""
        
        if self.df is None or self.eda_results is None:
            return {
                'response': 'Por favor, carregue um arquivo CSV primeiro.',
                'needs_computation': False,
                'chart_paths': []
            }
        
        # Verificar se a pergunta pode ser respondida com dados pré-computados
        precomputed_answer = self._check_precomputed_answers(question)
        
        if precomputed_answer:
            return {
                'response': precomputed_answer['answer'],
                'needs_computation': False,
                'chart_paths': precomputed_answer.get('charts', []),
                'data_source': 'precomputed'
            }
        
        # Se não pode ser respondida com dados pré-computados, usar LLM
        return self._answer_with_llm(question)
    
    def _check_precomputed_answers(self, question: str) -> Optional[Dict[str, Any]]:
        """Verifica se a pergunta pode ser respondida com análise pré-computada"""
        
        question_lower = question.lower()
        
        # Perguntas sobre informações básicas
        if any(keyword in question_lower for keyword in ['quantas linhas', 'quantos registros', 'tamanho', 'dimensão']):
            return {
                'answer': f"O dataset contém {self.df.shape[0]:,} linhas (registros) e {self.df.shape[1]} colunas (variáveis).",
                'charts': []
            }
        
        # Perguntas sobre colunas
        if any(keyword in question_lower for keyword in ['quais colunas', 'que variáveis', 'nomes das colunas']):
            cols_list = ', '.join(self.df.columns.tolist())
            return {
                'answer': f"As colunas do dataset são: {cols_list}",
                'charts': []
            }
        
        # Perguntas sobre tipos de dados
        if any(keyword in question_lower for keyword in ['tipos de dados', 'tipo das variáveis', 'dtypes']):
            numeric_cols = self.eda_results['data_types']['numeric_columns']
            categorical_cols = self.eda_results['data_types']['categorical_columns']
            
            answer = f"O dataset possui:\n"
            answer += f"• {len(numeric_cols)} variáveis numéricas: {', '.join(numeric_cols[:5])}{'...' if len(numeric_cols) > 5 else ''}\n"
            answer += f"• {len(categorical_cols)} variáveis categóricas: {', '.join(categorical_cols[:5])}{'...' if len(categorical_cols) > 5 else ''}"
            
            return {'answer': answer, 'charts': []}
        
        # Perguntas sobre valores ausentes
        if any(keyword in question_lower for keyword in ['valores ausentes', 'dados faltantes', 'missing', 'nulos']):
            missing_info = self.eda_results['missing_values']
            total_missing = missing_info['total_missing']
            
            if total_missing == 0:
                answer = "✅ Não há valores ausentes no dataset. Todos os campos estão preenchidos."
            else:
                cols_with_missing = missing_info['columns_with_missing']
                answer = f"⚠️ Encontrados {total_missing:,} valores ausentes em {len(cols_with_missing)} colunas: {', '.join(cols_with_missing[:3])}"
            
            charts = [chart for chart in self.eda_results['charts'] if 'missing' in chart]
            return {'answer': answer, 'charts': charts}
        
        # Perguntas sobre estatísticas descritivas
        if any(keyword in question_lower for keyword in ['estatísticas', 'descritivas', 'resumo', 'summary']):
            stats = self.eda_results['descriptive_stats']['numeric_summary']
            
            answer = "📊 **Estatísticas Descritivas Principais:**\n\n"
            
            # Pegar algumas colunas principais para mostrar
            main_cols = list(stats.keys())[:3]
            
            for col in main_cols:
                col_stats = stats[col]
                answer += f"**{col}:**\n"
                answer += f"• Média: {col_stats['mean']:.2f}\n"
                answer += f"• Mediana: {col_stats['50%']:.2f}\n"
                answer += f"• Desvio Padrão: {col_stats['std']:.2f}\n"
                answer += f"• Min/Max: {col_stats['min']:.2f} / {col_stats['max']:.2f}\n\n"
            
            charts = [chart for chart in self.eda_results['charts'] if 'distribution' in chart]
            return {'answer': answer, 'charts': charts}
        
        # Perguntas sobre correlações
        if any(keyword in question_lower for keyword in ['correlação', 'correlações', 'relacionamento', 'relação']):
            corr_info = self.eda_results['correlations']
            
            if 'message' in corr_info:
                answer = corr_info['message']
            else:
                strong_corrs = corr_info['strong_correlations'][:5]
                
                if not strong_corrs:
                    answer = "Não foram encontradas correlações significativas entre as variáveis numéricas."
                else:
                    answer = "🔗 **Correlações Mais Fortes Encontradas:**\n\n"
                    for corr in strong_corrs:
                        answer += f"• {corr['var1']} ↔ {corr['var2']}: {corr['correlation']:.3f} ({corr['strength']})\n"
            
            charts = [chart for chart in self.eda_results['charts'] if 'correlation' in chart]
            return {'answer': answer, 'charts': charts}
        
        # Perguntas sobre outliers
        if any(keyword in question_lower for keyword in ['outliers', 'valores atípicos', 'anomalias']):
            outliers_info = self.eda_results['outliers']
            
            answer = "🎯 **Análise de Outliers (Método IQR):**\n\n"
            
            for col, info in list(outliers_info.items())[:5]:
                outlier_count = info['iqr_outliers_count']
                outlier_percent = info['iqr_outliers_percent']
                
                if outlier_count > 0:
                    answer += f"• **{col}**: {outlier_count} outliers ({outlier_percent:.1f}%)\n"
                else:
                    answer += f"• **{col}**: Sem outliers detectados\n"
            
            charts = [chart for chart in self.eda_results['charts'] if 'boxplot' in chart]
            return {'answer': answer, 'charts': charts}
        
        # Perguntas sobre insights
        if any(keyword in question_lower for keyword in ['insights', 'descobertas', 'principais achados', 'resumo']):
            insights = self.eda_results['insights']
            recommendations = self.eda_results['recommendations']
            
            answer = "💡 **Principais Insights:**\n\n"
            for insight in insights:
                answer += f"{insight}\n"
            
            answer += "\n🔧 **Recomendações:**\n\n"
            for rec in recommendations:
                answer += f"{rec}\n"
            
            return {'answer': answer, 'charts': self.eda_results['charts'][:3]}
        
        return None
    
    def _answer_with_llm(self, question: str) -> Dict[str, Any]:
        """Usa LLM para responder perguntas específicas que precisam de análise customizada"""
        
        # Preparar contexto com dados pré-computados
        context = self._prepare_context_for_llm()
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Você é um especialista em análise de dados. Use as informações pré-computadas fornecidas para responder à pergunta do usuário.

DADOS PRÉ-COMPUTADOS:
{context}

INSTRUÇÕES:
1. Use APENAS as informações fornecidas no contexto
2. Se a pergunta requer análise específica não disponível, sugira como fazer
3. Seja preciso e direto
4. Use formatação markdown para melhor legibilidade
5. Se apropriado, mencione gráficos disponíveis

PERGUNTA: {question}

Resposta:"""),
            ("human", "{question}")
        ])
        
        try:
            response = self.llm.invoke(
                prompt.format_messages(
                    context=context,
                    question=question
                )
            )
            
            return {
                'response': response.content.strip(),
                'needs_computation': False,
                'chart_paths': self.eda_results['charts'],
                'data_source': 'llm_with_precomputed'
            }
            
        except Exception as e:
            return {
                'response': f"Erro ao processar pergunta: {str(e)}",
                'needs_computation': False,
                'chart_paths': []
            }
    
    def _prepare_context_for_llm(self) -> str:
        """Prepara contexto resumido para o LLM"""
        
        context = f"""
INFORMAÇÕES BÁSICAS:
- Dataset: {self.df.shape[0]} linhas, {self.df.shape[1]} colunas
- Colunas: {', '.join(self.df.columns.tolist())}
- Memória: {self.eda_results['basic_info']['memory_usage_mb']:.1f} MB

TIPOS DE DADOS:
- Numéricas ({len(self.eda_results['data_types']['numeric_columns'])}): {', '.join(self.eda_results['data_types']['numeric_columns'])}
- Categóricas ({len(self.eda_results['data_types']['categorical_columns'])}): {', '.join(self.eda_results['data_types']['categorical_columns'])}

VALORES AUSENTES:
- Total: {self.eda_results['missing_values']['total_missing']}
- Colunas com missing: {', '.join(self.eda_results['missing_values']['columns_with_missing'])}

CORRELAÇÕES PRINCIPAIS:
"""
        
        # Adicionar correlações principais
        if 'strong_correlations' in self.eda_results['correlations']:
            for corr in self.eda_results['correlations']['strong_correlations'][:3]:
                context += f"- {corr['var1']} ↔ {corr['var2']}: {corr['correlation']:.3f}\n"
        
        context += f"""
INSIGHTS PRINCIPAIS:
"""
        for insight in self.eda_results['insights'][:3]:
            context += f"- {insight}\n"
        
        context += f"""
GRÁFICOS DISPONÍVEIS:
"""
        for chart in self.eda_results['charts']:
            chart_name = chart.split('/')[-1].replace('.html', '').replace('_', ' ').title()
            context += f"- {chart_name}\n"
        
        return context
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Retorna dados para dashboard automático"""
        if self.eda_results is None:
            return {}
        
        return {
            'basic_info': self.eda_results['basic_info'],
            'insights': self.eda_results['insights'],
            'recommendations': self.eda_results['recommendations'],
            'charts': self.eda_results['charts'],
            'key_stats': {
                'total_missing': self.eda_results['missing_values']['total_missing'],
                'numeric_columns': len(self.eda_results['data_types']['numeric_columns']),
                'categorical_columns': len(self.eda_results['data_types']['categorical_columns']),
                'strong_correlations': len(self.eda_results['correlations'].get('strong_correlations', []))
            }
        }

def create_smart_agent(openai_api_key: str) -> SmartDataAgent:
    """Cria instância do agente inteligente"""
    return SmartDataAgent(openai_api_key)
