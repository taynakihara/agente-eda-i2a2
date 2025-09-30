"""
Módulo de Análise Exploratória de Dados Automática
Realiza EDA completa imediatamente no upload do CSV, sem usar LLM
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
import os
from typing import Dict, List, Tuple, Any
import base64
import io

warnings.filterwarnings('ignore')

class AutoEDA:
    """Classe para análise exploratória automática de dados"""
    
    def __init__(self, output_dir: str = "/tmp/charts"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.chart_counter = 0
        
    def analyze_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Realiza análise completa do dataset automaticamente"""
        
        print("🔍 Iniciando análise automática...")
        
        analysis = {
            'basic_info': self._get_basic_info(df),
            'descriptive_stats': self._get_descriptive_stats(df),
            'missing_values': self._analyze_missing_values(df),
            'data_types': self._analyze_data_types(df),
            'correlations': self._analyze_correlations(df),
            'outliers': self._detect_outliers(df),
            'distributions': self._analyze_distributions(df),
            'categorical_analysis': self._analyze_categorical(df),
            'time_analysis': self._analyze_time_patterns(df),
            'charts': self._generate_charts(df),
            'insights': self._generate_insights(df),
            'recommendations': self._generate_recommendations(df)
        }
        
        print("✅ Análise automática concluída!")
        return analysis
    
    def _get_basic_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Informações básicas do dataset"""
        return {
            'shape': df.shape,
            'columns': list(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'dtypes': df.dtypes.to_dict(),
            'sample_data': df.head().to_dict('records')
        }
    
    def _get_descriptive_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Estatísticas descritivas completas"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return {'message': 'Nenhuma coluna numérica encontrada'}
        
        desc_stats = df[numeric_cols].describe()
        
        return {
            'numeric_summary': desc_stats.to_dict(),
            'skewness': df[numeric_cols].skew().to_dict(),
            'kurtosis': df[numeric_cols].kurtosis().to_dict(),
            'variance': df[numeric_cols].var().to_dict()
        }
    
    def _analyze_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Análise de valores ausentes"""
        missing_count = df.isnull().sum()
        missing_percent = (missing_count / len(df)) * 100
        
        return {
            'missing_count': missing_count.to_dict(),
            'missing_percent': missing_percent.to_dict(),
            'total_missing': missing_count.sum(),
            'columns_with_missing': missing_count[missing_count > 0].index.tolist()
        }
    
    def _analyze_data_types(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Análise dos tipos de dados"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        return {
            'numeric_columns': numeric_cols,
            'categorical_columns': categorical_cols,
            'datetime_columns': datetime_cols,
            'numeric_count': len(numeric_cols),
            'categorical_count': len(categorical_cols),
            'datetime_count': len(datetime_cols)
        }
    
    def _analyze_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Análise de correlações"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return {'message': 'Insuficientes colunas numéricas para correlação'}
        
        corr_matrix = df[numeric_cols].corr()
        
        # Encontrar correlações mais fortes
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.1:  # Apenas correlações significativas
                    corr_pairs.append({
                        'var1': col1,
                        'var2': col2,
                        'correlation': corr_val,
                        'strength': self._correlation_strength(abs(corr_val))
                    })
        
        # Ordenar por força da correlação
        corr_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        return {
            'correlation_matrix': corr_matrix.to_dict(),
            'strong_correlations': corr_pairs[:10],  # Top 10
            'highest_correlation': max(corr_pairs, key=lambda x: abs(x['correlation'])) if corr_pairs else None
        }
    
    def _correlation_strength(self, corr_val: float) -> str:
        """Classifica a força da correlação"""
        if corr_val >= 0.7:
            return 'Muito Forte'
        elif corr_val >= 0.5:
            return 'Forte'
        elif corr_val >= 0.3:
            return 'Moderada'
        elif corr_val >= 0.1:
            return 'Fraca'
        else:
            return 'Muito Fraca'
    
    def _detect_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detecção de outliers usando IQR e Z-score"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outliers_info = {}
        
        for col in numeric_cols:
            # Método IQR
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            iqr_outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            
            # Método Z-score
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            z_outliers = df[z_scores > 3]
            
            outliers_info[col] = {
                'iqr_outliers_count': len(iqr_outliers),
                'iqr_outliers_percent': (len(iqr_outliers) / len(df)) * 100,
                'z_outliers_count': len(z_outliers),
                'z_outliers_percent': (len(z_outliers) / len(df)) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
        
        return outliers_info
    
    def _analyze_distributions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Análise das distribuições das variáveis"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        distributions = {}
        
        for col in numeric_cols:
            data = df[col].dropna()
            
            # Teste de normalidade
            _, p_value = stats.normaltest(data)
            is_normal = p_value > 0.05
            
            distributions[col] = {
                'mean': data.mean(),
                'median': data.median(),
                'mode': data.mode().iloc[0] if len(data.mode()) > 0 else None,
                'std': data.std(),
                'min': data.min(),
                'max': data.max(),
                'range': data.max() - data.min(),
                'skewness': data.skew(),
                'kurtosis': data.kurtosis(),
                'is_normal': is_normal,
                'normality_p_value': p_value,
                'unique_values': data.nunique(),
                'unique_percent': (data.nunique() / len(data)) * 100
            }
        
        return distributions
    
    def _analyze_categorical(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Análise de variáveis categóricas"""
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        categorical_analysis = {}
        
        for col in categorical_cols:
            value_counts = df[col].value_counts()
            
            categorical_analysis[col] = {
                'unique_count': df[col].nunique(),
                'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
                'most_frequent_count': value_counts.iloc[0] if len(value_counts) > 0 else 0,
                'least_frequent': value_counts.index[-1] if len(value_counts) > 0 else None,
                'least_frequent_count': value_counts.iloc[-1] if len(value_counts) > 0 else 0,
                'value_counts': value_counts.head(10).to_dict(),
                'missing_count': df[col].isnull().sum()
            }
        
        return categorical_analysis
    
    def _analyze_time_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Análise de padrões temporais"""
        # Procurar colunas que podem ser temporais
        time_cols = []
        
        # Verificar colunas datetime
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        time_cols.extend(datetime_cols)
        
        # Verificar colunas que podem ser timestamps
        for col in df.columns:
            if 'time' in col.lower() or 'date' in col.lower():
                time_cols.append(col)
        
        time_analysis = {}
        
        for col in time_cols:
            if col in df.columns:
                try:
                    # Tentar converter para datetime se não for
                    if df[col].dtype != 'datetime64[ns]':
                        time_data = pd.to_datetime(df[col], errors='coerce')
                    else:
                        time_data = df[col]
                    
                    if time_data.notna().sum() > 0:
                        time_analysis[col] = {
                            'min_date': time_data.min(),
                            'max_date': time_data.max(),
                            'date_range_days': (time_data.max() - time_data.min()).days,
                            'missing_dates': time_data.isna().sum()
                        }
                except:
                    continue
        
        return time_analysis
    
    def _generate_charts(self, df: pd.DataFrame) -> List[str]:
        """Gera gráficos automáticos"""
        chart_paths = []
        
        # 1. Matriz de correlação
        chart_paths.append(self._create_correlation_heatmap(df))
        
        # 2. Distribuições das variáveis numéricas
        chart_paths.append(self._create_distributions_plot(df))
        
        # 3. Box plots para detecção de outliers
        chart_paths.append(self._create_boxplots(df))
        
        # 4. Gráfico de valores ausentes
        chart_paths.append(self._create_missing_values_plot(df))
        
        # 5. Análise de variáveis categóricas
        chart_paths.append(self._create_categorical_plots(df))
        
        return [path for path in chart_paths if path is not None]
    
    def _create_correlation_heatmap(self, df: pd.DataFrame) -> str:
        """Cria heatmap de correlação"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return None
        
        try:
            corr_matrix = df[numeric_cols].corr()
            
            fig = px.imshow(
                corr_matrix,
                title="Matriz de Correlação",
                color_continuous_scale="RdBu_r",
                aspect="auto"
            )
            
            chart_path = f"{self.output_dir}/correlation_heatmap.html"
            fig.write_html(chart_path)
            self.chart_counter += 1
            
            return chart_path
        except:
            return None
    
    def _create_distributions_plot(self, df: pd.DataFrame) -> str:
        """Cria gráficos de distribuição"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return None
        
        try:
            # Limitar a 6 colunas para não sobrecarregar
            cols_to_plot = numeric_cols[:6]
            
            fig = make_subplots(
                rows=2, cols=3,
                subplot_titles=[f"Distribuição de {col}" for col in cols_to_plot]
            )
            
            for i, col in enumerate(cols_to_plot):
                row = (i // 3) + 1
                col_pos = (i % 3) + 1
                
                fig.add_trace(
                    go.Histogram(x=df[col], name=col, showlegend=False),
                    row=row, col=col_pos
                )
            
            fig.update_layout(
                title="Distribuições das Variáveis Numéricas",
                height=600
            )
            
            chart_path = f"{self.output_dir}/distributions.html"
            fig.write_html(chart_path)
            self.chart_counter += 1
            
            return chart_path
        except:
            return None
    
    def _create_boxplots(self, df: pd.DataFrame) -> str:
        """Cria box plots para detecção de outliers"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return None
        
        try:
            # Limitar a 6 colunas
            cols_to_plot = numeric_cols[:6]
            
            fig = make_subplots(
                rows=2, cols=3,
                subplot_titles=[f"Box Plot - {col}" for col in cols_to_plot]
            )
            
            for i, col in enumerate(cols_to_plot):
                row = (i // 3) + 1
                col_pos = (i % 3) + 1
                
                fig.add_trace(
                    go.Box(y=df[col], name=col, showlegend=False),
                    row=row, col=col_pos
                )
            
            fig.update_layout(
                title="Box Plots - Detecção de Outliers",
                height=600
            )
            
            chart_path = f"{self.output_dir}/boxplots.html"
            fig.write_html(chart_path)
            self.chart_counter += 1
            
            return chart_path
        except:
            return None
    
    def _create_missing_values_plot(self, df: pd.DataFrame) -> str:
        """Cria gráfico de valores ausentes"""
        missing_data = df.isnull().sum()
        
        if missing_data.sum() == 0:
            return None
        
        try:
            fig = px.bar(
                x=missing_data.index,
                y=missing_data.values,
                title="Valores Ausentes por Coluna",
                labels={'x': 'Colunas', 'y': 'Quantidade de Valores Ausentes'}
            )
            
            chart_path = f"{self.output_dir}/missing_values.html"
            fig.write_html(chart_path)
            self.chart_counter += 1
            
            return chart_path
        except:
            return None
    
    def _create_categorical_plots(self, df: pd.DataFrame) -> str:
        """Cria gráficos para variáveis categóricas"""
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        if len(categorical_cols) == 0:
            return None
        
        try:
            # Pegar primeira coluna categórica
            col = categorical_cols[0]
            value_counts = df[col].value_counts().head(10)
            
            fig = px.bar(
                x=value_counts.index,
                y=value_counts.values,
                title=f"Distribuição da Variável {col}",
                labels={'x': col, 'y': 'Frequência'}
            )
            
            chart_path = f"{self.output_dir}/categorical_distribution.html"
            fig.write_html(chart_path)
            self.chart_counter += 1
            
            return chart_path
        except:
            return None
    
    def _generate_insights(self, df: pd.DataFrame) -> List[str]:
        """Gera insights automáticos sobre os dados"""
        insights = []
        
        # Insight sobre tamanho do dataset
        insights.append(f"📊 Dataset contém {df.shape[0]:,} registros e {df.shape[1]} variáveis")
        
        # Insight sobre valores ausentes
        missing_total = df.isnull().sum().sum()
        if missing_total == 0:
            insights.append("✅ Não há valores ausentes no dataset")
        else:
            missing_percent = (missing_total / (df.shape[0] * df.shape[1])) * 100
            insights.append(f"⚠️ {missing_total:,} valores ausentes ({missing_percent:.1f}% do total)")
        
        # Insight sobre tipos de dados
        numeric_count = len(df.select_dtypes(include=[np.number]).columns)
        categorical_count = len(df.select_dtypes(include=['object', 'category']).columns)
        insights.append(f"🔢 {numeric_count} variáveis numéricas e {categorical_count} categóricas")
        
        # Insight sobre duplicatas
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            insights.append(f"🔄 {duplicates:,} registros duplicados encontrados")
        else:
            insights.append("✅ Não há registros duplicados")
        
        # Insight sobre variabilidade
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            high_variance_cols = []
            for col in numeric_cols:
                cv = df[col].std() / df[col].mean() if df[col].mean() != 0 else 0
                if cv > 1:
                    high_variance_cols.append(col)
            
            if high_variance_cols:
                insights.append(f"📈 Variáveis com alta variabilidade: {', '.join(high_variance_cols[:3])}")
        
        return insights
    
    def _generate_recommendations(self, df: pd.DataFrame) -> List[str]:
        """Gera recomendações automáticas"""
        recommendations = []
        
        # Recomendações sobre valores ausentes
        missing_cols = df.columns[df.isnull().any()].tolist()
        if missing_cols:
            recommendations.append(f"🔧 Considere tratar valores ausentes em: {', '.join(missing_cols[:3])}")
        
        # Recomendações sobre outliers
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_cols = []
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
            if len(outliers) > len(df) * 0.05:  # Mais de 5% outliers
                outlier_cols.append(col)
        
        if outlier_cols:
            recommendations.append(f"⚠️ Investigar outliers em: {', '.join(outlier_cols[:3])}")
        
        # Recomendações sobre correlações
        if len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols].corr()
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.8:
                        high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
            
            if high_corr_pairs:
                recommendations.append("🔗 Variáveis altamente correlacionadas detectadas - considere redução de dimensionalidade")
        
        # Recomendações sobre balanceamento (se houver variável target binária)
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64'] and df[col].nunique() == 2:
                value_counts = df[col].value_counts()
                ratio = value_counts.min() / value_counts.max()
                if ratio < 0.1:
                    recommendations.append(f"⚖️ Variável '{col}' está desbalanceada - considere técnicas de balanceamento")
                    break
        
        return recommendations
