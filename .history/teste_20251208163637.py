# salvar como criar_notebook_projeto_core.py e executar: python criar_notebook_projeto_core.py
import nbformat as nbf
from textwrap import dedent

nb = nbf.v4.new_notebook()
cells = []

cells.append(
    nbf.v4.new_markdown_cell(
        "# Projeto Core — Análises Completas\n\n"
        "Notebook com EDA avançada, visualizações, correlações, testes estatísticos, "
        "feature engineering e modelos preditivos para CPC, CTR, CPA e CVR.\n\n"
        "**Arquivos usados:** `/mnt/data/campaign_performance.csv`, "
        "`/mnt/data/creative_details.csv`, `/mnt/data/device_info.csv`"
    )
)

cells.append(
    nbf.v4.new_code_cell(
        dedent(
            """\
# Carregamento de bibliotecas e dados
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest

# Configurações
pd.set_option('display.max_columns', None)
np.random.seed(42)

# Carrega datasets
campaign = pd.read_csv('/mnt/data/campaign_performance.csv', parse_dates=['date'])
creative = pd.read_csv('/mnt/data/creative_details.csv')
device = pd.read_csv('/mnt/data/device_info.csv')

print('campaign:', campaign.shape)
print('creative:', creative.shape)
print('device:', device.shape)

# Mostrar primeiras linhas
campaign.head()
"""
        )
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        "## 1) Limpeza e criação de métricas básicas\nCriamos CTR, CPC, CPA, CVR e verificamos valores extremos e nulos."
    )
)

cells.append(
    nbf.v4.new_code_cell(
        dedent(
            """\
# Métricas básicas
campaign = campaign.copy()
campaign['CTR'] = campaign['clicks'] / campaign['impressions']
campaign['CPC'] = campaign['spend'] / campaign['clicks'].replace(0, np.nan)
campaign['CPA'] = campaign['spend'] / campaign['conversions'].replace(0, np.nan)
campaign['CVR'] = campaign['conversions'] / campaign['clicks'].replace(0, np.nan)  # conversão por clique
campaign[['CTR','CPC','CPA','CVR']].describe()
"""
        )
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        "### Tratamento de valores infinitos ou NaN\nSubstituímos inf/NaN onde necessário e criamos flags para muitas conversões/clicks."
    )
)

cells.append(
    nbf.v4.new_code_cell(
        dedent(
            """\
# Substituir inf/-inf por NaN e tratar
campaign.replace([np.inf, -np.inf], np.nan, inplace=True)

# Flags e filtros
campaign['has_clicks'] = campaign['clicks'] > 0
campaign['has_conversions'] = campaign['conversions'] > 0

# Estatísticas rápidas
campaign[['CTR','CPC','CPA','CVR','clicks','conversions']].agg(['median','mean','std','max']).T
"""
        )
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        "## 2) Merge com metadados criativos e device\nUnimos `creative_details` e `device_info` para enriquecer a base."
    )
)

cells.append(
    nbf.v4.new_code_cell(
        dedent(
            """\
df = campaign.merge(creative, on='creative_id', how='left')
df = df.merge(device, on='adset_id', how='left')
print('merged shape:', df.shape)
df.head()
"""
        )
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        "## 3) Análises aprofundadas (agrupamentos e tabela pivô)\nKPIs por canal, formato e tom."
    )
)

cells.append(
    nbf.v4.new_code_cell(
        dedent(
            """\
# KPI por canal
kpi_channel = df.groupby('channel').agg({
    'impressions':'sum','clicks':'sum','conversions':'sum','spend':'sum','revenue':'sum'
}).reset_index()
kpi_channel['CTR'] = kpi_channel['clicks']/kpi_channel['impressions']
kpi_channel['CPC'] = kpi_channel['spend']/kpi_channel['clicks']
kpi_channel['CPA'] = kpi_channel['spend']/kpi_channel['conversions']
kpi_channel['CVR'] = kpi_channel['conversions']/kpi_channel['clicks']

kpi_channel
"""
        )
    )
)

cells.append(
    nbf.v4.new_code_cell(
        dedent(
            """\
# KPI por formato de criativo
kpi_format = df.groupby('creative_format').agg({
    'impressions':'sum','clicks':'sum','conversions':'sum','spend':'sum','revenue':'sum'
}).reset_index()
kpi_format['CTR'] = kpi_format['clicks']/kpi_format['impressions']
kpi_format['CPC'] = kpi_format['spend']/kpi_format['clicks']
kpi_format['CPA'] = kpi_format['spend']/kpi_format['conversions']
kpi_format['CVR'] = kpi_format['conversions']/kpi_format['clicks']

kpi_format
"""
        )
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        "## 4) Gráficos mais avançados\nUsamos matplotlib (sem escolha explícita de cores)."
    )
)

cells.append(
    nbf.v4.new_code_cell(
        dedent(
            """\
# Série temporal de impressões e conversões por canal (agregado por semana)
df_week = df.set_index('date').groupby(['channel', pd.Grouper(freq='W')]).agg({
    'impressions':'sum','conversions':'sum','spend':'sum'
}).reset_index()

channels = df_week['channel'].unique()
plt.figure(figsize=(12,6))
for ch in channels:
    subset = df_week[df_week['channel']==ch]
    plt.plot(subset['date'], subset['conversions'], label=ch)
plt.legend()
plt.title('Conversões semanais por canal')
plt.xlabel('Semana')
plt.ylabel('Conversões')
plt.show()
"""
        )
    )
)

cells.append(
    nbf.v4.new_code_cell(
        dedent(
            """\
# Scatter: spend vs conversions (por creative_format)
formats = df['creative_format'].unique()
plt.figure(figsize=(10,6))
for f in formats:
    s = df[df['creative_format']==f]
    plt.scatter(s['spend'], s['conversions'], alpha=0.4, label=f)
plt.legend()
plt.xlabel('Spend (R$)')
plt.ylabel('Conversions')
plt.title('Spend vs Conversions por formato de criativo')
plt.show()
"""
        )
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        "## 5) Correlações entre variáveis numéricas\nMatriz de correlação e análise de relações importantes."
    )
)

cells.append(
    nbf.v4.new_code_cell(
        dedent(
            """\
numeric_cols = ['impressions','clicks','spend','conversions','revenue','CTR','CPC','CPA','CVR']
corr = df[numeric_cols].corr()
corr
"""
        )
    )
)

cells.append(
    nbf.v4.new_code_cell(
        dedent(
            """\
# Heatmap simples com matplotlib (sem estilo externo)
plt.figure(figsize=(8,6))
plt.imshow(corr, interpolation='nearest')
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title('Matriz de Correlação (numérica)')
plt.show()
"""
        )
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        "## 6) Testes Estatísticos\n- t-test: CTR de vídeo vs imagem\n- Teste de proporções: conversão entre tones (emocional vs tecnico)\n- Chi-square: formato x ter conversões (sim/não)"
    )
)

cells.append(
    nbf.v4.new_code_cell(
        dedent(
            """\
# t-test: CTR vídeo vs imagem
ctr_video = df[df['creative_format']=='video']['CTR'].dropna()
ctr_image = df[df['creative_format']=='image']['CTR'].dropna()
tstat, pval = stats.ttest_ind(ctr_video, ctr_image, equal_var=False, nan_policy='omit')
print('t-stat:', tstat, 'p-val:', pval)
"""
        )
    )
)

cells.append(
    nbf.v4.new_code_cell(
        dedent(
            """\
# Proporção: CVR (conversions per click) emocional vs tecnico
emo = df[df['tone']=='emocional']
tec = df[df['tone']=='tecnico']

count = np.array([emo['conversions'].sum(), tec['conversions'].sum()])
nobs = np.array([emo['clicks'].sum(), tec['clicks'].sum()])

zstat, pval_prop = proportions_ztest(count, nobs)
print('z-stat:', zstat, 'p-val:', pval_prop)
"""
        )
    )
)

cells.append(
    nbf.v4.new_code_cell(
        dedent(
            """\
# Chi-square: formato x tem_conversao (sim/não)
cont_table = pd.crosstab(df['creative_format'], df['has_conversions'])
from scipy.stats import chi2_contingency
chi2, p_chi, dof, ex = chi2_contingency(cont_table)
print('chi2:', chi2, 'p-val:', p_chi)
"""
        )
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        "## 7) Feature Engineering\nCriamos dummies, agregações por criativo e adset, e features temporais."
    )
)

cells.append(
    nbf.v4.new_code_cell(
        dedent(
            """\
data = df.copy()

# temporais
data['dayofweek'] = data['date'].dt.dayofweek
data['weekofyear'] = data['date'].dt.isocalendar().week

# agregações por creative_id
agg_creative = data.groupby('creative_id').agg({
    'impressions':'mean','clicks':'mean','conversions':'mean','spend':'mean','revenue':'mean'
}).rename(columns=lambda x: f'creative_{x}_mean').reset_index()

data = data.merge(agg_creative, on='creative_id', how='left')

# dummies (uma-hot para modelos)
cat_cols = ['creative_format','creative_theme','tone','device','os','placement']
data = pd.get_dummies(data, columns=cat_cols, drop_first=True)

# preencher NaNs de CPC/CPA com medianas
data['CPC'] = data['CPC'].fillna(data['CPC'].median())
data['CPA'] = data['CPA'].fillna(data['CPA'].median())

data.shape
"""
        )
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        "## 8) Modelagem preditiva\nModelos para prever CPC, CTR, CPA e CVR. Usamos LinearRegression e RandomForestRegressor como baseline."
    )
)

cells.append(
    nbf.v4.new_code_cell(
        dedent(
            """\
# Função utilitária para treinar modelos e reportar métricas
def train_and_report(X, y, target_name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train_s, y_train)
    pred_lr = lr.predict(X_test_s)
    
    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)
    
    results = {
        'lr_mae': mean_absolute_error(y_test, pred_lr),
        'lr_rmse': mean_squared_error(y_test, pred_lr, squared=False),
        'lr_r2': r2_score(y_test, pred_lr),
        'rf_mae': mean_absolute_error(y_test, pred_rf),
        'rf_rmse': mean_squared_error(y_test, pred_rf, squared=False),
        'rf_r2': r2_score(y_test, pred_rf)
    }
    print(f"Results for {target_name}:\\n", results)
    return {'lr': lr, 'rf': rf, 'scaler': scaler, 'metrics': results}
"""
        )
    )
)

cells.append(
    nbf.v4.new_code_cell(
        dedent(
            """\
# Seleção de features (simples)
features = ['impressions','clicks','spend','revenue','creative_impressions_mean','creative_clicks_mean','creative_conversions_mean','creative_spend_mean','creative_revenue_mean','dayofweek','weekofyear']
# adicionar algumas dummies (limitar o tamanho)
features += [c for c in data.columns if c.startswith('creative_format_')][:2]
features += [c for c in data.columns if c.startswith('tone_')][:2]
features = [f for f in features if f in data.columns]
print('n features:', len(features))
features
"""
        )
    )
)

cells.append(
    nbf.v4.new_code_cell(
        dedent(
            """\
# Preparar targets e treinar para cada um
targets = {
    'CTR': data['CTR'].fillna(0),
    'CPC': data['CPC'],
    'CPA': data['CPA'].fillna(data['CPA'].median()),
    'CVR': data['CVR'].fillna(0)
}

models_report = {}
for tname, y in targets.items():
    X = data[features].fillna(0)
    print('\\nTraining for', tname)
    models_report[tname] = train_and_report(X, y, tname)
"""
        )
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        "## 9) Importância de variáveis (Random Forest para um target de exemplo)\nMostramos features mais importantes para o target CPA como exemplo."
    )
)

cells.append(
    nbf.v4.new_code_cell(
        dedent(
            """\
# Exemplo: feature importance para CPA (se modelo existe)
if 'CPA' in models_report:
    importances = models_report['CPA']['rf'].feature_importances_
    feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)
    print(feat_imp.head(15))
else:
    print('Modelo CPA não disponível')
"""
        )
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        "## 10) Observações finais e próximos passos\n- Interpretar com cuidado: correlação não é causalidade.\n- Melhorar modelos com tuning, validação temporal e features mais ricas.\n- Possível próximos passos: A/B testing, uplift modeling, attribution modeling."
    )
)

nb["cells"] = cells

# ajusta aqui o caminho caso queira salvar em outro local
path = "projeto_core_full_analysis.ipynb"
with open(path, "w") as f:
    nbf.write(nb, f)

print(f"Notebook criado em: {path}")
