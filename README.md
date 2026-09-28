# refatoracao_mestrado

Refatoração do pipeline de análise de consumo energético de ônibus elétricos
(segmentação de viagens em microtrips a partir de picos de altitude, cálculo
de energia via integração de tensão x corrente, e visualizações).

## Estrutura

```
src/
  common.py  # configuração (paths, constantes, tabela de Schedule) e helpers
             # compartilhados (distância, energia acumulada)
  etl.py     # carga dos dados brutos (CSV + Parquet), segmentação em
             # microtrips, limpeza e exportação para Excel
  viz.py     # gráficos (Plotly/Folium): perfil de altitude, mapa de rota,
             # energia/velocidade/altitude vs distância, 3D e dashboard
```

## Uso

```bash
pip install -r requirements.txt

# ETL: gera microtrips_lidarAltitude_june.xlsx
python src/etl.py

# Visualizações de exemplo (reaproveita as funções de carga do etl.py)
python src/viz.py
```

## Configuração

Os parâmetros do dataset (caminho do Drive compartilhado, dispositivo,
ano/mês, janela de datas, limiares de limpeza) estão centralizados em
`src/common.py`. A chave de API do Stadia Maps pode ser definida via variável
de ambiente `STADIA_API_KEY` em vez de ficar hardcoded no código.
