"""Configurações e utilitários compartilhados entre o ETL e as visualizações."""
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid

# ---------------------------------------------------------------------------
# Paths e parâmetros do dataset
# ---------------------------------------------------------------------------
ROOT: str = 'G:\\Drives compartilhados\\DB LVMU\\'

# Entidades e medidas disponíveis no dataset particionado (referência).
ENTITIES: List[str] = ['can', 'location', 'device', 'inertial', 'climatic']
MEASURES: List[str] = [
    'absspeed', 'absfail', 'airpressure', 'brakepedal', 'auxbatteryvoltage',
    'auxbatteryfail', 'gearposition', 'distancekm', 'doorramp', 'ecaslight',
    'mainbatterystatus', 'motorfail', 'mainbatterytemp', 'motorleftmode',
    'motorleftrot', 'motorrightrot', 'motorrightmode', 'statusdoor', 'watertemp',
]

# Entidades/medidas efetivamente usadas pelo pipeline (evita índices mágicos).
ENTITY_LOCATION: str = 'location'
ENTITY_CAN: str = 'can'
MEASURE_MAIN_BATTERY_TEMP: str = 'mainbatterytemp'

DEVICE_ID: str = 'GIO5B97'
YEAR: int = 2024
MONTH: int = 5

EXCLUDED_TRIP_DATE: str = '2024-05-02'

# Parâmetros do run de segmentação (viagens de junho/2024).
PERIOD: str = 'month'
PERIOD_START_DATE: pd.Timestamp = pd.Timestamp('2024-06-03 07:00:00')
PERIOD_END_DATE: pd.Timestamp = pd.Timestamp('2024-06-24 23:00:00')
SEGMENTATION_PROMINENCE: float = 1.0
ALTITUDE_COL_SEGMENTATION: str = 'alt_lidar'

# Regras de limpeza pós-segmentação.
VALID_SCHEDULE_RANGE: Tuple[int, int] = (1, 15)
# Viagens com menos microtrips que isso são descartadas (indício de rota
# incompleta/atípica: retorno em "L" ou pela via errada).
MIN_MICROTRIPS_PER_TRIP: int = 18
# Viagens com Schedule mal atribuído, identificadas manualmente.
TRIPS_WITH_WRONG_SCHEDULE: List[int] = [3, 52]
CORRECTED_SCHEDULE_LABEL: int = 9

# Chave de API do Stadia Maps (mapas do Folium). NUNCA hardcode a chave real
# aqui: defina a variável de ambiente STADIA_API_KEY localmente (ex. em um
# arquivo .env não versionado) antes de rodar src/viz.py.
STADIA_API_KEY: str = os.environ.get('STADIA_API_KEY', '')

# ---------------------------------------------------------------------------
# Tabela de horários (Schedule) por horário de início da viagem
# ---------------------------------------------------------------------------
SCHEDULE_RANGES: List[Tuple[str, str, str]] = [
    ('06:59:00', '07:37:59', '1'),
    ('07:38:00', '08:17:59', '2'),
    ('08:18:00', '08:57:59', '3'),
    ('08:58:00', '09:37:59', '4'),
    ('09:38:00', '10:17:59', '5'),
    ('10:18:00', '10:57:59', '6'),
    ('10:58:00', '11:37:59', '7'),
    ('11:38:00', '12:17:59', '8'),
    ('12:18:00', '12:57:59', '9'),
    ('12:58:00', '13:37:59', '10'),
    ('14:58:00', '15:37:59', '11'),
    ('15:38:00', '16:17:59', '12'),
    ('16:18:00', '16:57:59', '13'),
    ('16:58:00', '17:37:59', '14'),
    ('17:38:00', '18:17:59', '15'),
    ('18:18:00', '18:57:59', '16'),
    ('18:58:00', '19:37:59', '17'),
    ('19:38:00', '20:20:00', '18'),
    ('21:18:00', '21:57:59', '19'),
    ('21:58:00', '23:05:00', '20'),
]


def assign_schedule(row: pd.Series) -> str:
    """Atribui o código de horário (Schedule) conforme o horário de início da viagem."""
    start_time = row['start'].time()
    for start_str, end_str, label in SCHEDULE_RANGES:
        if pd.Timestamp(start_str).time() <= start_time <= pd.Timestamp(end_str).time():
            return label
    return '00'


# ---------------------------------------------------------------------------
# Helpers de distância
# ---------------------------------------------------------------------------
def haversine_np(lat1, lon1, lat2, lon2):
    """Distância (km) entre arrays de pontos lat/lon consecutivos (graus)."""
    R = 6371000  # raio da Terra em metros
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c / 1000  # km


def compute_cumulative_distance_km(
    df: pd.DataFrame, lat_col: str = 'latitude', lon_col: str = 'longitude'
) -> np.ndarray:
    """Distância acumulada (km, Haversine) entre pontos consecutivos de um segmento.

    `df` deve estar ordenado por tempo. Usado pelas funções de plot; a
    segmentação em microtrips usa `geopy.distance.geodesic` (mais precisa),
    mantido separado em `etl.py`.
    """
    lat = df[lat_col].values
    lon = df[lon_col].values
    dist = np.zeros(len(df))
    if len(df) > 1:
        dist[1:] = haversine_np(lat[:-1], lon[:-1], lat[1:], lon[1:])
    return np.cumsum(dist)


# ---------------------------------------------------------------------------
# Helpers de energia
# ---------------------------------------------------------------------------
def calculate_energy_consumption(segment: dict, df_bat: pd.DataFrame) -> float:
    """Estima a energia consumida (kWh) em um segmento a partir de tensão/corrente.

    Args:
        segment: dict com chaves 'start' e 'end' (pd.Timestamp).
        df_bat: dados de bateria com colunas 'voltage', 'current', 'ts_rounded'.
    """
    start_time, end_time = segment['start'], segment['end']
    segment_bat = df_bat[
        (df_bat['ts_rounded'] >= start_time) & (df_bat['ts_rounded'] <= end_time)
    ].reset_index(drop=True)

    if not segment_bat.empty:
        segment_bat['power'] = segment_bat['voltage'] * segment_bat['current']
        time_seconds = (segment_bat['ts_rounded'] - segment_bat['ts_rounded'].iloc[0]).dt.total_seconds()
        energy_wh = trapezoid(segment_bat['power'], time_seconds) / 3600
        return energy_wh / 1000
    return np.nan


def compute_cumulative_energy_kwh(
    seg_loc: pd.DataFrame,
    seg_bat: pd.DataFrame,
    voltage_col: str = 'voltage',
    current_col: str = 'current',
    ts_col: str = 'ts_rounded',
) -> List[float]:
    """Energia acumulada (kWh) entre cada par de pontos consecutivos de `seg_loc`.

    `seg_loc` e `seg_bat` devem estar ordenados por tempo e com índice
    resetado (0..N-1). Usado pelas funções de plot que comparam energia,
    velocidade e altitude ao longo da distância percorrida.
    """
    energy_cum = []
    energy_total = 0.0
    for i in range(len(seg_loc) - 1):
        t0 = pd.to_datetime(seg_loc.loc[i, ts_col])
        t1 = pd.to_datetime(seg_loc.loc[i + 1, ts_col])
        seg = seg_bat[(seg_bat[ts_col] >= t0) & (seg_bat[ts_col] <= t1)].copy()
        if not seg.empty:
            power = seg[voltage_col].astype('float64') * seg[current_col].astype('float64')
            time_seconds = (pd.to_datetime(seg[ts_col]) - pd.to_datetime(seg[ts_col].iloc[0])).dt.total_seconds()
            energy_kwh = trapezoid(power, time_seconds) / 3600 / 1000
        else:
            energy_kwh = 0.0
        energy_total += energy_kwh
        energy_cum.append(energy_total)
    energy_cum.append(energy_total)
    return energy_cum
