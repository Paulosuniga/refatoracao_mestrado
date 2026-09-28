"""ETL: carga de dados brutos, segmentação em microtrips e exportação para Excel."""
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from geopy.distance import geodesic
from scipy.signal import find_peaks

from common import (
    ALTITUDE_COL_SEGMENTATION,
    CORRECTED_SCHEDULE_LABEL,
    DEVICE_ID,
    ENTITY_CAN,
    ENTITY_LOCATION,
    EXCLUDED_TRIP_DATE,
    MEASURE_MAIN_BATTERY_TEMP,
    MIN_MICROTRIPS_PER_TRIP,
    MONTH,
    PERIOD,
    PERIOD_END_DATE,
    PERIOD_START_DATE,
    ROOT,
    SEGMENTATION_PROMINENCE,
    TRIPS_WITH_WRONG_SCHEDULE,
    VALID_SCHEDULE_RANGE,
    YEAR,
    assign_schedule,
    calculate_energy_consumption,
)


# ---------------------------------------------------------------------------
# Carga de dados
# ---------------------------------------------------------------------------
def load_trips(
    path_2024_06: str = 'trips_06_24.csv', path_2024_05: str = 'trips_05_24.csv'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    trips_06_24 = pd.read_csv(path_2024_06)
    trips_05_24 = pd.read_csv(path_2024_05)
    return trips_06_24, trips_05_24


def load_location_data(year: int = YEAR, month: int = MONTH, device_id: str = DEVICE_ID) -> pd.DataFrame:
    path_entity = f'{ROOT}{ENTITY_LOCATION}\\'
    df_loc = pd.read_parquet(
        path_entity,
        filters=[('year', '=', year), ('month', '=', month), ('device_id', '=', device_id)],
        engine='pyarrow',
    )
    df_loc = df_loc.drop(['year', 'month', 'day', 'device_id'], axis=1)
    df_loc = df_loc.sort_values(by='timestampgnss')
    df_loc['ts_rounded'] = df_loc['timestampgnss'].dt.floor('s')
    df_loc = df_loc.drop(columns=['timestampgnss', 'ts', 'accuracyhor', 'bearing', 'accuracyver', 'accuracyspeed'])
    df_loc['ts_rounded'] = df_loc['ts_rounded'].dt.tz_localize(None)
    return df_loc


def load_battery_data(
    year: int = YEAR, month: int = MONTH, device_id: str = DEVICE_ID
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Retorna (df_bat, df_grouped_bat): dados de bateria por leitura e agregados por segundo."""
    path_entity = f'{ROOT}{ENTITY_CAN}\\'
    path_measure = f'{path_entity}{MEASURE_MAIN_BATTERY_TEMP}\\'

    df_bat = pd.read_parquet(
        path_measure,
        filters=[('year', '=', year), ('month', '=', month), ('device_id', '=', device_id)],
    )
    df_bat = df_bat.drop(['year', 'month', 'day', 'device_id'], axis=1)
    df_bat = df_bat.sort_values(by='ts')
    df_bat['ts_rounded'] = df_bat['ts'].dt.floor('s')

    df_grouped_bat = df_bat.groupby('ts_rounded').agg({
        'voltage': ['mean', 'median'],
        'current': ['mean', 'median'],
        'soc': 'last',
        'soh': 'last',
    }).reset_index()
    df_grouped_bat = df_grouped_bat.loc[:, [
        ('ts_rounded', ''),
        ('voltage', 'mean'),
        ('current', 'mean'),
        ('soc', 'last'),
        ('soh', 'last'),
    ]]

    df_bat = df_bat.drop(columns=['ts', 'soh'])
    df_bat['ts_rounded'] = df_bat['ts_rounded'].dt.tz_localize(None)
    return df_bat, df_grouped_bat


# ---------------------------------------------------------------------------
# Segmentação em microtrips
# ---------------------------------------------------------------------------
def segment_trip_by_peaks(trip_loc: pd.DataFrame, alt_col: str, prominence: float = 3.0) -> List[Dict[str, int]]:
    """Segmenta uma viagem em microtrips por máximos e mínimos locais de altitude (sem suavização)."""
    y = trip_loc[alt_col].values
    peaks_max, _ = find_peaks(y, prominence=prominence)
    peaks_min, _ = find_peaks(-y, prominence=prominence)
    cortes = np.sort(np.concatenate([[0], peaks_max, peaks_min, [len(trip_loc) - 1]]))
    cortes = np.unique(cortes)
    cortes = cortes[(cortes >= 0) & (cortes < len(trip_loc))]
    return [
        {'start_idx': int(cortes[i]), 'end_idx': int(cortes[i + 1])}
        for i in range(len(cortes) - 1)
    ]


def extract_microtrip_features(
    trip_loc: pd.DataFrame,
    microtrip: Dict[str, int],
    df_bat: pd.DataFrame,
    id_trip: int,
    id_microtrip: int,
) -> Dict[str, Union[float, int, pd.Timestamp]]:
    """Calcula distância, velocidade média, energia e SOC inicial de um microtrip."""
    start_idx, end_idx = microtrip['start_idx'], microtrip['end_idx']
    seg = trip_loc.iloc[start_idx:end_idx + 1]
    distance = np.sum([
        geodesic(
            (seg.iloc[i]['latitude'], seg.iloc[i]['longitude']),
            (seg.iloc[i + 1]['latitude'], seg.iloc[i + 1]['longitude']),
        ).meters
        for i in range(len(seg) - 1)
    ]) / 1000  # metros -> km

    mean_speed = seg['speed'].mean()
    soc_init = np.nan
    if 'soc' in df_bat.columns:
        soc_candidates = df_bat[df_bat['ts_rounded'] >= seg.iloc[0]['ts_rounded']].head(1)
        if not soc_candidates.empty:
            soc_init = soc_candidates['soc'].iloc[0]

    energy_kwh = calculate_energy_consumption(
        {'start': seg.iloc[0]['ts_rounded'], 'end': seg.iloc[-1]['ts_rounded']},
        df_bat,
    )

    return {
        'start': seg.iloc[0]['ts_rounded'],
        'end': seg.iloc[-1]['ts_rounded'],
        'distance': distance,
        'mean_speed': mean_speed,
        'energy_kwh': energy_kwh,
        'id_trip': id_trip,
        'id_microtrip': id_microtrip,
        'soc_init': soc_init,
    }


def segment_microtrips_by_period(
    trips_df: pd.DataFrame,
    df_loc: pd.DataFrame,
    df_bat: pd.DataFrame,
    alt_col: str,
    period: str,
    start_date: Optional[pd.Timestamp],
    end_date: Optional[pd.Timestamp] = None,
    prominence: float = 3.0,
) -> pd.DataFrame:
    """Segmenta todas as viagens válidas no período, por picos de altitude."""
    trips_df = trips_df.copy()
    trips_df['start'] = pd.to_datetime(trips_df['start'])
    trips_df['end'] = pd.to_datetime(trips_df['end'])

    if period == 'day':
        if start_date is None:
            raise ValueError("'start_date' must be specified for period 'day'.")
        start_time = pd.Timestamp(start_date)
        end_time = start_time + pd.Timedelta(days=1)
    elif period == 'week':
        if start_date is None or end_date is None:
            raise ValueError("'start_date' and 'end_date' must be specified for period 'week'.")
        start_time = pd.Timestamp(start_date)
        end_time = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    elif period == 'month':
        if start_date is None:
            raise ValueError("'start_date' must be specified for period 'month'.")
        start_time = pd.Timestamp(start_date)
        end_time = start_time + pd.DateOffset(months=1)
    else:
        raise ValueError("Period must be 'day', 'week', or 'month'.")

    filtered_trips = trips_df[
        (trips_df['start'] >= start_time) & (trips_df['end'] <= end_time)
    ].reset_index(drop=True)

    all_microtrips = []
    for idx, trip in filtered_trips.iterrows():
        trip_start, trip_end = trip['start'], trip['end']
        id_trip = idx + 1
        trip_loc = df_loc[
            (df_loc['ts_rounded'] >= trip_start) & (df_loc['ts_rounded'] <= trip_end)
        ].reset_index(drop=True)
        if len(trip_loc) < 3:
            continue
        microtrips = segment_trip_by_peaks(trip_loc, alt_col, prominence)
        for id_micro, micro in enumerate(microtrips, 1):
            features = extract_microtrip_features(trip_loc, micro, df_bat, id_trip, id_micro)
            all_microtrips.append(features)

    return pd.DataFrame(all_microtrips)


# ---------------------------------------------------------------------------
# Limpeza e validação
# ---------------------------------------------------------------------------
def filter_valid_trips(trips: pd.DataFrame, excluded_date: str = EXCLUDED_TRIP_DATE) -> pd.DataFrame:
    trips_valid = trips[trips['valid'] == True].reset_index(drop=True)  # noqa: E712
    trips_valid['start'] = pd.to_datetime(trips_valid['start'])
    trips_valid['end'] = pd.to_datetime(trips_valid['end'])
    trips_valid = trips_valid[
        trips_valid['start'].dt.date != pd.to_datetime(excluded_date).date()
    ].copy()
    return trips_valid


def filter_by_schedule(microtrips: pd.DataFrame, schedule_range: Tuple[int, int] = VALID_SCHEDULE_RANGE) -> pd.DataFrame:
    microtrips = microtrips.copy()
    microtrips['Schedule'] = microtrips.apply(assign_schedule, axis=1)
    microtrips['Schedule'] = pd.to_numeric(microtrips['Schedule'], errors='coerce').astype('int32')
    low, high = schedule_range
    return microtrips[microtrips['Schedule'].between(low, high)]


def remove_short_trips(microtrips: pd.DataFrame, min_microtrips: int = MIN_MICROTRIPS_PER_TRIP) -> pd.DataFrame:
    """Remove viagens com menos de `min_microtrips` segmentos.

    Indício de rota incompleta/atípica: retorno em formato de "L" no pico ou
    retorno pelo lado errado da via.
    """
    microtrips_per_trip = (
        microtrips.groupby('id_trip')['id_microtrip'].count().reset_index(name='microtrips_count')
    )
    problem_trip_ids = microtrips_per_trip[
        microtrips_per_trip['microtrips_count'] < min_microtrips
    ]['id_trip']
    return microtrips[~microtrips['id_trip'].isin(problem_trip_ids)].reset_index(drop=True)


def remove_trips_with_nan_energy(
    df: pd.DataFrame, energy_col: str = 'energy_kwh', trip_col: str = 'id_trip'
) -> pd.DataFrame:
    """Remove todos os id_trip que tenham ao menos um NaN em `energy_col`."""
    trips_with_nan = df[df[energy_col].isna()][trip_col].unique()
    return df[~df[trip_col].isin(trips_with_nan)].reset_index(drop=True)


def reassign_consecutive_trip_ids(microtrips: pd.DataFrame) -> pd.DataFrame:
    microtrips = microtrips.copy()
    microtrips['id_trip'] = pd.Categorical(microtrips['id_trip']).codes + 1
    return microtrips


def fix_known_schedule_errors(
    microtrips: pd.DataFrame,
    trip_ids: List[int] = TRIPS_WITH_WRONG_SCHEDULE,
    corrected_schedule: int = CORRECTED_SCHEDULE_LABEL,
) -> pd.DataFrame:
    """Corrige o Schedule de viagens com rótulo mal atribuído (identificadas manualmente)."""
    microtrips = microtrips.copy()
    microtrips.loc[microtrips['id_trip'].isin(trip_ids), 'Schedule'] = corrected_schedule
    return microtrips


# ---------------------------------------------------------------------------
# Pipeline completo
# ---------------------------------------------------------------------------
def run_etl(output_path: str = 'microtrips_lidarAltitude_june.xlsx') -> pd.DataFrame:
    trips_06_24, _trips_05_24 = load_trips()
    df_loc = load_location_data()
    df_bat, _df_grouped_bat = load_battery_data()

    trips_valid = filter_valid_trips(trips_06_24)
    print(trips_valid.head())
    print(f'Total de viajes válidos: {len(trips_valid)}')

    microtrips_result = segment_microtrips_by_period(
        trips_df=trips_valid,
        df_loc=df_loc,
        df_bat=df_bat,
        alt_col=ALTITUDE_COL_SEGMENTATION,
        period=PERIOD,
        start_date=PERIOD_START_DATE,
        end_date=PERIOD_END_DATE,
        prominence=SEGMENTATION_PROMINENCE,
    )

    microtrips_schedule = filter_by_schedule(microtrips_result)
    print(f'Total de microviajes en los viajes Schedule 1-15: {len(microtrips_schedule)}')
    print(f"Viajes únicos encontrados: {microtrips_schedule['id_trip'].unique()}")

    microtrips_schedule_clean = remove_short_trips(microtrips_schedule)
    print(f'Original shape: {microtrips_schedule.shape}')
    print(f'Filtered shape (only good trips): {microtrips_schedule_clean.shape}')

    missing_energy = microtrips_schedule_clean[microtrips_schedule_clean['energy_kwh'].isna()]
    print('id_trip con energy_kwh == NaN:')
    print(missing_energy['id_trip'].unique())

    microtrips_schedule_clean_filtered = remove_trips_with_nan_energy(microtrips_schedule_clean)
    print(microtrips_schedule_clean_filtered['id_trip'].unique())

    microtrips_result_clean = reassign_consecutive_trip_ids(microtrips_schedule_clean_filtered)
    print(microtrips_result_clean['id_trip'].unique())

    ids_with_schedule_10 = microtrips_result_clean[microtrips_result_clean['Schedule'] == 10]['id_trip'].unique()
    print(ids_with_schedule_10)

    columns_to_inspect = ['start', 'end', 'distance', 'mean_speed', 'energy_kwh', 'id_trip', 'id_microtrip', 'Schedule']
    print('----- Viaje id_trip = 3 -----')
    print(microtrips_result_clean[microtrips_result_clean['id_trip'] == 3][columns_to_inspect])
    print('\n----- Viaje id_trip = 52 -----')
    print(microtrips_result_clean[microtrips_result_clean['id_trip'] == 52][columns_to_inspect])

    microtrips_result_clean = fix_known_schedule_errors(microtrips_result_clean)
    print(microtrips_result_clean[microtrips_result_clean['id_trip'].isin(TRIPS_WITH_WRONG_SCHEDULE)][
        ['id_trip', 'id_microtrip', 'Schedule']
    ])

    microtrips_result_clean.to_excel(output_path, index=False)
    print(f'Archivo guardado como {output_path}')

    return microtrips_result_clean


if __name__ == '__main__':
    run_etl()
