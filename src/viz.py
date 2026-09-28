"""Visualizações: perfis de altitude, mapas de rota, gráficos combinados e dashboards."""
import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.colors
import plotly.graph_objs as go
import plotly.subplots as sp

from common import (
    STADIA_API_KEY,
    compute_cumulative_distance_km,
    compute_cumulative_energy_kwh,
)


# ---------------------------------------------------------------------------
# Perfil de altitude colorido por microtrip (Plotly)
# ---------------------------------------------------------------------------
def plot_altitude_profile_colored_by_microtrip_plotly(
    df_loc: pd.DataFrame,
    microtrips_result: pd.DataFrame,
    trip_index: int = 0,
    altitude_col: str = 'alt_lidar_interp',
    html_output: str = 'altitude_colored_microtrips_plotly.html',
):
    # 1. Select microtrips for this trip
    trip_id = trip_index + 1 if 'id_trip' in microtrips_result.columns else trip_index
    if 'id_trip' in microtrips_result.columns:
        microtrips = microtrips_result[microtrips_result['id_trip'] == trip_id]
    else:
        microtrips = microtrips_result.copy()
    if microtrips.empty:
        print('No microtrips for this trip.')
        return

    # 2. Temporal range for this trip
    trip_start = pd.to_datetime(microtrips['start'].iloc[0])
    trip_end = pd.to_datetime(microtrips['end'].iloc[-1])
    loc_seg = df_loc[(df_loc['ts_rounded'] >= trip_start) & (df_loc['ts_rounded'] <= trip_end)].copy()
    loc_seg['ts_rounded'] = pd.to_datetime(loc_seg['ts_rounded'])
    if loc_seg.empty:
        print('No location points for this trip.')
        return

    # 3. Global cumulative distance
    loc_seg['dist_cum'] = compute_cumulative_distance_km(loc_seg)

    # 4. Prepare Plotly figure
    fig = go.Figure()

    # Optional: plot the global profile as a faint gray base
    fig.add_trace(go.Scatter(
        x=loc_seg['dist_cum'],
        y=loc_seg[altitude_col],
        mode='lines',
        line=dict(color='lightgray', width=2),
        name='Full profile',
        hoverinfo='skip',
    ))

    # 5. Colormap: plotly qualitative (good for many segments!)
    colors = plotly.colors.qualitative.Plotly + plotly.colors.qualitative.D3 + plotly.colors.qualitative.Dark24
    n_colors = len(colors)

    # 6. Plot each microtrip with its color
    for idx, row in microtrips.reset_index().iterrows():
        t_start, t_end = pd.to_datetime(row['start']), pd.to_datetime(row['end'])
        mask = (loc_seg['ts_rounded'] >= t_start) & (loc_seg['ts_rounded'] <= t_end)
        if mask.sum() < 2:
            continue
        color = colors[idx % n_colors]
        fig.add_trace(go.Scatter(
            x=loc_seg.loc[mask, 'dist_cum'],
            y=loc_seg.loc[mask, altitude_col],
            mode='lines+markers',
            line=dict(color=color, width=4),
            marker=dict(size=4),
            name=f'Microtrip {idx+1}',
            hoverinfo='x+y+name',
        ))

    fig.update_layout(
        title=f'Altitude profile with colored microtrip segments (trip {trip_index})',
        xaxis_title='Cumulative distance (km)',
        yaxis_title='Altitude (m)',
        template='plotly_white',
        legend=dict(
            orientation='v',
            yanchor='top',
            y=1,
            xanchor='left',
            x=1.02,
            font=dict(size=14),
            bordercolor='black',
            borderwidth=1,
        ),
        margin=dict(l=60, r=200, t=60, b=60),
    )
    fig.write_html(html_output)
    print(f'Interactive HTML plot saved as {html_output}')
    return fig


# ---------------------------------------------------------------------------
# Rota no mapa (Folium), colorida por microtrip
# ---------------------------------------------------------------------------
def plot_microtrips_by_color(id_trip, microtrips_df: pd.DataFrame, df_loc: pd.DataFrame, api_key: str = STADIA_API_KEY):
    # Selecciona microtrips de ese viaje
    selected_microtrips = microtrips_df[microtrips_df['id_trip'] == id_trip]
    if selected_microtrips.empty:
        print(f'No microtrips for id_trip {id_trip}')
        return None

    # Toma todos los puntos del viaje
    trip_start = selected_microtrips['start'].min()
    trip_end = selected_microtrips['end'].max()
    trip_loc = df_loc[(df_loc['ts_rounded'] >= trip_start) & (df_loc['ts_rounded'] <= trip_end)].copy()

    # Centro del mapa
    lat_init, lon_init = trip_loc.iloc[0][['latitude', 'longitude']]
    mapa = folium.Map(
        location=[lat_init, lon_init],
        zoom_start=16,
        tiles=f'https://tiles.stadiamaps.com/tiles/alidade_smooth_dark/{{z}}/{{x}}/{{y}}.png?api_key={api_key}',
        attr='Stadia.AlidadeSmoothDark',
    )

    # Paleta de colores para microtrips
    num_microtrips = len(selected_microtrips)
    cmap = plt.get_cmap('tab20', num_microtrips)

    # Traza cada microtrip de color diferente
    for i, (_, row) in enumerate(selected_microtrips.iterrows()):
        mask = (trip_loc['ts_rounded'] >= row['start']) & (trip_loc['ts_rounded'] <= row['end'])
        points = trip_loc[mask][['latitude', 'longitude']].values.tolist()
        color_rgb = cmap(i)[:3]
        color_hex = '#%02x%02x%02x' % tuple(int(c * 255) for c in color_rgb)
        if len(points) > 1:
            folium.PolyLine(points, color=color_hex, weight=6, opacity=0.85,
                             popup=f"Microtrip {row['id_microtrip']}").add_to(mapa)
        # Opcional: inicio y fin de cada microtrip
        folium.CircleMarker(location=points[0], radius=4, color='lime', fill=True).add_to(mapa)
        folium.CircleMarker(location=points[-1], radius=4, color='red', fill=True).add_to(mapa)

    return mapa


# ---------------------------------------------------------------------------
# Energia + velocidade + altitude vs distância (múltiplas viagens)
# ---------------------------------------------------------------------------
def plot_energy_speed_altitude_vs_distance_multitrip(
    trip_indices: list,
    trips_df: pd.DataFrame,
    df_loc: pd.DataFrame,
    df_bat: pd.DataFrame,
    current_col: str = 'current',
    voltage_col: str = 'voltage',
    speed_col: str = 'speed',
    altitude_col: str = 'alt_lidar_interp',
    html_output: str = 'energy_speed_altitude_vs_distance_multitrip.html',
):
    """
    Plots accumulated Energy (kWh), Speed (km/h), and Altitude (m) vs Cumulative Distance (km) for several trips.
    - Energy: Y1 (left)
    - Speed: Y2 (right)
    - Altitude: Y3 (right, offset)
    """
    fig = go.Figure()
    color_list = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'gray']

    for idx, trip_index in enumerate(trip_indices):
        start = pd.to_datetime(trips_df.loc[trip_index, 'start'])
        end = pd.to_datetime(trips_df.loc[trip_index, 'end'])
        seg_loc = df_loc[(df_loc['ts_rounded'] >= start) & (df_loc['ts_rounded'] <= end)].copy()
        seg_bat = df_bat[(df_bat['ts_rounded'] >= start) & (df_bat['ts_rounded'] <= end)].copy()
        if seg_loc.empty or seg_bat.empty:
            print(f'Warning: No data for trip {trip_index}')
            continue
        seg_loc = seg_loc.sort_values('ts_rounded').reset_index(drop=True)

        dist_cum = compute_cumulative_distance_km(seg_loc)
        energy_cum = compute_cumulative_energy_kwh(seg_loc, seg_bat, voltage_col=voltage_col, current_col=current_col)

        # Speed and Altitude
        speed_kmh = seg_loc[speed_col].values * 3.6
        altitude = seg_loc[altitude_col].values

        color = color_list[idx % len(color_list)]

        # Energy - Y1
        fig.add_trace(go.Scatter(
            x=dist_cum,
            y=energy_cum,
            mode='lines+markers',
            name=f'Trip {trip_index} - Energy',
            line=dict(color=color, width=2),
            yaxis='y1',
        ))

        # Speed - Y2
        fig.add_trace(go.Scatter(
            x=dist_cum,
            y=speed_kmh,
            mode='lines',
            name=f'Trip {trip_index} - Speed',
            line=dict(color=color, width=1, dash='dot'),
            yaxis='y2',
        ))

        # Altitude - Y3
        fig.add_trace(go.Scatter(
            x=dist_cum,
            y=altitude,
            mode='lines',
            name=f'Trip {trip_index} - Altitude',
            line=dict(color=color, width=1, dash='dash'),
            yaxis='y3',
        ))

    # Layout with three y-axes
    fig.update_layout(
        title='Accumulated Energy (kWh), Speed (km/h), and Altitude (m) vs Cumulative Distance',
        xaxis=dict(title='Cumulative distance (km)'),
        yaxis=dict(
            title='Accumulated Energy (kWh)',
            side='left',
        ),
        yaxis2=dict(
            title='Speed (km/h)',
            side='right',
            anchor='x',
            overlaying='y',
        ),
        yaxis3=dict(
            title='Altitude (m)',
            side='right',
            anchor='free',
            overlaying='y',
            position=1.0,  # slightly to the right of the speed axis
        ),
        template='plotly_white',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.35,
            xanchor='center',
            x=0.5,
            font=dict(size=13),
        ),
    )
    fig.write_html(html_output)
    print(f'Graph saved as {html_output}')
    return fig


# ---------------------------------------------------------------------------
# Altitude x Energia x Distância em 3D, colorido por velocidade
# ---------------------------------------------------------------------------
def plot_3d_altitude_energy_vs_distance_with_speedcolor(
    trips_df: pd.DataFrame,
    df_loc: pd.DataFrame,
    df_bat: pd.DataFrame,
    trip_index: int = 0,
    current_col: str = 'current',
    voltage_col: str = 'voltage',
    speed_col: str = 'speed',
    altitude_col: str = 'alt_lidar_interp',
    html_output: str = '3d_altitude_energy_distance_speed.html',
):
    # Seleccionar el viaje
    start = pd.to_datetime(trips_df.loc[trip_index, 'start'])
    end = pd.to_datetime(trips_df.loc[trip_index, 'end'])
    seg_loc = df_loc[(df_loc['ts_rounded'] >= start) & (df_loc['ts_rounded'] <= end)].copy()
    seg_bat = df_bat[(df_bat['ts_rounded'] >= start) & (df_bat['ts_rounded'] <= end)].copy()
    seg_loc = seg_loc.sort_values('ts_rounded').reset_index(drop=True)

    dist_cum = compute_cumulative_distance_km(seg_loc)
    energy_cum = compute_cumulative_energy_kwh(seg_loc, seg_bat, voltage_col=voltage_col, current_col=current_col)

    # Preparar datos
    altitude = seg_loc[altitude_col].values
    speed_kmh = seg_loc[speed_col].values * 3.6

    # Gráfico 3D
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=dist_cum,
        y=altitude,
        z=energy_cum,
        mode='markers+lines',
        marker=dict(
            size=5,
            color=speed_kmh,  # Color por velocidad
            colorscale='Turbo',
            colorbar=dict(title='Speed (km/h)'),
        ),
        line=dict(
            color='rgba(100,100,100,0.2)',
            width=2,
        ),
        name='Profile',
    ))

    fig.update_layout(
        title='3D profile: Altitude vs Accumulated Energy vs Distance (color=Speed)',
        scene=dict(
            xaxis_title='Cumulative distance (km)',
            yaxis_title='Altitude (m)',
            zaxis_title='Accumulated energy (kWh)',
        ),
        width=900,
        height=800,
    )
    fig.write_html(html_output)
    fig.show()
    print(f'Graph saved as {html_output}')
    return fig


# ---------------------------------------------------------------------------
# Dashboard facetado (energia, altitude LiDAR, velocidade vs distância)
# ---------------------------------------------------------------------------
def plot_faceted_dashboard(
    trip_index: int,
    trips_df: pd.DataFrame,
    df_loc: pd.DataFrame,
    df_bat: pd.DataFrame,
    current_col: str = 'current',
    voltage_col: str = 'voltage',
    speed_col: str = 'speed',
    altitude_col: str = 'alt_lidar',  # SOLO LiDAR
    html_output: str = 'dashboard_faceted.html',
):
    # 1. Segmentación básica
    start = pd.to_datetime(trips_df.loc[trip_index, 'start'])
    end = pd.to_datetime(trips_df.loc[trip_index, 'end'])
    seg_loc = df_loc[(df_loc['ts_rounded'] >= start) & (df_loc['ts_rounded'] <= end)].copy()
    seg_bat = df_bat[(df_bat['ts_rounded'] >= start) & (df_bat['ts_rounded'] <= end)].copy()
    seg_loc = seg_loc.sort_values('ts_rounded').reset_index(drop=True)

    dist_cum = compute_cumulative_distance_km(seg_loc)
    # Altitud LiDAR
    altitude_lidar = seg_loc[altitude_col].values
    # Velocidad
    speed_kmh = seg_loc[speed_col].values * 3.6

    # 2. Energía acumulada
    energy_cum = compute_cumulative_energy_kwh(seg_loc, seg_bat, voltage_col=voltage_col, current_col=current_col)

    # 3. Crear subplots (2x2)
    fig = sp.make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Accumulated Energy (kWh) vs Distance',
            'Altitude LiDAR (m) vs Distance',
            'Speed (km/h) vs Distance',
            ' ',  # Vacío
        ),
    )
    # 4. Agregar trazas
    fig.add_trace(go.Scatter(
        x=dist_cum, y=energy_cum, mode='lines+markers', line=dict(color='orange'), name='Energy (kWh)'
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=dist_cum, y=altitude_lidar, mode='lines+markers', line=dict(color='green'), name='Altitude LiDAR'
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=dist_cum, y=speed_kmh, mode='lines+markers', line=dict(color='blue'), name='Speed (km/h)'
    ), row=2, col=1)

    # 5. Títulos y ejes
    fig.update_xaxes(title_text='Cumulative distance (km)', row=1, col=1)
    fig.update_xaxes(title_text='Cumulative distance (km)', row=1, col=2)
    fig.update_xaxes(title_text='Cumulative distance (km)', row=2, col=1)
    fig.update_yaxes(title_text='Energy (kWh)', row=1, col=1)
    fig.update_yaxes(title_text='Altitude LiDAR (m)', row=1, col=2)
    fig.update_yaxes(title_text='Speed (km/h)', row=2, col=1)

    fig.update_layout(
        height=800, width=1100,
        title=f'Faceted Dashboard (Trip {trip_index})',
        showlegend=False,
        template='plotly_white',
    )
    fig.write_html(html_output)
    fig.show()
    print(f'Dashboard guardado como {html_output}')
    return fig


# ---------------------------------------------------------------------------
# Demo: reproduz os usos originais (requer os dados brutos do ETL)
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    from common import (
        ALTITUDE_COL_SEGMENTATION,
        PERIOD,
        PERIOD_END_DATE,
        PERIOD_START_DATE,
        SEGMENTATION_PROMINENCE,
    )
    from etl import (
        filter_by_schedule,
        filter_valid_trips,
        load_battery_data,
        load_location_data,
        load_trips,
        segment_microtrips_by_period,
    )

    trips_06_24, _trips_05_24 = load_trips()
    df_loc = load_location_data()
    df_bat, _df_grouped_bat = load_battery_data()
    trips_valid = filter_valid_trips(trips_06_24)

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

    plot_altitude_profile_colored_by_microtrip_plotly(
        df_loc=df_loc,
        microtrips_result=microtrips_schedule,
        trip_index=12,
        altitude_col='alt_lidar_interp',
        html_output='altitude_colored_microtrips_plotly_trip12.html',
    )

    # Viajes anómalos identificados manualmente na etapa de limpeza do ETL.
    for trip_id in [84, 124, 137, 142, 175, 178, 188, 194]:
        map_obj = plot_microtrips_by_color(trip_id, microtrips_schedule, df_loc)
        if map_obj:
            map_obj.save(f'microtrips_trip{trip_id}_colored.html')
            print(f'Map for trip {trip_id} saved!')

    plot_energy_speed_altitude_vs_distance_multitrip(
        trip_indices=[17, 18, 19],
        trips_df=trips_valid,
        df_loc=df_loc,
        df_bat=df_bat,
        current_col='current',
        voltage_col='voltage',
        speed_col='speed',
        altitude_col='alt_lidar_interp',
        html_output='altitude_html/energy_speed_altitude_vs_distance_multitrip.html',
    )

    plot_3d_altitude_energy_vs_distance_with_speedcolor(
        trips_df=trips_valid,
        df_loc=df_loc,
        df_bat=df_bat,
        trip_index=0,
        current_col='current',
        voltage_col='voltage',
        speed_col='speed',
        altitude_col='alt_lidar_interp',
        html_output='3d_altitude_energy_distance_speed.html',
    )

    plot_faceted_dashboard(
        trip_index=2,
        trips_df=trips_valid,
        df_loc=df_loc,
        df_bat=df_bat,
        current_col='current',
        voltage_col='voltage',
        speed_col='speed',
        altitude_col='alt_lidar_interp',
        html_output='altitude_html/dashboard_faceted_trip0.html',
    )
