import pandas as pd
import numpy as np
import json
import requests
import time
from datetime import datetime, timedelta
import os
import math

# --- Novas importações para os modelos de predição ---
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression

# ============================================================================
# --- CONFIGURAÇÃO DO CLIENTE ---
# ID e Estações do cliente "Clayton Sheiki Tessaro"
# ============================================================================
CLIENTE_ID = 92088 # ID DO CLIENTE (GROWER_ID)
CLIENTE_NOME = "Clayton Sheiki Tessaro" # NOME DO CLIENTE
ESTACOES_DO_CLIENTE = [
    {'name': 'Santa Ernestina T05', 'id_estacao': '80977', 'latitude': -12.4756, 'longitude': -55.6867},
    {'name': 'Santa Ernestina T07', 'id_estacao': '80985', 'latitude': -12.4168, 'longitude': -55.7401},
    {'name': 'Santa Ernestina T04', 'id_estacao': '80986', 'latitude': -12.4778, 'longitude': -55.7003},
    {'name': 'Santa Ernestina T12', 'id_estacao': '80984', 'latitude': -12.4005, 'longitude': -55.7182},
    {'name': 'Santa Ernestina', 'id_estacao': '39266', 'latitude': -12.4048, 'longitude': -55.740738},
    {'name': 'Santa Ernestina T03', 'id_estacao': '37191', 'latitude': -12.496, 'longitude': -55.6931},
    {'name': 'Santa Ernestina T13', 'id_estacao': '59504', 'latitude': -12.3868, 'longitude': -55.7189},
    {'name': 'Santa Ernestina T11', 'id_estacao': '65610', 'latitude': -12.4196, 'longitude': -55.7304},
]
# Define quantos anos de dados históricos buscar
ANOS_DE_HISTORICO = 2
# ============================================================================


class RelatorioClimaCompleto:
    """
    Classe para gerar um relatório HTML, lendo credenciais do ambiente
    e focada em um único cliente pré-definido.
    """
    def __init__(self, grower_id: int, grower_name: str, stations: list):
        
        # ============================================================================
        # !! IMPORTANTE !! Lendo credenciais das Variáveis de Ambiente (Secrets)
        # ============================================================================
        
        # Tente ler as variáveis de ambiente. Se falhar, avise.
        hist_sessionid = os.environ.get('HISTORICAL_SESSIONID')
        hist_csrftoken = os.environ.get('HISTORICAL_CSRFTOKEN')
        fc_sessionid = os.environ.get('FORECAST_SESSIONID')
        fc_csrftoken = os.environ.get('FORECAST_CSRFTOKEN')
        
        if not all([hist_sessionid, hist_csrftoken, fc_sessionid, fc_csrftoken]):
            print("ERRO CRÍTICO: As variáveis de ambiente (credenciais) não foram encontradas.")
            print("Certifique-se de configurar os Secrets no GitHub.")
            raise ValueError("Credenciais da API não definidas.")

        # --- CREDENCIAIS PARA DADOS HISTÓRICOS ---
        self.historical_cookies = { 'sessionid': hist_sessionid, 'csrftoken': hist_csrftoken }
        self.historical_headers = {
            "accept": "application/json", "Content-Type": "application/json",
            "X-CSRFToken": hist_csrftoken,
            "Referer": "https://admin.farmcommand.com/",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        # --- CREDENCIAIS PARA PREVISÃO DO TEMPO DIÁRIA ---
        self.forecast_cookies = { 'sessionid': fc_sessionid, 'csrftoken': fc_csrftoken }
        self.forecast_headers = {
            "accept": "application/json", "Content-Type": "application/json",
            "X-CSRFToken": fc_csrftoken,
            "Referer": "https://admin.farmcommand.com/",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        # --- CREDENCIAIS PARA PREVISÃO DO TEMPO HORÁRIA ---
        self.hourly_forecast_cookies = { 'sessionid': fc_sessionid, 'csrftoken': fc_csrftoken }
        self.hourly_forecast_headers = {
            "accept": "application/json", "Content-Type": "application/json",
            "X-CSRFToken": fc_csrftoken,
            "Referer": "https://admin.farmcommand.com/",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        # ============================================================================
        
        self.weather_url_base = "https://admin.farmcommand.com/weather/{}/historical-summary-hourly/"
        self.assets_url = "https://admin.farmcommand.com/asset/?season=1083"
        self.field_border_url = "https://admin.farmcommand.com/fieldborder/?assetID={}&format=json"
        self.forecast_url = "https://admin.farmcommand.com/weather/wsi/daily-forecast/"
        self.hourly_forecast_url = "https://admin.farmcommand.com/weather/wsi/hourly-forecast/"

        # --- DADOS DO CLIENTE INJETADOS (Substituindo o Excel) ---
        self.stations_info = stations
        self.grower_name_cache = {grower_id: grower_name}
        self.target_grower_id = grower_id
        # --- Fim da Modificação ---

    # --- INÍCIO DOS MÉTODOS COLADOS DO SCRIPT ORIGINAL ---
    
    def _traduzir_dia_semana(self, dow: str) -> str:
        """Traduz o dia da semana do inglês para o português."""
        dias = {
            "Monday": "Seg", "Tuesday": "Ter", "Wednesday": "Qua",
            "Thursday": "Qui", "Friday": "Sex", "Saturday": "Sáb", "Sunday": "Dom"
        }
        return dias.get(dow, dow)

    def _traduzir_descricao_clima(self, phrase: str) -> str:
        """Traduz descrições comuns do clima para português, baseado no seu script funcional."""
        if not phrase: return "N/D"
        translations = {
            "Sunny": "Ensolarado", "Mostly Sunny": "Predominantemente Ensolarado",
            "Partly Cloudy": "Parcialmente Nublado", "Partly Sunny": "Parcialmente Ensolarado",
            "Mostly Cloudy": "Predominantemente Nublado", "Cloudy": "Nublado",
            "Showers": "Pancadas de Chuva", "Rain": "Chuva", "Thunderstorms": "Trovoadas",
            "Scattered Thunderstorms": "Trovoadas Esparsas", "Isolated Thunderstorms": "Trovoadas Isoladas",
            "PM Thunderstorms": "Trovoadas à Tarde", "AM Showers": "Pancadas de Chuva pela Manhã",
            "PM Showers": "Pancadas de Chuva à Tarde", "Light Rain": "Chuva Leve", "Clear": "Céu Limpo",
            "Hazy": "Neblina Seca", "Fog": "Nevoeiro", "Mix of sun and clouds": "Sol e Nuvens",
            "Few Showers": "Poucas Pancadas"
        }
        return translations.get(phrase, phrase)
    
    def _make_request(self, url: str, params: dict = None) -> dict | list | None:
        """Realiza uma requisição GET para a API de dados históricos."""
        try:
            response = requests.get(url, params=params, cookies=self.historical_cookies, headers=self.historical_headers, timeout=180)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f" -> Erro de requisição para {url}: {e}.")
            return None

    def get_field_borders_for_grower(self, grower_id: int) -> list:
        """Busca os talhões e suas geometrias para um cliente."""
        print(f"\nBuscando talhões para o cliente ID: {grower_id}...")
        all_assets = self._make_request(self.assets_url)
        if not all_assets: return []
        farm_ids = [item["id"] for item in all_assets if item.get("parent") == grower_id and item.get("category") == "Farm"]
        if not farm_ids:
            farm_info = next((item for item in all_assets if item["id"] == grower_id and item["category"] == "Farm"), None)
            if farm_info: farm_ids = [grower_id]
        field_ids = [item["id"] for item in all_assets if item.get("parent") in farm_ids and item.get("category") == "Field"]
        all_borders = []
        for field_id in field_ids:
            border_data = self._make_request(self.field_border_url.format(field_id))
            if not border_data or not border_data[0].get("shapeData"): continue
            try:
                field_info = next((item for item in all_assets if item["id"] == field_id), None)
                shape_data = json.loads(border_data[0]["shapeData"])
                geom = shape_data.get('features', [{}])[0].get('geometry', shape_data)
                coords_raw = []
                if geom.get('type') == 'Polygon': coords_raw = geom.get('coordinates', [[]])[0]
                elif geom.get('type') == 'MultiPolygon': coords_raw = geom.get('coordinates', [[[]]])[0][0]
                coords_leaflet = [[c[1], c[0]] for c in coords_raw]
                if coords_leaflet:
                    all_borders.append({'field_id': field_id, 'field_name': field_info['label'] if field_info else f"Talhão {field_id}", 'centroid': [border_data[0]["centroid_lat"], border_data[0]["centroid_lon"]], 'geometry': {'type': 'Polygon', 'coordinates': [coords_leaflet]}})
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                print(f" -> Falha ao processar borda para o talhão ID {field_id}: {e}")
        print(f"Encontrados {len(all_borders)} talhões.")
        return all_borders

    def buscar_dados_climaticos(self, station_id: str, start_date: str, end_date: str) -> list:
        """
        Busca dados climáticos horários fazendo requisições em blocos de 60 dias para maior eficiência.
        """
        all_results = []
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        current_dt = start_dt

        print(f"--- Iniciando busca otimizada (blocos de 60 dias) para Estação: {station_id} ---")
        
        while current_dt <= end_dt:
            chunk_end_dt = min(current_dt + timedelta(days=60), end_dt)
            api_start = current_dt.strftime('%Y-%m-%dT00:00:00')
            api_end = chunk_end_dt.strftime('%Y-%m-%dT23:59:59')
            
            print(f"       Buscando dados de {current_dt.strftime('%Y-%m-%d')} a {chunk_end_dt.strftime('%Y-%m-%d')}...")
            
            url = self.weather_url_base.format(station_id)
            params = {'startDate': api_start, 'endDate': api_end, 'format': 'json'}
            json_data = self._make_request(url, params=params)
            
            if json_data and 'results' in json_data:
                all_results.extend(json_data['results'])
            
            time.sleep(0.2)
            current_dt = chunk_end_dt + timedelta(days=1)
            
        print(f"--- Busca para a estação {station_id} concluída. {len(all_results)} registros horários encontrados. ---")
        return all_results

    def buscar_previsao_clima(self, lat: float, lon: float) -> list:
        """Busca e processa os dados de previsão para os próximos 10 dias de forma robusta."""
        data = {"lat": lat, "lon": lon, "unit": "m"}
        max_retries = 3
        delay_seconds = 5
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.forecast_url,
                    headers=self.forecast_headers,
                    cookies=self.forecast_cookies,
                    json=data,
                    timeout=60
                )
                response.raise_for_status()
                api_data = response.json()
                
                previsoes_processadas = []
                forecasts_raw = api_data.get("forecasts", [])[:10]

                if not forecasts_raw:
                    print(" -> Aviso: API de previsão diária retornou uma lista vazia.")
                    return []

                for forecast in forecasts_raw:
                    dia_dados = forecast.get('day', {})
                    data_previsao = datetime.fromtimestamp(forecast.get("fcst_valid", 0))
                    previsao_dia = {
                        "data": data_previsao.strftime('%d/%m'),
                        "dia_semana": self._traduzir_dia_semana(forecast.get("dow", "")),
                        "min_temp": forecast.get("min_temp"),
                        "max_temp": forecast.get("max_temp"),
                        "descricao": self._traduzir_descricao_clima(dia_dados.get("phrase_32char")),
                        "prob_precip": dia_dados.get("pop", 0),
                        "qtd_precip": forecast.get("qpf", 0.0),
                        "vento_vel": dia_dados.get("wspd", 0),
                        "vento_dir": dia_dados.get("wdir_cardinal", "N/D"),
                    }
                    previsoes_processadas.append(previsao_dia)
                
                print(f" -> Sucesso! Previsão diária de {len(previsoes_processadas)} dias encontrada.")
                return previsoes_processadas

            except requests.exceptions.RequestException as e:
                print(f" -> Falha na requisição da previsão diária (Tentativa {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(delay_seconds)
                else:
                    print(" -> ERRO CRÍTICO: Todas as tentativas de buscar a previsão diária falharam.")
                    return []
        
        return []

    def buscar_previsao_horaria(self, lat: float, lon: float) -> list:
        """Busca e processa os dados de previsão horária para as próximas 48 horas."""
        data = {"lat": lat, "lon": lon, "unit": "m"}
        max_retries = 3
        delay_seconds = 5
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.hourly_forecast_url,
                    headers=self.hourly_forecast_headers,
                    cookies=self.hourly_forecast_cookies,
                    json=data,
                    timeout=60
                )
                response.raise_for_status()
                api_data = response.json()
                
                previsoes_processadas = []
                forecasts_raw = api_data.get("forecasts", [])[:48]

                if not forecasts_raw:
                    print(" -> Aviso: API de previsão horária retornou uma lista vazia.")
                    return []

                for hour_data in forecasts_raw:
                    previsao_hora = {
                        "fcst_valid_local": hour_data.get("fcst_valid_local"),
                        "temp": hour_data.get("temp"),
                        "rh": hour_data.get("rh"),
                        "wspd": hour_data.get("wspd"),
                        "delta_t": hour_data.get("delta_t"),
                        "pop": hour_data.get("pop", 0),
                        "qpf": hour_data.get("qpf", 0.0)
                    }
                    previsoes_processadas.append(previsao_hora)
                
                print(f" -> Sucesso! Previsão de {len(previsoes_processadas)} horas encontrada.")
                return previsoes_processadas

            except requests.exceptions.RequestException as e:
                print(f" -> Falha na requisição da previsão horária (Tentativa {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(delay_seconds)
                else:
                    print(" -> ERRO CRÍTICO: Todas as tentativas de buscar a previsão horária falharam.")
                    return []
        
        return []

    def _is_in_mato_grosso(self, lat: float, lon: float) -> bool:
        """Verifica se um par de coordenadas está dentro da caixa delimitadora do Mato Grosso."""
        if lat is None or lon is None: return False
        lat_min, lat_max = -18.2, -7.5
        lon_min, lon_max = -61.8, -50.0
        return (lat_min <= lat <= lat_max) and (lon_min <= lon <= lon_max)

    def processar_para_dataframe(self, json_list: list, station_id: str, station_name: str) -> pd.DataFrame:
        """Processa a resposta JSON da API horária e aplica filtros de limpeza de forma robusta."""
        if not json_list: return pd.DataFrame()

        def get_wind_direction(record):
            wind_dir_data = record.get('wind_direction_deg')
            if isinstance(wind_dir_data, dict): return wind_dir_data.get('avg')
            elif isinstance(wind_dir_data, (int, float)): return wind_dir_data
            return None

        records = [{
            'datetime': r.get('local_time'), 'precipitacao_mm': r.get('total_precip_mm'),
            'temp_media_c': r.get('avg_temp_c'), 'temp_min_c': r.get('min_temp_c'),
            'temp_max_c': r.get('max_temp_c'), 'umidade_media_perc': r.get('avg_relative_humidity'),
            'umidade_min_perc': r.get('min_relative_humidity'), 'umidade_max_perc': r.get('max_relative_humidity'),
            'vento_medio_kph': r.get('avg_windspeed_kph'), 'rajada_max_kph': r.get('wind_gust_kph', {}).get('max'),
            'vento_direcao_graus': get_wind_direction(r), 'delta_t': r.get('avgDeltaT'), 'gfdi': r.get('avgGFDI')
        } for r in json_list]
        
        df = pd.DataFrame(records)
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce', utc=True)
        df = df.dropna(subset=['datetime']).sort_values('datetime')
        
        for col in df.columns:
            if col != 'datetime': df[col] = pd.to_numeric(df[col], errors='coerce')
        
        print(f"       Analisando {len(df)} registros horários para a estação '{station_name}'...")
        initial_rows = len(df)
        
        # --- INÍCIO DA REVISÃO COMPLETA DOS FILTROS ---

        # 1. Filtro robusto de TEMPERATURA
        temp_invalida = (
            (df['temp_media_c'] > 50) | (df['temp_media_c'] == 0.0) | (df['temp_media_c'] == 1.0) |
            (df['temp_max_c'] > 50)   | (df['temp_max_c'] == 0.0)   | (df['temp_max_c'] == 1.0) |
            (df['temp_min_c'] > 50)   | (df['temp_min_c'] == 0.0)   | (df['temp_min_c'] == 1.0)
        )
        colunas_temp = ['temp_media_c', 'temp_min_c', 'temp_max_c', 'delta_t', 'gfdi']
        df.loc[temp_invalida, colunas_temp] = np.nan

        # 2. Filtro robusto de UMIDADE
        umidade_invalida = (
            (df['umidade_media_perc'] == 0) |
            (df['umidade_max_perc'] == 0) |
            (df['umidade_min_perc'] == 0)
        )
        colunas_umidade = ['umidade_media_perc', 'umidade_min_perc', 'umidade_max_perc', 'delta_t', 'gfdi']
        df.loc[umidade_invalida, colunas_umidade] = np.nan

        # 3. NOVO Filtro de VENTO
        vento_invalido = (df['vento_medio_kph'] > 150) | (df['rajada_max_kph'] > 150)
        colunas_vento = ['vento_medio_kph', 'rajada_max_kph']
        df.loc[vento_invalido, colunas_vento] = np.nan
        
        # 4. Filtro regional para Mato Grosso (opcional, mantido)
        station_info = next((s for s in self.stations_info if s['name'] == station_name), None)
        if station_info and self._is_in_mato_grosso(station_info['latitude'], station_info['longitude']):
            df.loc[df['temp_media_c'] < 10, ['temp_media_c', 'temp_min_c', 'temp_max_c']] = np.nan

        # --- FIM DA REVISÃO ---
        
        # Remove apenas linhas onde os dados essenciais para os gráficos são nulos após a limpeza.
        df.dropna(subset=['temp_media_c', 'umidade_media_perc'], inplace=True)
        final_rows = len(df)
        
        if (initial_rows - final_rows) > 0:
            print(f"       Limpeza concluída. Total de {initial_rows - final_rows} registros inválidos removidos.")
        
        df['nome_estacao'] = station_name
        df['station_id'] = station_id
        return df

    def gerar_predicao_clima(self, df_horario: pd.DataFrame) -> dict:
        """
        Gera predições de chuva mensais e semanais usando os dados históricos.
        """
        print("\n--- Iniciando Geração de Predições de Longo Prazo ---")
        if df_horario.empty:
            print(" -> Aviso: DataFrame de entrada vazio. Predições não podem ser geradas.")
            return {}
            
        df_diario = df_horario.set_index('datetime').resample('D')['precipitacao_mm'].sum().reset_index()
        df_diario.columns = ['date', 'precipitacao_total']
        
        if len(df_diario) < 365:
            print(f" -> Aviso: Dados insuficientes para predição. Encontrados apenas {len(df_diario)} dias.")
            return {}

        print(f" -> {len(df_diario)} dias de histórico de chuva agregados para análise.")
        
        monthly_df = df_diario.set_index('date').resample('ME').sum().reset_index()
        monthly_df.columns = ['date', 'total_precipitation']
        
        weekly_df = df_diario.set_index('date').resample('W-MON').sum().reset_index()
        weekly_df.columns = ['date', 'total_precipitation']

        # --- Lógica de Datas Futuras ---
        current_date = datetime.now()
        first_day_of_current_month = current_date.replace(day=1)
        first_day_of_next_month = (first_day_of_current_month + timedelta(days=32)).replace(day=1)
        
        future_monthly_dates = pd.date_range(start=first_day_of_next_month, periods=12, freq='ME')
        future_weekly_dates = pd.date_range(start=current_date, periods=7, freq='W-MON')[1:]

        forecasts_12_months = {}
        forecasts_6_weeks = {}

        # --- Média Histórica ---
        print("\n -> Calculando: Média Histórica...")
        average_monthly_precipitation = monthly_df.groupby(monthly_df['date'].dt.month)['total_precipitation'].mean()
        forecasts_12_months['Média Histórica'] = [average_monthly_precipitation.get(m.month, 0) for m in future_monthly_dates]
        
        # --- Prophet ---
        print(" -> Executando modelo: Prophet...")
        try:
            prophet_df_monthly = monthly_df.rename(columns={'date': 'ds', 'total_precipitation': 'y'})
            prophet_df_monthly['ds'] = prophet_df_monthly['ds'].dt.tz_localize(None)
            if len(prophet_df_monthly) > 12:
                model = Prophet(seasonality_mode='multiplicative', yearly_seasonality=True)
                model.fit(prophet_df_monthly)
                future_df_m = pd.DataFrame({'ds': future_monthly_dates})
                forecast_m = model.predict(future_df_m)
                forecasts_12_months['Prophet'] = [max(0, y) for y in forecast_m['yhat']]
        except Exception as e:
            print(f"       ! Falha ao rodar Prophet (mensal): {e}")

        # --- ARIMA ---
        print(" -> Executando modelo: ARIMA...")
        try:
            if len(monthly_df) > 24:
                model = ARIMA(monthly_df['total_precipitation'], order=(1,0,0), seasonal_order=(1,1,1,12), enforce_stationarity=False, enforce_invertibility=False)
                model_fit = model.fit()
                forecast_m = model_fit.get_forecast(steps=12).predicted_mean
                forecasts_12_months['ARIMA'] = [max(0, y) for y in forecast_m]
        except Exception as e:
            print(f"       ! Falha ao rodar ARIMA (mensal): {e}")
        
        # --- Regressão Linear ---
        print(" -> Executando modelo: Regressão Linear...")
        try:
            monthly_df['month'] = monthly_df['date'].dt.month
            monthly_df['year'] = monthly_df['date'].dt.year
            lr_monthly_preds = []
            for date in future_monthly_dates:
                target_month = date.month
                historical_data = monthly_df[monthly_df['month'] == target_month]
                if len(historical_data) > 2:
                    X = historical_data['year'].values.reshape(-1, 1)
                    y = historical_data['total_precipitation'].values
                    model = LinearRegression().fit(X, y)
                    lr_monthly_preds.append(max(0, model.predict([[date.year]])[0]))
                else:
                    lr_monthly_preds.append(np.nan)
            forecasts_12_months['Regressão Linear'] = lr_monthly_preds
        except Exception as e:
            print(f"       ! Falha ao rodar Regressão Linear (mensal): {e}")

        # --- Predição Semanal ---
        average_weekly_precipitation = weekly_df.groupby(weekly_df['date'].dt.isocalendar().week)['total_precipitation'].mean()
        future_weeks = [d.isocalendar()[1] for d in future_weekly_dates]
        forecasts_6_weeks['Média Histórica'] = [average_weekly_precipitation.get(w, 0) for w in future_weeks]

        # --- Montar o dicionário de resultados ---
        weekly_labels = []
        for d in future_weekly_dates:
            end_of_week = d + timedelta(days=6)
            weekly_labels.append(f"{d.strftime('%d/%b')} - {end_of_week.strftime('%d/%b')}")

        resultado_final = {
            "mensal": {
                "labels": [d.strftime('%b/%y') for d in future_monthly_dates],
                "previsoes": forecasts_12_months
            },
            "semanal": {
                "labels": weekly_labels,
                "previsoes": forecasts_6_weeks
            }
        }
        print("--- Predições de Longo Prazo Geradas com Sucesso ---")
        return resultado_final
    # --- FIM DOS MÉTODOS COLADOS ---

    # ============================================================================
    # MÉTODO MODIFICADO: gerar_html_final
    # Remove a abertura do navegador e salva em uma pasta 'dist'
    # ============================================================================
    def gerar_html_final(self, df: pd.DataFrame, geodata: dict, all_forecasts: dict, prediction_data: dict):
        """Gera o arquivo HTML final, otimizado para salvar no servidor."""
        print("\nGerando relatório HTML...")
        json_data = df.to_json(orient='records', date_format='iso')
        json_geodata = json.dumps(geodata)
        json_all_forecasts = json.dumps(all_forecasts)
        json_prediction_data = json.dumps(prediction_data)

        # --- INÍCIO DO TEMPLATE HTML COLADO ---
        html_template = """
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <title>Relatório Climático Completo</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-datalabels@2.2.0"></script>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 0; background-color: #0a192f; color: #e6f1ff; } .container { padding: 20px; } h1, h2, h3, h4 { color: #ccd6f6; } .tabs { display: flex; border-bottom: 2px solid #1a3d6e; margin-bottom: 20px; flex-wrap: wrap;} .tab-button { padding: 10px 20px; cursor: pointer; background-color: transparent; border: none; color: #8892b0; font-size: 16px; } .tab-button.active { color: #64ffda; border-bottom: 2px solid #64ffda; font-weight: bold; } .tab-content { display: none; } .tab-content.active { display: block; } .header, .map-header, .kpi-grid, .charts-grid { margin-bottom: 25px; } .header, .map-header { background-color: #112240; padding: 15px; border-radius: 8px; display: flex; gap: 20px; align-items: center; flex-wrap: wrap; border: 1px solid #1a3d6e;} .header label, .map-header label { font-weight: bold; margin-right: 5px; color: #8892b0; } .header input, .header select, .map-header select { background-color: #0a192f; border: 1px solid #1a3d6e; color: #e6f1ff; padding: 8px; border-radius: 5px; } .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; } .kpi-card { background-color: #112240; padding: 20px; border-radius: 8px; text-align: center; border: 1px solid #1a3d6e; } .kpi-card h4 { margin: 0 0 10px 0; color: #8892b0; font-size: 1em; text-transform: uppercase; } .kpi-card .value { font-size: 2.5em; font-weight: bold; color: #64ffda; } .charts-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(450px, 1fr)); gap: 20px; } .chart-card { background-color: #112240; padding: 20px; border-radius: 8px; border: 1px solid #1a3d6e; } .chart-card h3, .chart-card h4 { text-align: center; margin-top: 0; color: #ccd6f6; } .chart-canvas-wrapper { position: relative; height: 400px; } #map-container { height: 65vh; width: 100%; border-radius: 8px; border: 1px solid #1a3d6e; } .leaflet-popup-content-wrapper { background-color: #112240; color: #e6f1ff; border-radius: 5px; } .leaflet-popup-tip { background-color: #112240; } .station-icon { font-size: 1.5em; text-shadow: 0 0 3px black; }
        .calendar-container { max-width: 900px; margin: auto; background-color: #112240; padding: 20px; border-radius: 8px; border: 1px solid #1a3d6e;} .calendar-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; } .calendar-header button { background: #64ffda; color: #0a192f; border: none; padding: 5px 15px; border-radius: 5px; cursor: pointer; font-weight: bold; } .calendar-weekdays { display: grid; grid-template-columns: repeat(7, 1fr); text-align: center; font-weight: bold; color: #8892b0; margin-bottom: 10px; } .calendar-grid { display: grid; grid-template-columns: repeat(7, 1fr); gap: 5px; } .calendar-day { background-color: #0a192f; min-height: 95px; padding: 5px; border-radius: 4px; font-size: 0.9em; border: 1px solid #1a3d6e; cursor: pointer; transition: background-color 0.2s; display: flex; flex-direction: column; } .calendar-day:hover { background-color: #1a3d6e; } .calendar-day.empty { background-color: transparent; border: none; cursor: default; } .calendar-day.today { border-color: #64ffda; } .calendar-day.selected { background-color: #64ffda; color: #0a192f; } .day-number { font-weight: bold; margin-bottom: 4px; } .day-rainfall { font-size: 1.1em; color: #82d8c3; font-weight: bold; } #daily-details-container { margin-top: 30px; border-top: 2px solid #1a3d6e; padding-top: 20px;}
        .day-rainfall-details { margin-top: 5px; flex-grow: 1; overflow-y: auto; } .station-rain { display: flex; justify-content: space-between; font-size: 0.75em; color: #a8b2d1; padding: 1px 0; } .station-rain span { font-weight: bold; color: #8892b0; margin-right: 4px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 60px; }
        .spraying-window-container { display: flex; flex-direction: column; } .spraying-window-bar { display: flex; width: 100%; height: 80px; border-radius: 5px; overflow: hidden; border: 1px solid #1a3d6e; } .spray-hour { flex: 1; text-align: center; color: rgba(255,255,255,0.9); user-select: none; text-shadow: 1px 1px 2px rgba(0,0,0,0.5); display: flex; flex-direction: column; justify-content: center; align-items: center; padding: 4px 0; } .spray-hour-content { display: flex; flex-direction: column; gap: 2px; } .spray-hour-time { font-size: 1em; font-weight: bold; } .spray-hour-value { font-size: 0.75em; line-height: 1; }
        .spray-hour-tooltip { position: relative; } .spray-hour-tooltip .tooltip-text { visibility: hidden; width: 150px; background-color: #0a192f; color: #fff; text-align: center; border-radius: 6px; padding: 5px 0; position: absolute; z-index: 10; bottom: 115%; left: 50%; margin-left: -75px; opacity: 0; transition: opacity 0.3s; border: 1px solid #64ffda;} .spray-hour-tooltip:hover .tooltip-text { visibility: visible; opacity: 1; }
        .spraying-window-axis { display: flex; width: 100%; margin-top: 5px; } .axis-label { flex: 1; text-align: center; font-size: 0.75em; color: #8892b0; } .spray-legend { display: flex; justify-content: center; gap: 20px; margin-top: 15px; font-size: 0.9em; } .legend-item { display: flex; align-items: center; gap: 8px; } .legend-color-box { width: 15px; height: 15px; border-radius: 3px; }
        #spraying-summary { font-size: 0.9em; text-align: center; color: #a8b2d1; margin-top: 15px; background-color: #0a192f; padding: 10px; border-radius: 5px; border: 1px solid #1a3d6e;} #spraying-window-card { grid-column: 1 / -1; }
        #forecast-table-container { background-color: #112240; padding: 20px; border-radius: 8px; border: 1px solid #1a3d6e; margin-top:20px; } .forecast-table { width: 100%; border-collapse: collapse; color: #ccd6f6; } .forecast-table th, .forecast-table td { padding: 12px 15px; text-align: left; border-bottom: 1px solid #1a3d6e; } .forecast-table th { color: #8892b0; text-transform: uppercase; font-size: 0.85em; } .forecast-table tr:last-child td { border-bottom: none; } .forecast-table tr:hover { background-color: #1a3d6e; } .forecast-date { font-weight: bold; color: #64ffda; } .forecast-temp-max { color: #ffb3b3; } .forecast-temp-min { color: #a6d8f8; } .forecast-precip-prob { font-weight: bold; color: #82d8c3; }
        #hourly-forecast-wrapper { background-color: #112240; padding: 20px; border-radius: 8px; border: 1px solid #1a3d6e; margin-bottom: 20px; }
        .forecast-header { background-color: #112240; padding: 15px; border-radius: 8px; display: flex; gap: 20px; align-items: center; flex-wrap: wrap; border: 1px solid #1a3d6e; margin-bottom: 20px;} .forecast-header label { font-weight: bold; margin-right: 5px; color: #8892b0; } .forecast-header select { background-color: #0a192f; border: 1px solid #1a3d6e; color: #e6f1ff; padding: 8px; border-radius: 5px; }

        .alert-section { margin-bottom: 30px; border-bottom: 2px solid #1a3d6e; padding-bottom: 15px; }
        .alert-section:last-child { border-bottom: none; }
        .alert-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 15px; }
        .alert-card { background-color: #112240; border-left: 5px solid #ffc107; padding: 15px 20px; border-radius: 8px; display: flex; align-items: flex-start; gap: 15px; border: 1px solid #1a3d6e; }
        .alert-card-rain { border-left-color: #36a2eb; } .alert-card-gust { border-left-color: #ff6384; } .alert-card-temp_high { border-left-color: #fd7e14; } .alert-card-temp_low { border-left-color: #4bc0c0; } .alert-card-hum_low { border-left-color: #ffce56; } .alert-card-delta_t { border-left-color: #9966ff; }
        .alert-icon { font-size: 2em; line-height: 1; } .alert-body { flex-grow: 1; } .alert-body h4 { margin: 0 0 5px 0; color: #ccd6f6; } .alert-body p { margin: 2px 0; color: #a8b2d1; font-size: 0.9em; } .alert-body strong { color: #e6f1ff; font-weight: bold; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Relatório Climático - __GROWER_NAME__</h1>
        <div class="header">
            <div><label for="start-date">Data Início:</label><input type="date" id="start-date"></div>
            <div><label for="end-date">Data Fim:</label><input type="date" id="end-date"></div>
            <div><label for="station-filter">Estação (Dados Históricos):</label><select id="station-filter"><option value="todas">Todas</option></select></div>
        </div>
        <div class="tabs">
            <button class="tab-button active" onclick="openTab(event, 'tabChuva')">Resumo de Chuva</button>
            <button class="tab-button" onclick="openTab(event, 'tabTemperatura')">Temperaturas</button>
            <button class="tab-button" onclick="openTab(event, 'tabUmidade')">Umidade Relativa</button>
            <button class="tab-button" onclick="openTab(event, 'tabVento')">Ventos</button>
            <button class="tab-button" onclick="openTab(event, 'tabDeltaT')">Análise de Pulverização</button>
            <button class="tab-button" onclick="openTab(event, 'tabMonitoramento')">Monitoramento Diário</button>
            <button class="tab-button" onclick="openTab(event, 'tabAvisos')">🔔 Avisos Climatológicos</button>
            <button class="tab-button" onclick="openTab(event, 'tabMapa')">🗺️ Mapa Interativo</button>
            <button class="tab-button" onclick="openTab(event, 'tabPrevisao')">🔮 Previsão 10 Dias</button>
            <button class="tab-button" onclick="openTab(event, 'tabPredicao')">📈 Predição</button>
        </div>
        <div id="tabChuva" class="tab-content active"><div class="kpi-grid"><div class="kpi-card"><h4>Chuva Acumulada Média</h4><div class="value" id="kpi-chuva">0,0</div><span style="color:#8892b0;">mm</span></div><div class="kpi-card"><h4>Média Diária</h4><div class="value" id="kpi-chuva-media">0,0</div><span style="color:#8892b0;">mm/dia</span></div><div class="kpi-card"><h4>Máx. Chuva 24h</h4><div class="value" id="kpi-max-chuva-24h">0,0</div><span style="color:#8892b0;">mm</span></div><div class="kpi-card"><h4>Dias com Chuva</h4><div class="value" id="kpi-dias-chuva">0</div><span style="color:#8892b0;">(> 1mm)</span></div></div><div class="charts-grid"><div class="chart-card"><h3>Precipitação Diária (mm)</h3><div class="chart-canvas-wrapper"><canvas id="chartChuvaDiaria"></canvas></div></div><div class="chart-card"><h3>Precipitação Mensal (mm)</h3><div class="chart-canvas-wrapper"><canvas id="chartChuvaMensal"></canvas></div></div><div class="chart-card"><h3>Precipitação por Estação (mm)</h3><div class="chart-canvas-wrapper"><canvas id="chartChuvaEstacao"></canvas></div></div></div></div>
        <div id="tabTemperatura" class="tab-content"><div class="kpi-grid"><div class="kpi-card"><h4>Temp. Máxima</h4><div class="value" id="kpi-temp-max">0,0</div><span style="color:#8892b0;">°C</span></div><div class="kpi-card"><h4>Temp. Média</h4><div class="value" id="kpi-temp-media">0,0</div><span style="color:#8892b0;">°C</span></div><div class="kpi-card"><h4>Temp. Mínima</h4><div class="value" id="kpi-temp-min">0,0</div><span style="color:#8892b0;">°C</span></div></div><div class="charts-grid"><div class="chart-card"><h3>Temperaturas Diárias (°C)</h3><div class="chart-canvas-wrapper"><canvas id="chartTemperatura"></canvas></div></div></div></div>
        <div id="tabUmidade" class="tab-content"><div class="kpi-grid"><div class="kpi-card"><h4>Umidade Máxima</h4><div class="value" id="kpi-umidade-max">0,0</div><span style="color:#8892b0;">%</span></div><div class="kpi-card"><h4>Umidade Média</h4><div class="value" id="kpi-umidade-media">0,0</div><span style="color:#8892b0;">%</span></div><div class="kpi-card"><h4>Umidade Mínima</h4><div class="value" id="kpi-umidade-min">0,0</div><span style="color:#8892b0;">%</span></div></div><div class="charts-grid"><div class="chart-card"><h3>Umidade Relativa Diária (%)</h3><div class="chart-canvas-wrapper"><canvas id="chartUmidade"></canvas></div></div></div></div>
        <div id="tabVento" class="tab-content"><div class="kpi-grid"><div class="kpi-card"><h4>Vento Médio</h4><div class="value" id="kpi-vento-medio">0,0</div><span style="color:#8892b0;">km/h</span></div><div class="kpi-card"><h4>Rajada Máxima</h4><div class="value" id="kpi-rajada-max">0,0</div><span style="color:#8892b0;">km/h</span></div></div><div class="charts-grid"><div class="chart-card"><h3>Média de Vento Mensal (km/h)</h3><div class="chart-canvas-wrapper"><canvas id="chartVentoMensal"></canvas></div></div><div class="chart-card"><h3>Vento Médio e Rajada Máxima Diária (km/h)</h3><div class="chart-canvas-wrapper"><canvas id="chartVentoDiario"></canvas></div></div><div class="chart-card"><h3>Frequência da Direção do Vento</h3><div class="chart-canvas-wrapper"><canvas id="chartVentoDirecao"></canvas></div></div><div class="chart-card"><h3>Média de Vento por Hora do Dia (km/h)</h3><div class="chart-canvas-wrapper"><canvas id="chartVentoHorario"></canvas></div></div><div class="chart-card"><h3>Rosa dos Ventos</h3><div class="chart-canvas-wrapper"><canvas id="chartVentoRosa"></canvas></div></div></div></div>
        <div id="tabDeltaT" class="tab-content"><div class="charts-grid" style="grid-template-columns: repeat(auto-fit, minmax(450px, 1fr));"><div class="chart-card"><h3>Análise Mensal da Janela de Pulverização (% de Horas)</h3><div class="chart-canvas-wrapper"><canvas id="chartSprayConditionsByMonth"></canvas></div></div><div class="chart-card"><h3>Condições Médias por Hora (Vento e Delta T)</h3><div class="chart-canvas-wrapper"><canvas id="chartVentoDeltaTHorario"></canvas></div></div><div class="chart-card"><h3>Média de GFDI por Hora</h3><div class="chart-canvas-wrapper"><canvas id="chartGFDIHorario"></canvas></div></div></div></div>
        <div id="tabMonitoramento" class="tab-content"><div class="calendar-container"><div class="calendar-header"><button id="prev-month-btn">&lt; Mês Anterior</button><h2 id="month-year-header"></h2><button id="next-month-btn">Próximo Mês &gt;</button></div><div class="calendar-weekdays"><div>Dom</div><div>Seg</div><div>Ter</div><div>Qua</div><div>Qui</div><div>Sex</div><div>Sáb</div></div><div id="calendar-grid" class="calendar-grid"></div></div><div id="daily-details-container" style="display: none;"><h2 id="selected-day-header" style="text-align: center;"></h2><div class="charts-grid" style="grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));"><div class="chart-card"><h3>Condições de Vento e Delta T</h3><div class="chart-canvas-wrapper"><canvas id="chartVentoDeltaTDiario"></canvas></div></div><div class="chart-card"><h3>Condições Térmicas e de Umidade</h3><div class="chart-canvas-wrapper"><canvas id="chartTempUmidadeDiario"></canvas></div></div><div class="chart-card"><h3>Precipitação Horária (mm)</h3><div class="chart-canvas-wrapper"><canvas id="chartChuvaHoraria"></canvas></div></div><div class="chart-card"><h3>Rosa dos Ventos do Dia</h3><div class="chart-canvas-wrapper"><canvas id="chartVentoRosaDiario"></canvas></div></div><div class="chart-card" id="spraying-window-card"><h3>Janela de Pulverização do Dia</h3><div id="spraying-window-container"></div><div class="spray-legend"><div class="legend-item"><div class="legend-color-box" style="background-color:#28a745;"></div>Ideal</div><div class="legend-item"><div class="legend-color-box" style="background-color:#ffc107;"></div>Atenção</div><div class="legend-item"><div class="legend-color-box" style="background-color:#dc3545;"></div>Evitar</div><div class="legend-item"><div class="legend-color-box" style="background-color:#6c757d;"></div>S/ Dados</div></div><p id="spraying-summary"></p></div></div></div></div>
        <div id="tabAvisos" class="tab-content">
                 <div id="future-alerts-container"></div>
                 <div id="historical-alerts-container"></div>
        </div>
        <div id="tabMapa" class="tab-content"><div class="map-header"><label for="map-metric-selector">Visualizar no Mapa:</label><select id="map-metric-selector"><option value="chuva" selected>Chuva Acumulada (mm)</option><option value="temp_media">Temperatura Média (°C)</option><option value="umidade_media">Umidade Média (%)</option><option value="vento_medio">Vento Médio (km/h)</option><option value="rajada_max">Rajada Máxima (km/h)</option></select></div><div id="map-container"></div></div>
        <div id="tabPrevisao" class="tab-content">
            <div class="forecast-header">
                <label for="forecast-station-selector">Visualizar Previsão Para:</label>
                <select id="forecast-station-selector"></select>
            </div>
            <div id="hourly-forecast-wrapper">
                <h3>Previsão Horária (Próximas 48h)</h3>
                <div id="hourly-forecast-container"></div>
                <div class="spray-legend">
                    <div class="legend-item"><div class="legend-color-box" style="background-color:#28a745;"></div>Ideal</div>
                    <div class="legend-item"><div class="legend-color-box" style="background-color:#ffc107;"></div>Atenção</div>
                    <div class="legend-item"><div class="legend-color-box" style="background-color:#dc3545;"></div>Evitar</div>
                    <div class="legend-item"><div class="legend-color-box" style="background-color:#6c757d;"></div>S/ Dados</div>
                </div>
            </div>
            <div id="forecast-table-container">
                <h3>Previsão Resumida (Próximos 10 Dias)</h3>
                <table class="forecast-table">
                    <thead>
                        <tr>
                            <th>Dia</th><th>Mínima</th><th>Máxima</th><th>Condição</th>
                            <th>Prob. de Precip.</th><th>Quantidade</th><th>Vento (km/h)</th>
                        </tr>
                    </thead>
                    <tbody id="forecast-table-body"></tbody>
                </table>
            </div>
        </div>
        <div id="tabPredicao" class="tab-content">
            <div class="charts-grid">
                <div class="chart-card">
                    <h3>Previsão de Chuva - Próximos 12 Meses</h3>
                    <div class="chart-canvas-wrapper"><canvas id="chartPredicaoMensal"></canvas></div>
                </div>
                <div class="chart-card">
                    <h3>Previsão de Chuva - Próximas 6 Semanas</h3>
                    <div class="chart-canvas-wrapper"><canvas id="chartPredicaoSemanal"></canvas></div>
                </div>
            </div>
        </div>
    </div>
    <script id="dados-climaticos" type="application/json">__JSON_DATA__</script>
    <script id="dados-geograficos" type="application/json">__GEODATA__</script>
    <script id="dados-todas-previsoes" type="application/json">__JSON_ALL_FORECASTS__</script>
    <script id="dados-predicao-ts" type="application/json">__JSON_PREDICTION_DATA__</script>
    <script>
        const MESES_PT_BR = ["Janeiro", "Fevereiro", "Março", "Abril", "Maio", "Junho", "Julho", "Agosto", "Setembro", "Outubro", "Novembro", "Dezembro"]; const CARDINAL_DIRECTIONS = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']; const SPRAY_COLORS = { Ideal: '#28a745', Atenção: '#ffc107', Evitar: '#dc3545', NoData: '#6c757d' }; let map, geoData, allData, allForecastData, charts = {}; let fieldLayers = {}, stationMarkers = {}, mapLegend; let calendarDate = new Date(); let currentFilteredData = []; let currentDailyAggregated = []; let selectedCalendarDay = null; let stationColors = {};
        const mapMetricsConfig = { chuva: { key: 'precipitacao_mm', agg: 'sum', label: 'Chuva Acumulada', unit: 'mm', colors: ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b'] }, temp_media: { key: 'temp_media_c', agg: 'avg', label: 'Temperatura Média', unit: '°C', colors: ['#fff5f0', '#fee0d2', '#fcbba1', '#fc9272', '#fb6a4a', '#ef3b2c', '#cb181d', '#a50f15', '#67000d'] }, umidade_media: { key: 'umidade_media_perc', agg: 'avg', label: 'Umidade Média', unit: '%', colors: ['#f7fcf5', '#e5f5e0', '#c7e9c0', '#a1d99b', '#74c476', '#41ab5d', '#238b45', '#006d2c', '#00441b'] }, vento_medio: { key: 'vento_medio_kph', agg: 'avg', label: 'Vento Médio', unit: 'km/h', colors: ['#fcfbfd', '#efedf5', '#dadaeb', '#bcbddc', '#9e9ac8', '#807dba', '#6a51a3', '#54278f', '#3f007d'] }, rajada_max: { key: 'rajada_max_kph', agg: 'max', label: 'Rajada Máxima', unit: 'km/h', colors: ['#ffffe5', '#fff7bc', '#fee391', '#fec44f', '#fe9929', '#ec7014', '#cc4c02', '#993404', '#662506'] } };
        
        const ALERT_THRESHOLDS = { RAIN_LIMIT: 50, GUST_LIMIT: 50, TEMP_HIGH: 40, TEMP_LOW: 5, HUM_LOW: 20, DELTA_T_HIGH: 9 };

        function openTab(evt, tabName) { document.querySelectorAll('.tab-content').forEach(tc => tc.classList.remove('active')); document.querySelectorAll('.tab-button').forEach(tb => tb.classList.remove('active')); document.getElementById(tabName).classList.add('active'); evt.currentTarget.classList.add('active'); if (tabName === 'tabMapa' && map) { setTimeout(() => map.invalidateSize(), 10); } }
        function degreesToCardinal(deg) { if (deg === null || isNaN(deg)) return null; return CARDINAL_DIRECTIONS[Math.round(deg / 22.5) % 16]; }
        function fNum(value, decimals = 1) { if (typeof value !== 'number' || isNaN(value)) return 'N/D'; return value.toLocaleString('pt-BR', { minimumFractionDigits: decimals, maximumFractionDigits: decimals }); }
        function getSprayingCondition(wind, deltaT) { if (wind === null || isNaN(wind) || deltaT === null || isNaN(deltaT)) return 'NoData'; if (wind > 9 || deltaT > 10) return 'Evitar'; if ((wind >= 2 && wind <= 8) && (deltaT >= 2 && deltaT <= 10)) return 'Ideal'; return 'Atenção'; }
        
        // --- FUNÇÕES DE PREVISÃO ATUALIZADAS ---
        function updateForecastDisplay() {
            const selectedStation = document.getElementById('forecast-station-selector').value;
            renderForecastTable(selectedStation);
            renderHourlyForecast(selectedStation);
        }

        function getAverageForecast(forecastType) {
            const stationNames = Object.keys(allForecastData[forecastType]);
            if (stationNames.length === 0) return [];
            
            const avgForecast = [];
            const firstStationData = allForecastData[forecastType][stationNames[0]];
            if (!firstStationData) return [];

            const numEntries = firstStationData.length;

            for (let i = 0; i < numEntries; i++) {
                const valuesToAvg = {};
                
                stationNames.forEach(name => {
                    const stationData = allForecastData[forecastType][name];
                    if (stationData && stationData[i]) {
                        for (const key in stationData[i]) {
                            const value = stationData[i][key];
                            if (typeof value === 'number') {
                                if (!valuesToAvg[key]) valuesToAvg[key] = [];
                                valuesToAvg[key].push(value);
                            }
                        }
                    }
                });

                const avgEntry = { ...firstStationData[i] }; // Pega a estrutura e dados não numéricos
                for (const key in valuesToAvg) {
                    if (valuesToAvg[key].length > 0) {
                        avgEntry[key] = valuesToAvg[key].reduce((a, b) => a + b, 0) / valuesToAvg[key].length;
                    }
                }
                avgForecast.push(avgEntry);
            }
            return avgForecast;
        }

        function renderForecastTable(station) {
            const tableBody = document.getElementById('forecast-table-body');
            tableBody.innerHTML = '';
            
            let forecastData;
            if (station === 'average') {
                forecastData = getAverageForecast('daily');
            } else {
                forecastData = allForecastData.daily[station];
            }
            
            if (!forecastData || forecastData.length === 0) {
                tableBody.innerHTML = '<tr><td colspan="7" style="text-align:center;">Não foi possível carregar a previsão do tempo.</td></tr>';
                return;
            }
            forecastData.forEach(day => {
                const row = document.createElement('tr');
                row.innerHTML = `<td><div class="forecast-date">${day.dia_semana}, ${day.data}</div></td><td class="forecast-temp-min">${fNum(day.min_temp, 0)}°C</td><td class="forecast-temp-max">${fNum(day.max_temp, 0)}°C</td><td>${day.descricao || 'N/D'}</td><td class="forecast-precip-prob">${fNum(day.prob_precip,0)}%</td><td>${fNum(day.qtd_precip, 2)} mm</td><td>${fNum(day.vento_vel, 0)} ${day.vento_dir || ''}</td>`;
                tableBody.appendChild(row);
            });
        }

        function renderHourlyForecast(station) {
            const container = document.getElementById('hourly-forecast-container');
            container.innerHTML = '';

            let hourlyData;
            if (station === 'average') {
                hourlyData = getAverageForecast('hourly');
            } else {
                hourlyData = allForecastData.hourly[station];
            }
            
            if (!hourlyData || hourlyData.length === 0) {
                container.innerHTML = '<p style="text-align:center;">Não foi possível carregar a previsão horária.</p>';
                return;
            }

            const todayStr = hourlyData.length > 0 ? hourlyData[0].fcst_valid_local.substring(0, 10) : new Date().toISOString().substring(0, 10);
            const todayData = hourlyData.filter(d => d.fcst_valid_local.startsWith(todayStr));
            const tomorrowData = hourlyData.filter(d => !d.fcst_valid_local.startsWith(todayStr));
            
            const createHourlyBar = (data, title) => {
                if(data.length === 0) return;
                const dayWrapper = document.createElement('div');
                dayWrapper.style.marginBottom = '20px';
                const titleEl = document.createElement('h4');
                titleEl.textContent = title;
                titleEl.style.textAlign = 'center'; titleEl.style.color = '#8892b0'; titleEl.style.marginBottom = '10px';
                const windowContainer = document.createElement('div');
                windowContainer.className = 'spraying-window-container';
                const barDiv = document.createElement('div');
                barDiv.className = 'spraying-window-bar';
                const axisDiv = document.createElement('div');
                axisDiv.className = 'spraying-window-axis';
                const dataMap = new Map(data.map(d => [parseInt(d.fcst_valid_local.substring(11, 13)), d]));
                for (let h = 0; h < 24; h++) {
                    const hourData = dataMap.get(h);
                    const wind = hourData ? hourData.wspd : null; const deltaT = hourData ? hourData.delta_t : null; const temp = hourData ? hourData.temp : null; const rh = hourData ? hourData.rh : null; const pop = hourData ? hourData.pop : null;
                    const condition = getSprayingCondition(wind, deltaT);
                    const hourDiv = document.createElement('div');
                    hourDiv.className = 'spray-hour spray-hour-tooltip';
                    hourDiv.style.backgroundColor = SPRAY_COLORS[condition];
                    const windText = (wind !== null && !isNaN(wind)) ? `${fNum(wind, 1)} km/h` : 'N/D'; const deltaTText = (deltaT !== null && !isNaN(deltaT)) ? `${fNum(deltaT, 1)}` : 'N/D'; const tempText = (temp !== null && !isNaN(temp)) ? `${fNum(temp, 0)}°C` : 'N/D';
                    hourDiv.innerHTML = `<div class="spray-hour-content">
                        <div class="spray-hour-time">${h}h</div>
                        <div class="spray-hour-value">🌡️ ${tempText}</div>
                        <div class="spray-hour-value">🌬️ ${windText}</div>
                        <div class="spray-hour-value">ΔT: ${deltaTText}</div>
                    </div>
                    <span class="tooltip-text"><b>Hora: ${String(h).padStart(2,'0')}:00</b><br>Temp: ${tempText}<br>Umidade: ${rh !== null ? fNum(rh,0) + '%' : 'N/D'}<br>Vento: ${windText}<br>ΔT: ${deltaTText} °C<br>Chuva: ${pop !== null ? fNum(pop,0) + '%' : 'N/D'}</span>`;
                    barDiv.appendChild(hourDiv);
                    const axisLabel = document.createElement('div');
                    axisLabel.className = 'axis-label';
                    if (h % 3 === 0) axisLabel.innerText = `${String(h).padStart(2, '0')}h`;
                    axisDiv.appendChild(axisLabel);
                }
                windowContainer.appendChild(barDiv); windowContainer.appendChild(axisDiv); dayWrapper.appendChild(titleEl); dayWrapper.appendChild(windowContainer); container.appendChild(dayWrapper);
            };
            createHourlyBar(todayData, 'Hoje');
            createHourlyBar(tomorrowData, 'Amanhã');
        }

        function renderPredictionCharts() {
            const predictionData = JSON.parse(document.getElementById('dados-predicao-ts').textContent);
            const modelColors = { 'Prophet': '#36a2eb', 'ARIMA': '#ff6384', 'Regressão Linear': '#4bc0c0', 'Média Histórica': '#ffce56' };
            const commonPredictionOptions = { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'top', labels: {color: '#e6f1ff'} } }, scales: { x: { ticks: { color: '#a8b2d1' }, grid: { color: 'rgba(136, 146, 176, 0.1)' } }, y: { ticks: { color: '#a8b2d1' }, grid: { color: 'rgba(136, 146, 176, 0.1)' }, beginAtZero: true, title: { display: true, text: 'Precipitação (mm)', color: '#8892b0'} } } };
            if (predictionData && predictionData.mensal && predictionData.mensal.labels) {
                const dataMensal = predictionData.mensal; const datasetsMensal = [];
                for (const modelName in dataMensal.previsoes) { if (modelName === 'Média Histórica') { datasetsMensal.push({ type: 'line', label: modelName, data: dataMensal.previsoes[modelName], borderColor: modelColors[modelName] || 'yellow', backgroundColor: 'transparent', tension: 0.1, pointRadius: 4, order: 0 }); } else { datasetsMensal.push({ type: 'bar', label: modelName, data: dataMensal.previsoes[modelName], backgroundColor: modelColors[modelName] || '#cccccc', order: 1 }); } }
                charts.predicaoMensal = new Chart(document.getElementById('chartPredicaoMensal'), { data: { labels: dataMensal.labels, datasets: datasetsMensal }, options: commonPredictionOptions });
            } else { document.getElementById('chartPredicaoMensal').parentElement.innerHTML = '<p style="text-align:center; padding-top:20px;">Dados insuficientes para gerar a predição mensal.</p>'; }
            if (predictionData && predictionData.semanal && predictionData.semanal.labels) {
                const dataSemanal = predictionData.semanal; const datasetsSemanal = [];
                for (const modelName in dataSemanal.previsoes) { datasetsSemanal.push({ type: 'line', label: modelName, data: dataSemanal.previsoes[modelName], borderColor: modelColors[modelName] || '#4bc0c0', backgroundColor: (modelColors[modelName] || '#4bc0c0').replace(')', ', 0.2)').replace('rgb', 'rgba'), fill: true, tension: 0.2 }); }
                charts.predicaoSemanal = new Chart(document.getElementById('chartPredicaoSemanal'), { data: { labels: dataSemanal.labels, datasets: datasetsSemanal }, options: commonPredictionOptions });
            } else { document.getElementById('chartPredicaoSemanal').parentElement.innerHTML = '<p style="text-align:center; padding-top:20px;">Dados insuficientes para gerar a predição semanal.</p>'; }
        }
        
        document.addEventListener('DOMContentLoaded', function() {
            Chart.register(ChartDataLabels); Chart.defaults.plugins.datalabels.display = false; Chart.defaults.color = '#a8b2d1'; Chart.defaults.borderColor = 'rgba(136, 146, 176, 0.2)';
            allData = JSON.parse(document.getElementById('dados-climaticos').textContent).map(d => { d.datetime = new Date(d.datetime); return d; });
            geoData = JSON.parse(document.getElementById('dados-geograficos').textContent);
            allForecastData = JSON.parse(document.getElementById('dados-todas-previsoes').textContent);

            function iniciarDashboard() { if (allData.length === 0 && (!geoData || !geoData.fields || geoData.fields.length === 0)) { document.querySelector('.container').innerHTML = '<h1>Nenhum dado encontrado para gerar o relatório.</h1>'; return; }; if (allData.length > 0) { const stationFilter = document.getElementById('station-filter'); const forecastStationFilter = document.getElementById('forecast-station-selector'); const uniqueStations = [...new Set(allData.map(d => d.nome_estacao))]; if(uniqueStations.length > 1) forecastStationFilter.innerHTML = '<option value="average">Média Geral</option>'; uniqueStations.forEach((name, index) => { const optionHtml = `<option value="${name}">${name}</option>`; stationFilter.innerHTML += optionHtml; forecastStationFilter.innerHTML += optionHtml; stationColors[name] = Chart.getSpacedColors(uniqueStations.length)[index]; }); const minDate = new Date(Math.min(...allData.map(d=>d.datetime))); const maxDate = new Date(Math.max(...allData.map(d=>d.datetime))); document.getElementById('start-date').valueAsDate = new Date(minDate.getUTCFullYear(), minDate.getUTCMonth(), minDate.getUTCDate()); document.getElementById('end-date').valueAsDate = new Date(maxDate.getUTCFullYear(), maxDate.getUTCMonth(), maxDate.getUTCDate()); calendarDate = maxDate; } const commonOptions = { responsive: true, maintainAspectRatio: false, plugins: { title: {display: false}, legend: { position: 'top', labels: {color: '#e6f1ff'} } }, scales: { x: { ticks: { color: '#a8b2d1' }, grid: { color: 'rgba(136, 146, 176, 0.1)' } }, y: { ticks: { color: '#a8b2d1' }, grid: { color: 'rgba(136, 146, 176, 0.1)' }, beginAtZero: true } } }; const stackedOptions = { ...commonOptions, scales: { x: {...commonOptions.scales.x, stacked: true}, y:{...commonOptions.scales.y, stacked: true, beginAtZero: true} } }; charts.chuvaDiaria = new Chart(document.getElementById('chartChuvaDiaria'), { type: 'bar', options: stackedOptions }); charts.chuvaMensal = new Chart(document.getElementById('chartChuvaMensal'), { type: 'bar', options: { ...stackedOptions, scales: { ...stackedOptions.scales, x: { ...stackedOptions.scales.x, ticks: { ...stackedOptions.scales.x.ticks, callback: function(value) { const label = this.getLabelForValue(value); const [year, monthNum] = label.split('-'); return MESES_PT_BR[parseInt(monthNum) - 1].substring(0,3) + ' ' + year; } } } } } }); charts.chuvaEstacao = new Chart(document.getElementById('chartChuvaEstacao'), { type: 'bar', options: { ...commonOptions, plugins: { ...commonOptions.plugins, datalabels: { display: true, anchor: 'end', align: 'top', formatter: v => fNum(v,0), color: 'white' } }, layout: { padding: { top: 30 } } } }); charts.temperatura = new Chart(document.getElementById('chartTemperatura'), { type: 'line', options: commonOptions }); charts.umidade = new Chart(document.getElementById('chartUmidade'), { type: 'line', options: commonOptions }); charts.ventoMensal = new Chart(document.getElementById('chartVentoMensal'), { type: 'bar', options: {...commonOptions, scales: {...commonOptions.scales, x: {...commonOptions.scales.x, offset: true }}} }); charts.ventoDiario = new Chart(document.getElementById('chartVentoDiario'), { type: 'line', options: commonOptions }); charts.ventoDirecao = new Chart(document.getElementById('chartVentoDirecao'), { type: 'doughnut', options: { ...commonOptions, plugins: { ...commonOptions.plugins, datalabels: { display: true, color: 'white', formatter: (v, ctx) => `${fNum(v,1)}%`} } } }); charts.ventoHorario = new Chart(document.getElementById('chartVentoHorario'), { type: 'line', options: { ...commonOptions, scales: { ...commonOptions.scales, y: { beginAtZero: true } } } }); charts.ventoRosa = new Chart(document.getElementById('chartVentoRosa'), { type: 'polarArea', options: { ...commonOptions, scales: { r: { ticks: { backdropColor: 'rgba(0,0,0,0.5)', color: 'white' } } } } }); charts.sprayConditionsByMonth = new Chart(document.getElementById('chartSprayConditionsByMonth'), { type: 'bar', options: { ...stackedOptions, scales: { ...stackedOptions.scales, y: { ...stackedOptions.scales.y, ticks: { callback: v => v + '%' } } } } }); charts.ventoDeltaTHorario = new Chart(document.getElementById('chartVentoDeltaTHorario'), {type: 'line', options: {...commonOptions, scales: { x: commonOptions.scales.x, y_deltat: { type: 'linear', position: 'left', title: { display: true, text: 'Delta T (°C)', color: '#ff6384' } }, y_vento: { type: 'linear', position: 'right', title: { display: true, text: 'Vento (km/h)', color: '#36a2eb' }, grid: { drawOnChartArea: false } } } } }); charts.gfdiHorario = new Chart(document.getElementById('chartGFDIHorario'), { type: 'line', options: { ...commonOptions, scales: { ...commonOptions.scales, y: { beginAtZero: false, title: { display: true, text: 'GFDI'} } } } }); charts.chuvaHoraria = new Chart(document.getElementById('chartChuvaHoraria'), {type: 'bar', options: {...commonOptions, plugins: {...commonOptions.plugins, legend: { display: false}}}}); charts.ventoDeltaTDiario = new Chart(document.getElementById('chartVentoDeltaTDiario'), {type: 'line', options: {...commonOptions, scales: { x: commonOptions.scales.x, y_deltat: { type: 'linear', position: 'left', title: { display: true, text: 'Delta T (°C)', color: '#ff6384' } }, y_vento: { type: 'linear', position: 'right', title: { display: true, text: 'Vento (km/h)', color: '#36a2eb' }, grid: { drawOnChartArea: false } } } }}); charts.tempUmidadeDiario = new Chart(document.getElementById('chartTempUmidadeDiario'), {type: 'line', options: {...commonOptions, scales: { x: commonOptions.scales.x, y_temp: { type: 'linear', position: 'left', title: { display: true, text: 'Temperatura (°C)', color: '#ff9f40' } }, y_rh: { type: 'linear', position: 'right', title: { display: true, text: 'Umidade (%)', color: '#4bc0c0' }, grid: { drawOnChartArea: false } } } }}); charts.ventoRosaDiario = new Chart(document.getElementById('chartVentoRosaDiario'), { type: 'polarArea', options: { ...commonOptions, plugins: {...commonOptions.plugins, legend: {position: 'right'}}, scales: { r: { ticks: { backdropColor: 'rgba(0,0,0,0.5)', color: 'white' } } } } }); document.getElementById('start-date').addEventListener('change', atualizarTudo); document.getElementById('end-date').addEventListener('change', atualizarTudo); document.getElementById('station-filter').addEventListener('change', atualizarTudo); document.getElementById('forecast-station-selector').addEventListener('change', updateForecastDisplay); document.getElementById('map-metric-selector').addEventListener('change', atualizarMapa); document.getElementById('prev-month-btn').addEventListener('click', () => { calendarDate.setUTCMonth(calendarDate.getUTCMonth() - 1); renderCalendar(calendarDate); }); document.getElementById('next-month-btn').addEventListener('click', () => { calendarDate.setUTCMonth(calendarDate.getUTCMonth() + 1); renderCalendar(calendarDate); }); iniciarMapa(); atualizarTudo(); renderPredictionCharts(); }
            function atualizarTudo() { const startDate = new Date(document.getElementById('start-date').value + "T00:00:00Z"); const endDate = new Date(document.getElementById('end-date').value + "T23:59:59Z"); const selectedStation = document.getElementById('station-filter').value; currentFilteredData = allData.filter(d => { const stationMatch = (selectedStation === 'todas' || d.nome_estacao === selectedStation); const dateMatch = d.datetime >= startDate && d.datetime <= endDate; return stationMatch && dateMatch; }); const dailyData = {}; currentFilteredData.forEach(d => { const day = d.datetime.toISOString().split('T')[0]; if (!dailyData[day]) { dailyData[day] = {precip_by_station: {}, temp_min_c: [], temp_max_c: [], temp_media_c: [], umidade_min_perc: [], umidade_max_perc: [], umidade_media_perc: [], vento_medio_kph: [], rajada_max_kph: []}; } if(d.precipitacao_mm > 0) { dailyData[day].precip_by_station[d.nome_estacao] = (dailyData[day].precip_by_station[d.nome_estacao] || 0) + d.precipitacao_mm; } if(d.temp_min_c !== null) dailyData[day].temp_min_c.push(d.temp_min_c); if(d.temp_max_c !== null) dailyData[day].temp_max_c.push(d.temp_max_c); if(d.temp_media_c !== null) dailyData[day].temp_media_c.push(d.temp_media_c); if(d.umidade_min_perc !== null) dailyData[day].umidade_min_perc.push(d.umidade_min_perc); if(d.umidade_max_perc !== null) dailyData[day].umidade_max_perc.push(d.umidade_max_perc); if(d.umidade_media_perc !== null) dailyData[day].umidade_media_perc.push(d.umidade_media_perc); if(d.vento_medio_kph !== null) dailyData[day].vento_medio_kph.push(d.vento_medio_kph); if(d.rajada_max_kph !== null) dailyData[day].rajada_max_kph.push(d.rajada_max_kph); }); const avg = (arr) => arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : NaN; currentDailyAggregated = Object.keys(dailyData).sort().map(day => { const d = dailyData[day]; const numStations = (document.getElementById('station-filter').value === 'todas') ? (Object.keys(stationColors).length || 1) : 1; const totalPrecip = Object.values(d.precip_by_station).reduce((a,b) => a+b, 0); return { data_str: day, precip_by_station: d.precip_by_station, precipitacao_mm: totalPrecip, precipitacao_media_mm: totalPrecip / numStations, temp_min_c: Math.min(...d.temp_min_c.filter(v => v !== null)), temp_max_c: Math.max(...d.temp_max_c.filter(v => v !== null)), temp_media_c: avg(d.temp_media_c.filter(v => v !== null)), umidade_min_perc: Math.min(...d.umidade_min_perc.filter(v => v !== null)), umidade_max_perc: Math.max(...d.umidade_max_perc.filter(v => v !== null)), umidade_media_perc: avg(d.umidade_media_perc.filter(v => v !== null)), vento_medio_kph: avg(d.vento_medio_kph.filter(v => v !== null)), rajada_max_kph: Math.max(0, ...d.rajada_max_kph.filter(v => v !== null)) }; }); 
                atualizarGraficos(currentFilteredData, currentDailyAggregated); 
                atualizarMapa(); 
                renderCalendar(calendarDate); 
                generateAndRenderHistoricalAlerts(currentFilteredData);
                generateAndRenderFutureAlerts();
                updateForecastDisplay();
                document.getElementById('daily-details-container').style.display = 'none'; 
                selectedCalendarDay = null; 
            }
            function atualizarGraficos(data, dailyAggregated) { if(allData.length > 0 && data.length === 0) { return; } const stationsInFilter = [...new Set(data.map(d => d.nome_estacao))].sort(); const dateLabels = dailyAggregated.map(d => d.data_str); const rainByStation = {}; data.forEach(d => { if(d.precipitacao_mm > 0) rainByStation[d.nome_estacao] = (rainByStation[d.nome_estacao] || 0) + d.precipitacao_mm; }); const stationsWithRain = Object.values(rainByStation); const avgAccumulatedRain = stationsWithRain.length > 0 ? stationsWithRain.reduce((a,b) => a+b, 0) / stationsWithRain.length : 0; const maxChuva24h = Math.max(0, ...dailyAggregated.map(d => Math.max(0, ...Object.values(d.precip_by_station)))); document.getElementById('kpi-chuva').innerText = fNum(avgAccumulatedRain); document.getElementById('kpi-chuva-media').innerText = fNum(dailyAggregated.reduce((s, d) => s + d.precipitacao_media_mm, 0) / (dailyAggregated.length || 1)); document.getElementById('kpi-max-chuva-24h').innerText = fNum(maxChuva24h); document.getElementById('kpi-dias-chuva').innerText = dailyAggregated.filter(d => d.precipitacao_media_mm > 1).length; const validTemps = dailyAggregated.filter(d => !isNaN(d.temp_media_c)); if (validTemps.length > 0) { document.getElementById('kpi-temp-max').innerText = fNum(Math.max(...validTemps.map(d => d.temp_max_c))); document.getElementById('kpi-temp-media').innerText = fNum(validTemps.reduce((s, d) => s + d.temp_media_c, 0) / validTemps.length); document.getElementById('kpi-temp-min').innerText = fNum(Math.min(...validTemps.map(d => d.temp_min_c))); } const validHumidity = dailyAggregated.filter(d => !isNaN(d.umidade_media_perc)); if (validHumidity.length > 0) { document.getElementById('kpi-umidade-max').innerText = fNum(Math.max(...validHumidity.map(d => d.umidade_max_perc)), 0); document.getElementById('kpi-umidade-media').innerText = fNum(validHumidity.reduce((s,d)=>s+d.umidade_media_perc,0)/validHumidity.length, 0); document.getElementById('kpi-umidade-min').innerText = fNum(Math.min(...validHumidity.map(d => d.umidade_min_perc)), 0); }
            charts.chuvaDiaria.data.labels = dateLabels; charts.chuvaDiaria.data.datasets = stationsInFilter.map(station => ({ label: station, data: dailyAggregated.map(day => day.precip_by_station[station] || 0), backgroundColor: stationColors[station] || '#64ffda', })); charts.chuvaDiaria.update(); const monthlyRain = {}; dailyAggregated.forEach(d => { const month = d.data_str.substring(0, 7); if (!monthlyRain[month]) monthlyRain[month] = {}; for(const station in d.precip_by_station){ monthlyRain[month][station] = (monthlyRain[month][station] || 0) + d.precip_by_station[station]; } }); const monthlyLabels = Object.keys(monthlyRain).sort(); charts.chuvaMensal.data.labels = monthlyLabels; charts.chuvaMensal.data.datasets = stationsInFilter.map(station => ({ label: station, data: monthlyLabels.map(month => (monthlyRain[month] && monthlyRain[month][station]) || 0), backgroundColor: stationColors[station] || '#64ffda', })); charts.chuvaMensal.update(); const dataEstacao = {}; data.forEach(d => { dataEstacao[d.nome_estacao] = (dataEstacao[d.nome_estacao] || 0) + (d.precipitacao_mm || 0); }); charts.chuvaEstacao.data.labels = Object.keys(dataEstacao); charts.chuvaEstacao.data.datasets = [{ label: 'Precipitação Total (mm)', data: Object.values(dataEstacao), backgroundColor: Object.keys(dataEstacao).map(s => stationColors[s]) }]; charts.chuvaEstacao.update();
            charts.temperatura.data.labels = dateLabels; charts.temperatura.data.datasets = [ { label: 'Temp. Máxima (°C)', data: dailyAggregated.map(d=>d.temp_max_c), borderColor: '#ff6384', fill: false, tension: 0.1 }, { label: 'Temp. Média (°C)', data: dailyAggregated.map(d=>d.temp_media_c), borderColor: '#ffce56', fill: false, tension: 0.1, borderDash: [5, 5] }, { label: 'Temp. Mínima (°C)', data: dailyAggregated.map(d=>d.temp_min_c), borderColor: '#36a2eb', fill: true, backgroundColor: 'rgba(54, 162, 235, 0.2)', tension: 0.1 } ]; charts.temperatura.update(); charts.umidade.data.labels = dateLabels; charts.umidade.data.datasets = [ { label: 'Umidade Máxima (%)', data: dailyAggregated.map(d=>d.umidade_max_perc), borderColor: '#4bc0c0', fill: false, tension: 0.1 }, { label: 'Umidade Média (%)', data: dailyAggregated.map(d=>d.umidade_media_perc), borderColor: '#9966ff', fill: false, tension: 0.1, borderDash: [5, 5] }, { label: 'Umidade Mínima (%)', data: dailyAggregated.map(d=>d.umidade_min_perc), borderColor: '#c9cbcf', fill: true, backgroundColor: 'rgba(75, 192, 192, 0.2)', tension: 0.1 } ]; charts.umidade.update(); const validWind = data.filter(d => d.vento_medio_kph !== null && !isNaN(d.vento_medio_kph)); const validGust = data.filter(d => d.rajada_max_kph !== null && !isNaN(d.rajada_max_kph)); document.getElementById('kpi-vento-medio').innerText = fNum(validWind.reduce((s,d)=>s + d.vento_medio_kph, 0) / (validWind.length || 1)); document.getElementById('kpi-rajada-max').innerText = fNum(Math.max(0, ...validGust.map(d=>d.rajada_max_kph))); const dataVentoMensal = {}; data.forEach(d=> { if (d.vento_medio_kph !== null && !isNaN(d.vento_medio_kph)) { const month = d.datetime.getUTCFullYear() + '-' + String(d.datetime.getUTCMonth()+1).padStart(2,'0'); if(!dataVentoMensal[month]) dataVentoMensal[month] = []; dataVentoMensal[month].push(d.vento_medio_kph); }}); const labelsVentoMensal = Object.keys(dataVentoMensal).sort(); charts.ventoMensal.data.labels = labelsVentoMensal.map(l => { const [y,m] = l.split('-'); return `${MESES_PT_BR[m-1].substring(0,3)} ${y}`; }); charts.ventoMensal.data.datasets = [{ label: 'Vento Médio (km/h)', data: labelsVentoMensal.map(l => dataVentoMensal[l].reduce((a,b)=>a+b,0)/dataVentoMensal[l].length), backgroundColor: '#36a2eb' }]; charts.ventoMensal.update(); charts.ventoDiario.data.labels = dateLabels; charts.ventoDiario.data.datasets = [ { label: 'Rajada Máxima (km/h)', data: dailyAggregated.map(d => d.rajada_max_kph), borderColor: '#4bc0c0', fill: false, tension: 0.4 }, { label: 'Vento Médio (km/h)', data: dailyAggregated.map(d => d.vento_medio_kph), borderColor: '#36a2eb', fill: true, backgroundColor: 'rgba(54, 162, 235, 0.1)', tension: 0.4 } ]; charts.ventoDiario.update(); const direcaoCounts = {}; CARDINAL_DIRECTIONS.forEach(dir => direcaoCounts[dir] = 0); let totalDirecoes = 0; data.forEach(d => { const cardinal = degreesToCardinal(d.vento_direcao_graus); if (cardinal) { direcaoCounts[cardinal]++; totalDirecoes++; } }); charts.ventoDirecao.data.labels = CARDINAL_DIRECTIONS; charts.ventoDirecao.data.datasets = [{ label: 'Frequência (%)', data: CARDINAL_DIRECTIONS.map(d => (direcaoCounts[d]/(totalDirecoes || 1))*100), backgroundColor: Chart.getSpacedColors(16) }]; charts.ventoDirecao.update(); const horarioCounts = Array(24).fill(0).map(()=>[]); data.forEach(d => { if(d.vento_medio_kph !== null && !isNaN(d.vento_medio_kph)) horarioCounts[d.datetime.getUTCHours()].push(d.vento_medio_kph); }); charts.ventoHorario.data.labels = Array(24).fill(0).map((_,i)=>`${String(i).padStart(2,'0')}:00`); charts.ventoHorario.data.datasets = [{ label: 'Vento Médio (km/h)', data: horarioCounts.map(h=>h.length > 0 ? h.reduce((a,b)=>a+b,0)/h.length : 0), borderColor:'#9966ff', backgroundColor: 'rgba(153, 102, 255, 0.2)', fill: true, tension: 0.4 }]; charts.ventoHorario.update(); const speedBrackets = [[0,3], [3,6], [6,9], [9,100]]; const roseData = {}; CARDINAL_DIRECTIONS.forEach(dir => roseData[dir] = Array(speedBrackets.length).fill(0)); let totalVentos = 0; data.forEach(d => { const cardinal = degreesToCardinal(d.vento_direcao_graus); const speed = d.vento_medio_kph; if(cardinal && speed >= 0) { totalVentos++; for(let i=0; i<speedBrackets.length; i++) { if(speed >= speedBrackets[i][0] && speed < speedBrackets[i][1]) { roseData[cardinal][i]++; break; } } }}); charts.ventoRosa.data.labels = CARDINAL_DIRECTIONS; charts.ventoRosa.data.datasets = speedBrackets.map((bracket, i) => ({ label: `[${bracket[0]},${bracket[1]}) km/h`, data: CARDINAL_DIRECTIONS.map(dir => (roseData[dir][i]/(totalVentos || 1))*100) })); charts.ventoRosa.update();
            const monthlyConditions = {}; data.forEach(d => { const monthKey = d.datetime.getUTCFullYear() + '-' + String(d.datetime.getUTCMonth() + 1).padStart(2, '0'); if (!monthlyConditions[monthKey]) monthlyConditions[monthKey] = { Ideal: 0, Atenção: 0, Evitar: 0, NoData: 0 }; const condition = getSprayingCondition(d.vento_medio_kph, d.delta_t); monthlyConditions[monthKey][condition]++; }); const monthlyLabelsSpray = Object.keys(monthlyConditions).sort(); charts.sprayConditionsByMonth.data.labels = monthlyLabelsSpray.map(l => { const [y, m] = l.split('-'); return `${MESES_PT_BR[parseInt(m)-1].substring(0,3)} ${y}`; }); charts.sprayConditionsByMonth.data.datasets = ['Ideal', 'Atenção', 'Evitar'].map(cond => ({ label: cond, data: monthlyLabelsSpray.map(m => { const total = Object.values(monthlyConditions[m]).reduce((a,b)=>a+b,0) - monthlyConditions[m].NoData; return total > 0 ? (monthlyConditions[m][cond] / total) * 100 : 0; }), backgroundColor: SPRAY_COLORS[cond] })); charts.sprayConditionsByMonth.update();
            const hourlyDeltaT = Array(24).fill(0).map(()=>[]); const hourlyWind = Array(24).fill(0).map(()=>[]); data.forEach(d => { const hour = d.datetime.getUTCHours(); if(d.delta_t !== null && !isNaN(d.delta_t)) hourlyDeltaT[hour].push(d.delta_t); if(d.vento_medio_kph !== null && !isNaN(d.vento_medio_kph)) hourlyWind[hour].push(d.vento_medio_kph); }); const hourLabels = Array(24).fill(0).map((_,i)=>`${String(i).padStart(2,'0')}:00`); charts.ventoDeltaTHorario.data.labels = hourLabels; charts.ventoDeltaTHorario.data.datasets = [ { label: 'Delta T Médio (°C)', data: hourlyDeltaT.map(h=>h.length > 0 ? h.reduce((a,b)=>a+b,0)/h.length : NaN), borderColor: '#ff6384', backgroundColor: 'rgba(255, 99, 132, 0.2)', yAxisID: 'y_deltat', fill: true, tension: 0.4 }, { label: 'Vento Médio (km/h)', data: hourlyWind.map(h=>h.length > 0 ? h.reduce((a,b)=>a+b,0)/h.length : NaN), borderColor: '#36a2eb', backgroundColor: 'rgba(54, 162, 235, 0.2)', yAxisID: 'y_vento', fill: true, tension: 0.4 } ]; charts.ventoDeltaTHorario.update();
            const hourlyGFDI = Array(24).fill(0).map(()=>[]); data.forEach(d => { if(d.gfdi !== null && !isNaN(d.gfdi)) hourlyGFDI[d.datetime.getUTCHours()].push(d.gfdi); }); charts.gfdiHorario.data.labels = hourLabels; charts.gfdiHorario.data.datasets = [{ label: 'GFDI Médio', data: hourlyGFDI.map(h=>h.length > 0 ? h.reduce((a,b)=>a+b,0)/h.length : NaN), borderColor:'#ffc107', backgroundColor: 'rgba(255, 193, 7, 0.2)', fill: true, tension: 0.4 }]; charts.gfdiHorario.update(); }
            function renderCalendar(date) { const year = date.getUTCFullYear(); const month = date.getUTCMonth(); document.getElementById('month-year-header').innerText = `${MESES_PT_BR[month]} de ${year}`; const grid = document.getElementById('calendar-grid'); grid.innerHTML = ''; const firstDay = new Date(Date.UTC(year, month, 1)).getUTCDay(); const daysInMonth = new Date(Date.UTC(year, month + 1, 0)).getUTCDate(); const today = new Date(); const todayStr = today.toISOString().split('T')[0]; const selectedStation = document.getElementById('station-filter').value; for (let i = 0; i < firstDay; i++) { grid.innerHTML += '<div class="calendar-day empty"></div>'; } for (let i = 1; i <= daysInMonth; i++) { const dayStr = `${year}-${String(month + 1).padStart(2, '0')}-${String(i).padStart(2, '0')}`; const dayData = currentDailyAggregated.find(d => d.data_str === dayStr); const dayEl = document.createElement('div'); dayEl.className = 'calendar-day'; if (dayStr === todayStr) dayEl.classList.add('today'); if (dayStr === selectedCalendarDay) dayEl.classList.add('selected'); let content = `<div class="day-number">${i}</div>`; if (dayData && dayData.precipitacao_mm > 0) { if (selectedStation === 'todas') { content += '<div class="day-rainfall-details">'; for (const stationName in dayData.precip_by_station) { const rain = dayData.precip_by_station[stationName]; content += `<div class="station-rain"><span>${stationName.substring(0,8)}</span> ${fNum(rain)} mm</div>`; } content += '</div>'; } else { content += `<div class="day-rainfall">${fNum(dayData.precipitacao_mm)} mm</div>`; } } dayEl.innerHTML = content; dayEl.addEventListener('click', () => showDailyDetails(dayStr, currentFilteredData)); grid.appendChild(dayEl); } }
            function renderSprayingWindow(hourlyData) { const container = document.getElementById('spraying-window-container'); container.innerHTML = ''; const barDiv = document.createElement('div'); barDiv.className = 'spraying-window-bar'; const axisDiv = document.createElement('div'); axisDiv.className = 'spraying-window-axis'; const dataMap = new Map(hourlyData.map(d => [d.datetime.getUTCHours(), d])); let restrictions = { wind_low: 0, wind_high: 0, delta_low: 0, delta_high: 0 }; for (let h = 0; h < 24; h++) { const hourData = dataMap.get(h); const wind = hourData ? hourData.vento_medio_kph : null; const deltaT = hourData ? hourData.delta_t : null; const condition = getSprayingCondition(wind, deltaT); if(condition === 'Evitar' || condition === 'Atenção'){ if(wind < 2) restrictions.wind_low++; if(wind > 9) restrictions.wind_high++; if(deltaT < 2) restrictions.delta_low++; if(deltaT > 10) restrictions.delta_high++; } const hourDiv = document.createElement('div'); hourDiv.className = 'spray-hour spray-hour-tooltip'; hourDiv.style.backgroundColor = SPRAY_COLORS[condition]; const windText = (wind !== null && !isNaN(wind)) ? `${fNum(wind, 1)} km/h` : 'N/D'; const deltaTText = (deltaT !== null && !isNaN(deltaT)) ? `${fNum(deltaT, 1)}` : 'N/D'; hourDiv.innerHTML = `<div class="spray-hour-content"><div class="spray-hour-time">${h}h</div><div class="spray-hour-value">ΔT: ${deltaTText}</div><div class="spray-hour-value">🌬️ ${windText}</div></div><span class="tooltip-text"><b>Hora: ${String(h).padStart(2,'0')}:00</b><br>Vento: ${windText}<br>ΔT: ${fNum(deltaT, 1)} °C</span>`; barDiv.appendChild(hourDiv); const axisLabel = document.createElement('div'); axisLabel.className = 'axis-label'; if (h % 3 === 0) { axisLabel.innerText = `${String(h).padStart(2,'0')}h`; } axisDiv.appendChild(axisLabel); } container.appendChild(barDiv); container.appendChild(axisDiv); let summaryText = "Condições ideais na maior parte do dia."; const maxRestriction = Object.keys(restrictions).reduce((a, b) => restrictions[a] > restrictions[b] ? a : b); if (restrictions[maxRestriction] > 3) { if(maxRestriction === 'wind_high') summaryText = "Principal restrição do dia: Vento forte (>9 km/h)."; else if(maxRestriction === 'delta_high') summaryText = "Principal restrição do dia: Delta T elevado (>10°C), alto risco de evaporação."; else if(maxRestriction === 'delta_low') summaryText = "Principal restrição do dia: Delta T baixo (<2°C), risco de escorrimento."; else if(maxRestriction === 'wind_low') summaryText = "Atenção: Períodos de vento muito baixo (<2 km/h), risco de inversão térmica."; } document.getElementById('spraying-summary').innerText = summaryText; }
            function showDailyDetails(dateStr, data) { const hourlyDataForDay = data.filter(d => d.datetime.toISOString().split('T')[0] === dateStr); const detailsContainer = document.getElementById('daily-details-container'); if (hourlyDataForDay.length === 0) { detailsContainer.style.display = 'none'; selectedCalendarDay = null; renderCalendar(calendarDate); return; } selectedCalendarDay = dateStr; renderCalendar(calendarDate); const [y,m,d] = dateStr.split('-'); document.getElementById('selected-day-header').innerText = `Detalhes de ${d}/${m}/${y}`; const hours = Array(24).fill(0).map((_, i) => `${String(i).padStart(2,'0')}:00`); const hourlyRain = Array(24).fill(NaN), hourlyTemp = Array(24).fill(NaN), hourlyHum = Array(24).fill(NaN), hourlyWind = Array(24).fill(NaN), hourlyDeltaT = Array(24).fill(NaN); hourlyDataForDay.forEach(rec => { const hour = rec.datetime.getUTCHours(); hourlyRain[hour] = (hourlyRain[hour] || 0) + (rec.precipitacao_mm || 0); hourlyTemp[hour] = rec.temp_media_c; hourlyHum[hour] = rec.umidade_media_perc; hourlyWind[hour] = rec.vento_medio_kph; hourlyDeltaT[hour] = rec.delta_t; }); charts.chuvaHoraria.data.labels = hours; charts.chuvaHoraria.data.datasets = [{ label: 'Chuva (mm)', data: hourlyRain, backgroundColor: '#64ffda' }]; charts.chuvaHoraria.update(); charts.tempUmidadeDiario.data.labels = hours; charts.tempUmidadeDiario.data.datasets = [ { label: 'Temperatura (°C)', data: hourlyTemp, borderColor: '#ff9f40', yAxisID: 'y_temp', tension: 0.2 }, { label: 'Umidade (%)', data: hourlyHum, borderColor: '#4bc0c0', yAxisID: 'y_rh', tension: 0.2 } ]; charts.tempUmidadeDiario.update(); charts.ventoDeltaTDiario.data.labels = hours; charts.ventoDeltaTDiario.data.datasets = [ { label: 'Delta T (°C)', data: hourlyDeltaT, borderColor: '#ff6384', yAxisID: 'y_deltat', tension: 0.2 }, { label: 'Vento (km/h)', data: hourlyWind, borderColor: '#36a2eb', yAxisID: 'y_vento', tension: 0.2 } ]; charts.ventoDeltaTDiario.update(); const speedBrackets = [[0,3], [3,6], [6,9], [9,100]]; const roseData = {}; CARDINAL_DIRECTIONS.forEach(dir => roseData[dir] = Array(speedBrackets.length).fill(0)); let totalVentos = 0; hourlyDataForDay.forEach(d => { const cardinal = degreesToCardinal(d.vento_direcao_graus); const speed = d.vento_medio_kph; if(cardinal && speed >= 0) { totalVentos++; for(let i=0; i<speedBrackets.length; i++) { if(speed >= speedBrackets[i][0] && speed < speedBrackets[i][1]) { roseData[cardinal][i]++; break; } } } }); charts.ventoRosaDiario.data.labels = CARDINAL_DIRECTIONS; charts.ventoRosaDiario.data.datasets = speedBrackets.map((bracket, i) => ({ label: `[${bracket[0]},${bracket[1]}) km/h`, data: CARDINAL_DIRECTIONS.map(dir => (roseData[dir][i]/(totalVentos || 1))*100) })); charts.ventoRosaDiario.update(); renderSprayingWindow(hourlyDataForDay); detailsContainer.style.display = 'block'; }
            function iniciarMapa() { if (!geoData || !geoData.fields || geoData.fields.length === 0) { document.getElementById('map-container').innerHTML = '<p style="text-align:center; padding-top: 50px;">Nenhum dado geográfico de talhão encontrado.</p>'; return; } const center = geoData.fields.length > 0 ? geoData.fields[0].centroid : [-14, -59]; map = L.map('map-container').setView(center, 12); const satelliteLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', { attribution: 'Tiles &copy; Esri' }); const streetLayer = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', { attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors' }).addTo(map); L.control.layers({"Ruas": streetLayer, "Satélite": satelliteLayer}, {}).addTo(map); geoData.fields.forEach(field => { const polygon = L.polygon(field.geometry.coordinates[0], { color: "#64ffda", weight: 2, opacity: 0.8, fillOpacity: 0.3 }); fieldLayers[field.field_id] = polygon; polygon.addTo(map); }); const stationIcon = L.divIcon({ html: '📡', className: 'station-icon', iconSize: [24, 24], iconAnchor: [12, 12] }); geoData.stations.forEach(station => { const marker = L.marker([station.latitude, station.longitude], { icon: stationIcon }).addTo(map); stationMarkers[station.name] = marker; }); mapLegend = L.control({position: 'bottomright'}); mapLegend.onAdd = function (map) { const div = L.DomUtil.create('div', 'info legend'); div.style.backgroundColor = 'rgba(17, 34, 64, 0.9)'; div.style.padding = '10px'; div.style.borderRadius = '5px'; div.style.color = '#e6f1ff'; return div; }; mapLegend.addTo(map); }
            function atualizarMapa() { if (!map) return; const startDate = new Date(document.getElementById('start-date').value + "T00:00:00Z"); const endDate = new Date(document.getElementById('end-date').value + "T23:59:59Z"); const selectedMetric = document.getElementById('map-metric-selector').value; const config = mapMetricsConfig[selectedMetric]; const filteredData = allData.filter(d => d.datetime >= startDate && d.datetime <= endDate); if (filteredData.length === 0) { Object.values(fieldLayers).forEach(layer => layer.setStyle({ fillColor: 'grey', color: 'grey', fillOpacity: 0.1 })); updateMapLegend(0, 0, () => 'grey', config, startDate, endDate); return; } const stationData = {}; geoData.stations.forEach(s => { stationData[s.name] = { values: [], lat: s.latitude, lon: s.longitude }; }); filteredData.forEach(d => { if (stationData[d.nome_estacao] && typeof d[config.key] === 'number') { stationData[d.nome_estacao].values.push(d[config.key]); } }); const stationAggregates = []; for (const name in stationData) { const data = stationData[name]; let aggValue; if (data.values.length > 0) { if (config.agg === 'sum') aggValue = data.values.reduce((a, b) => a + b, 0); else if (config.agg === 'avg') aggValue = data.values.reduce((a, b) => a + b, 0) / data.values.length; else if (config.agg === 'max') aggValue = Math.max(...data.values); stationAggregates.push({ lat: data.lat, lon: data.lon, value: aggValue }); if (stationMarkers[name]) stationMarkers[name].bindPopup(`<b>Estação: ${name}</b><br>${config.label}: ${fNum(aggValue)} ${config.unit}`); } } if (stationAggregates.length === 0) { Object.values(fieldLayers).forEach(layer => layer.setStyle({ fillColor: 'grey', color: 'grey', fillOpacity: 0.1 })); updateMapLegend(0, 0, () => 'grey', config, startDate, endDate); return; } const fieldValues = []; geoData.fields.forEach(field => { const interpolatedValue = idwInterpolation(field.centroid[0], field.centroid[1], stationAggregates); if (!isNaN(interpolatedValue)) fieldValues.push(interpolatedValue); field.interpolatedValue = interpolatedValue; }); const minVal = fieldValues.length > 0 ? Math.min(...fieldValues) : 0; const maxVal = fieldValues.length > 0 ? Math.max(...fieldValues) : 0; const colorScale = createColorScale(minVal, maxVal, config.colors); geoData.fields.forEach(field => { const layer = fieldLayers[field.field_id]; if (layer) { const value = field.interpolatedValue; const color = !isNaN(value) ? colorScale(value) : 'grey'; layer.setStyle({ fillColor: color, color: color, weight: 1.5, fillOpacity: 0.6 }); layer.bindPopup(`<b>Talhão: ${field.field_name}</b><br>${config.label} (estimado): ${fNum(value)} ${config.unit}`); } }); updateMapLegend(minVal, maxVal, colorScale, config, startDate, endDate); }
            function haversineDistance(lat1, lon1, lat2, lon2) { const R = 6371; const toRad = val => val * Math.PI / 180; const dLat = toRad(lat2 - lat1); const dLon = toRad(lon2 - lon1); const a = Math.sin(dLat / 2) * Math.sin(dLat / 2) + Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLon / 2) * Math.sin(dLon / 2); const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a)); return R * c; }
            function idwInterpolation(targetLat, targetLon, stations, power = 2) { let n = 0, d = 0; for(const s of stations){ const dist = haversineDistance(targetLat, targetLon, s.lat, s.lon); if(dist < 0.001) return s.value; const w = 1.0 / Math.pow(dist, power); n += w * s.value; d += w; } return d > 0 ? n / d : NaN; }
            function createColorScale(min, max, colors) { return function(value) { if (value <= min) return colors[0]; if (value >= max) return colors[colors.length - 1]; const r = max - min; if (r < 1e-9) return colors[Math.floor(colors.length/2)]; const p = (value - min) / r; const i = Math.min(Math.floor(p * colors.length), colors.length - 1); return colors[i]; }; }
            function updateMapLegend(min, max, scale, config, start, end) { const div = mapLegend.getContainer(); const fDate = (d) => d.toLocaleDateString('pt-BR',{timeZone:'UTC'}); let html = `<h4>${config.label}</h4><p style="font-size:0.8em;margin:0 0 5px 0;">Período: ${fDate(start)} a ${fDate(end)}</p>`; let grades = []; const step = (max-min)/5; if(step<1e-9 || min===max){grades=[min]}else{for(let i=0;i<=5;i++){grades.push(min+i*step)}} if(grades.length===1){html+=`<i style="background:${scale(grades[0])};width:18px;height:18px;float:left;margin-right:8px;opacity:0.7;"></i> ${fNum(grades[0],1)} ${config.unit}<br>`}else{for(let i=0;i<grades.length-1;i++){const from=grades[i];const to=grades[i+1];html+=`<i style="background:${scale(from+step/2)};width:18px;height:18px;float:left;margin-right:8px;opacity:0.7;"></i> ${fNum(from,1)} &ndash; ${fNum(to,1)} ${config.unit}<br>`}} div.innerHTML = html; }
            
            function generateAndRenderHistoricalAlerts(hourlyData) {
                const dailyDataByStation = {};
                hourlyData.forEach(d => {
                    const dayStr = d.datetime.toISOString().split('T')[0];
                    const station = d.nome_estacao;
                    const key = `${dayStr}|${station}`;
                    if (!dailyDataByStation[key]) {
                        dailyDataByStation[key] = {
                            date: dayStr, station: station,
                            precip: 0, gust: [], temp_max: [], temp_min: [], hum_min: []
                        };
                    }
                    if (d.precipitacao_mm > 0) dailyDataByStation[key].precip += d.precipitacao_mm;
                    if (d.rajada_max_kph) dailyDataByStation[key].gust.push(d.rajada_max_kph);
                    if (d.temp_max_c) dailyDataByStation[key].temp_max.push(d.temp_max_c);
                    if (d.temp_min_c) dailyDataByStation[key].temp_min.push(d.temp_min_c);
                    if (d.umidade_min_perc) dailyDataByStation[key].hum_min.push(d.umidade_min_perc);
                });

                const alertsByMonth = {};
                for (const key in dailyDataByStation) {
                    const dayData = dailyDataByStation[key];
                    const month = dayData.date.substring(0, 7);
                    if (!alertsByMonth[month]) alertsByMonth[month] = [];

                    const maxGust = dayData.gust.length > 0 ? Math.max(...dayData.gust) : 0;
                    const maxTemp = dayData.temp_max.length > 0 ? Math.max(...dayData.temp_max) : -Infinity;
                    const minTemp = dayData.temp_min.length > 0 ? Math.min(...dayData.temp_min) : Infinity;
                    const minHum = dayData.hum_min.length > 0 ? Math.min(...dayData.hum_min) : Infinity;

                    if (dayData.precip > ALERT_THRESHOLDS.RAIN_LIMIT) {
                        alertsByMonth[month].push({type: 'rain', date: dayData.date, icon: '🌧️', title: 'Chuva Volumosa', description: `Acumulado de <strong>${fNum(dayData.precip)} mm</strong> no dia.`, station: dayData.station });
                    }
                    if (maxGust > ALERT_THRESHOLDS.GUST_LIMIT) {
                        alertsByMonth[month].push({type: 'gust', date: dayData.date, icon: '💨', title: 'Rajada de Vento Forte', description: `Rajada máxima de <strong>${fNum(maxGust,0)} km/h</strong> registrada.`, station: dayData.station});
                    }
                    if (maxTemp > ALERT_THRESHOLDS.TEMP_HIGH) {
                        alertsByMonth[month].push({type: 'temp_high', date: dayData.date, icon: '🌡️', title: 'Temperatura Alta', description: `Máxima de <strong>${fNum(maxTemp)} °C</strong> registrada.`, station: dayData.station});
                    }
                    if (minTemp < ALERT_THRESHOLDS.TEMP_LOW) {
                        alertsByMonth[month].push({type: 'temp_low', date: dayData.date, icon: '🧊', title: 'Temperatura Baixa', description: `Mínima de <strong>${fNum(minTemp)} °C</strong> registrada.`, station: dayData.station});
                    }
                    if (minHum < ALERT_THRESHOLDS.HUM_LOW) {
                        alertsByMonth[month].push({type: 'hum_low', date: dayData.date, icon: '💧', title: 'Umidade Baixa', description: `Umidade relativa mínima de <strong>${fNum(minHum,0)}%</strong>.`, station: dayData.station});
                    }
                }

                const container = document.getElementById('historical-alerts-container');
                container.innerHTML = '';
                const sortedMonths = Object.keys(alertsByMonth).sort().reverse();
                
                if (sortedMonths.length === 0) {
                    container.innerHTML = '<div class="alert-section"><h2>Alertas Históricos</h2><p style="text-align:center; padding: 20px;">Nenhum alerta climatológico significativo encontrado para o período selecionado.</p></div>';
                    return;
                }

                let finalHtml = '<div class="alert-section"><h2>Alertas Históricos</h2>';
                sortedMonths.forEach(month => {
                    const [year, monthNum] = month.split('-');
                    const monthName = `${MESES_PT_BR[parseInt(monthNum)-1]} de ${year}`;
                    finalHtml += `<h3>${monthName}</h3><div class="alert-grid">`;
                    
                    alertsByMonth[month].sort((a,b) => a.date.localeCompare(b.date)).forEach(alert => {
                        const [y,m,d] = alert.date.split('-');
                        finalHtml += `
                            <div class="alert-card alert-card-${alert.type}">
                                <div class="alert-icon">${alert.icon}</div>
                                <div class="alert-body">
                                    <h4>${alert.title}</h4>
                                    <p><strong>Data:</strong> ${d}/${m}/${y}</p>
                                    <p>${alert.description}</p>
                                    <p><strong>Estação:</strong> ${alert.station}</p>
                                </div>
                            </div>
                        `;
                    });
                    finalHtml += '</div>';
                });
                finalHtml += '</div>';
                container.innerHTML = finalHtml;
            }

            // --- FUNÇÃO DE ALERTAS FUTUROS MODIFICADA ---
            function generateAndRenderFutureAlerts() {
                const container = document.getElementById('future-alerts-container');
                container.innerHTML = '';
                let allAlerts = [];
                const deltaTAlerts = {}; // { "station|date": [hours] }

                for(const stationName in allForecastData.daily) {
                    const dailyForecastData = allForecastData.daily[stationName];
                    const hourlyForecastData = allForecastData.hourly[stationName];

                    if(dailyForecastData) {
                        dailyForecastData.forEach(day => {
                            if (day.qtd_precip > ALERT_THRESHOLDS.RAIN_LIMIT) allAlerts.push({ station: stationName, type: 'rain', date: day.data, icon: '🌧️', title: 'Previsão de Chuva Intensa', description: `Previsto acumulado de <strong>${fNum(day.qtd_precip)} mm</strong>.` });
                            if (day.vento_vel > ALERT_THRESHOLDS.GUST_LIMIT) allAlerts.push({ station: stationName, type: 'gust', date: day.data, icon: '💨', title: 'Previsão de Vento Forte', description: `Ventos de até <strong>${fNum(day.vento_vel, 0)} km/h</strong> previstos.` });
                            if (day.max_temp > ALERT_THRESHOLDS.TEMP_HIGH) allAlerts.push({ station: stationName, type: 'temp_high', date: day.data, icon: '🌡️', title: 'Previsão de Temperatura Alta', description: `Máxima prevista de <strong>${fNum(day.max_temp)} °C</strong>.` });
                            if (day.min_temp < ALERT_THRESHOLDS.TEMP_LOW) allAlerts.push({ station: stationName, type: 'temp_low', date: day.data, icon: '🧊', title: 'Previsão de Temperatura Baixa', description: `Mínima prevista de <strong>${fNum(day.min_temp)} °C</strong>.` });
                        });
                    }
                    if(hourlyForecastData) {
                        hourlyForecastData.forEach(hour => {
                            if(hour.delta_t > ALERT_THRESHOLDS.DELTA_T_HIGH) {
                                const date = new Date(hour.fcst_valid_local);
                                const dateStr = `${String(date.getDate()).padStart(2,'0')}/${String(date.getMonth()+1).padStart(2,'0')}`;
                                const hourStr = date.getHours();
                                const key = `${stationName}|${dateStr}`;
                                if (!deltaTAlerts[key]) deltaTAlerts[key] = [];
                                deltaTAlerts[key].push(hourStr);
                            }
                        });
                    }
                }
                
                // Formata e adiciona os alertas de Delta T
                for(const key in deltaTAlerts) {
                    const [stationName, dateStr] = key.split('|');
                    const hours = deltaTAlerts[key].sort((a,b) => a-b).map(h => h + 'h').join(', ');
                    allAlerts.push({ station: stationName, type: 'delta_t', date: dateStr, icon: '☀️', title: 'Delta T Elevado', description: `Condições de Delta T acima de <strong>${ALERT_THRESHOLDS.DELTA_T_HIGH}°C</strong> previstas para as horas: <strong>${hours}</strong>.` });
                }

                if (allAlerts.length === 0) return;
                
                let finalHtml = '<div class="alert-section"><h2>Alertas Previstos</h2><div class="alert-grid">';
                allAlerts.sort((a, b) => a.date.localeCompare(b.date) || a.station.localeCompare(b.station)).forEach(alert => {
                    finalHtml += `
                        <div class="alert-card alert-card-${alert.type}">
                            <div class="alert-icon">${alert.icon}</div>
                            <div class="alert-body">
                                <h4>${alert.title}</h4>
                                <p><strong>Data:</strong> ${alert.date}</p>
                                <p>${alert.description}</p>
                                <p><strong>Estação:</strong> ${alert.station}</p>
                            </div>
                        </div>
                    `;
                });
                finalHtml += '</div></div>';
                container.innerHTML = finalHtml;
            }
            Chart.getSpacedColors = function(count) { const colors = []; for (let i = 0; i < count; i++) { const hue = (360 / count) * i; colors.push(`hsl(${hue}, 70%, 60%)`); } return colors; };
            iniciarDashboard();
        });
    </script>
</body>
</html>
        """
        # --- FIM DO TEMPLATE HTML ---

        html_final = html_template.replace('__GROWER_NAME__', geodata.get('grower_name', 'Cliente'))
        html_final = html_final.replace('__JSON_DATA__', json_data)
        html_final = html_final.replace('__GEODATA__', json_geodata)
        html_final = html_final.replace('__JSON_ALL_FORECASTS__', json_all_forecasts)
        html_final = html_final.replace('__JSON_PREDICTION_DATA__', json_prediction_data)
        
        # Define a pasta de saída
        output_dir = "dist"
        # Cria a pasta 'dist' se ela não existir
        os.makedirs(output_dir, exist_ok=True)
        
        # Salva como index.html para ter um link mais limpo
        filename = os.path.join(output_dir, "index.html")
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_final)
        
        print(f"\nRelatório '{filename}' gerado com sucesso!")
        # A linha 'webbrowser.open' foi removida.
    # ============================================================================
    # FIM DO MÉTODO MODIFICADO
    # ============================================================================

    # ============================================================================
    # NOVO MÉTODO: gerar_relatorio_unico
    # Orquestra tudo, sem inputs, para o cliente definido no __init__
    # ============================================================================
    def gerar_relatorio_unico(self):
        """Orquestra todo o processo para o cliente único definido na inicialização."""
        
        grower_name = self.grower_name_cache[self.target_grower_id]
        print(f"\n--- Iniciando Relatório para Cliente: {grower_name} (ID: {self.target_grower_id}) ---")
        
        field_borders = self.get_field_borders_for_grower(self.target_grower_id)
        
        # --- Lógica de Previsão para todas as estações do cliente ---
        all_forecasts = {'daily': {}, 'hourly': {}}
        if not self.stations_info:
             print("AVISO: Nenhuma estação encontrada para buscar a previsão do tempo.")
        else:
            for station in self.stations_info:
                station_name = station.get('name', f"ID {station['id_estacao']}")
                lat, lon = station.get('latitude'), station.get('longitude')
                if lat is not None and lon is not None:
                    print(f"\n--- Buscando previsão do tempo para: '{station_name}' ---")
                    all_forecasts['daily'][station_name] = self.buscar_previsao_clima(lat, lon)
                    all_forecasts['hourly'][station_name] = self.buscar_previsao_horaria(lat, lon)
                else:
                    print(f"AVISO: Estação '{station_name}' não possui coordenadas válidas.")
        
        # --- Define o período (baseado na constante global) ---
        end_date_dt = datetime.now() - timedelta(days=1)
        start_date_dt = end_date_dt - timedelta(days=365 * ANOS_DE_HISTORICO)
        start_date = start_date_dt.strftime('%Y-%m-%d')
        end_date = end_date_dt.strftime('%Y-%m-%d')
        
        print(f"\nPeríodo de dados históricos: {start_date} a {end_date} ({ANOS_DE_HISTORICO} anos)")

        all_dfs = []
        for station in self.stations_info:
            station_id = station['id_estacao']
            station_name = station.get('name', f"ID {station_id}")
            raw_data = self.buscar_dados_climaticos(station_id, start_date, end_date)
            if raw_data:
                df_station = self.processar_para_dataframe(raw_data, station_id, station_name)
                all_dfs.append(df_station)

        if not all_dfs:
            print("\nAVISO: Nenhum dado climático foi encontrado para as estações deste cliente.")
            df_completo = pd.DataFrame()
            dados_predicao = {}
        else:
            df_completo = pd.concat(all_dfs, ignore_index=True)
            print(f"\nTotal de {len(df_completo)} registros horários processados.")
            dados_predicao = self.gerar_predicao_clima(df_completo)
        
        geodata = {
            'grower_name': grower_name, 
            'fields': field_borders, 
            'stations': self.stations_info
        }
        
        # Chama o método modificado para salvar o arquivo
        self.gerar_html_final(df_completo, geodata, all_forecasts, dados_predicao)
    # ============================================================================
    # FIM DO NOVO MÉTODO
    # ============================================================================


# ============================================================================
# --- BLOCO DE EXECUÇÃO PRINCIPAL ---
# ============================================================================
if __name__ == "__main__":
    print("Iniciando o script de geração de relatório...")
    
    try:
        # 1. Instancia a classe com os dados de configuração
        analisador = RelatorioClimaCompleto(
            grower_id=CLIENTE_ID,
            grower_name=CLIENTE_NOME,
            stations=ESTACOES_DO_CLIENTE
        )
        
        # 2. Roda a geração do relatório
        analisador.gerar_relatorio_unico()
        
        print("\n--- Geração de Relatório Concluída com Sucesso ---")
        
    except Exception as e:
        print(f"\n--- ERRO FATAL DURANTE A EXECUÇÃO: {e} ---")
        import traceback
        traceback.print_exc()
        # Faz o script falhar para que o GitHub Actions reporte o erro

        exit(1)





