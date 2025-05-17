import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shiny import App, ui, render
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
from matplotlib.ticker import ScalarFormatter
import matplotlib.dates as mdates
from matplotlib.dates import MonthLocator

# OPCIONAL: Ocultar warnings molestos
warnings.filterwarnings("ignore")

# Diccionario visual
nombres_visuales = {
    "AUTOS": "Autos",
    "MOTOS": "Motos",
    "AUTOBUS DE 2 EJES": "Autobús\n(2 ejes)",
    "AUTOBUS DE 3 EJES": "Autobús\n(3 ejes)",
    "AUTOBUS DE 4 EJES": "Autobús\n(4 ejes)",
    "CAMIONES DE 2 EJES": "Camión\n(2 ejes)",
    "CAMIONES DE 3 EJES": "Camión\n(3 ejes)",
    "CAMIONES DE 4 EJES": "Camión\n(4 ejes)",
    "CAMIONES DE 5 EJES": "Camión\n(5 ejes)",
    "CAMIONES DE 6 EJES": "Camión\n(6 ejes)",
    "CAMIONES DE 7 EJES": "Camión\n(7 ejes)",
    "CAMIONES DE 8 EJES": "Camión\n(8 ejes)",
    "CAMIONES DE 9 EJES": "Camión\n(9 ejes)",
    "TRICICLOS": "Triciclos"
}

# Lista limpia de vehículos válidos
vehiculos = list(nombres_visuales.keys())

# Cargar datos
def cargar_datos():
    df = pd.read_csv("Aforos-RedPropia.csv", encoding='latin-1')
    meses_dict = {
        "enero": 1, "febrero": 2, "marzo": 3, "abril": 4,
        "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
        "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12
    }
    df["MES"] = df["MES"].str.lower().map(meses_dict)
    df["AÑO"] = df["AÑO"].astype(str).str.strip()
    df["FECHA"] = pd.to_datetime(df["AÑO"] + "-" + df["MES"].astype(str) + "-01")

    columnas = df.columns.difference(["NOMBRE", "TIPO", "AÑO", "MES", "FECHA"])
    for col in columnas:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(r"[^0-9.]", "", regex=True), errors='coerce').fillna(0)

    return df

df = cargar_datos()
anios = sorted(df["AÑO"].unique())
meses = {str(i): f"{i:02d}" for i in range(1, 13)}

# Interfaz
app_ui = ui.page_fluid(
    ui.tags.style("""
        .indicador-card {
            text-align: center;
            font-size: 1.1em;
        }
        .titulo-icono {
            font-size: 1.3em;
            font-weight: bold;
        }
    """),

    ui.div(
    ui.h3("🚌 Dashboard CAPUFE - Movimientos Vehiculares (2021–2025)"),
    style="text-align: center;"
    ),
    ui.hr(),

    # Indicadores superiores tipo tarjeta
    ui.layout_columns(
        ui.card(ui.div([ui.h4("🚗 Total Autos"), ui.output_text("total_autos")], class_="indicador-card")),
        ui.card(ui.div([ui.h4("📅 Frecuencia"), ui.output_text("frecuencia")], class_="indicador-card")),
        ui.card(ui.div([ui.h4("📈 Pronóstico"), ui.output_text("valor_pronostico")], class_="indicador-card"))
    ),

    ui.hr(),

    # Filtros en la izquierda y gráficas en la derecha
    ui.layout_sidebar(
        ui.sidebar(
            ui.h4("🎛️ Filtros"),
            ui.h5("📈 Pronóstico"),
            ui.input_select("vehiculo", "Tipo de Vehículo", choices={k: v for k, v in nombres_visuales.items()}),
            ui.input_select("anio", "Año", choices=anios),
            ui.input_select("mes", "Mes", choices=meses),
            ui.h5("📊 Distribución"),
            ui.input_select("anio_visual", "Año:", choices=anios),
            ui.input_checkbox_group(
                "tipos_visual",
                "Tipos de Vehículo",
                choices={k: v for k, v in nombres_visuales.items()},
                selected=vehiculos[:5]
            ),
            width=300
        ),
        ui.row(
            # Primera fila visual: gráficas
            ui.column(6, 
                ui.card(
                    ui.h5("📊 Distribución Por Año"),
                    ui.div(ui.output_plot("grafico_distribucion"), style="height: auto;"),
                    style="height: auto; min-height: 0;"
                )
            ),
            ui.column(6, 
                ui.card(
                    ui.h5("🔮 Pronóstico SARIMAX"),
                    ui.div(ui.output_plot("grafico_pronostico"), style="height: auto;"),
                    style="height: auto; min-height: 0;"
                )
            ),
            # Segunda fila visual: tabla y estadísticas (pero en las mismas columnas 6 + 6)
            ui.column(6,
                ui.card(
                    ui.h4("📋 Tabla Del Año Seleccionado"),  # Esto se queda fuera del centrado
                    ui.div(
                        ui.output_table("tabla_datos"),
                        style="max-height: 300px; overflow-y: auto; text-align: center;"
                    ),
                style="height: auto; min-height: 0;"  # Opcional si usas row/column
                )
            ),
            ui.column(6,
                ui.card(
                    ui.h4("📈 Estadísticas"),
                    ui.output_text("tipo_mayor"),
                    ui.output_text("tipo_menor"),
                    style="height: auto; min-height: 0; margin-top: 16px;"
                )
            )
        )
    ),
)

# Servidor
def server(input, output, session):

    @output
    @render.text
    def total_autos():
        anio = input.anio_visual()
        total = df[df["AÑO"] == anio]["AUTOS"].sum()
        return f"{int(total):,}"

    @output
    @render.text
    def frecuencia():
        return "1 mes"

    @output
    @render.text
    def valor_pronostico():
        try:
            vehiculo = input.vehiculo()
            anio = input.anio()
            mes = input.mes()

            serie = df.groupby("FECHA")[vehiculo].sum()
            serie.index.freq = "MS"
            fecha_objetivo = pd.Timestamp(f"{anio}-{mes}-01")
            fecha_final = serie.index[-1]

            if fecha_objetivo in serie.index:
                valor = serie.loc[fecha_objetivo]
                return f"📈 Valor real: {int(valor):,}"
            else:
                steps = (fecha_objetivo.year - fecha_final.year) * 12 + (fecha_objetivo.month - fecha_final.month)
                steps = max(1, steps)
                model = SARIMAX(serie, order=(1,0,1), seasonal_order=(1,0,1,12)).fit(disp=False)
                pred = model.get_forecast(steps=steps).predicted_mean
                print("Predicción cruda (texto):", pred.tail())  # Para inspección
                pred = pred.clip(lower=0)
                return f"📈 Pronóstico: {int(pred.iloc[-1]):,}"

        except Exception as e:
            print("ERROR en valor_pronostico:", e)
            return "N/A"

    @output
    @render.plot
    def grafico_pronostico():
        try:
            vehiculo = input.vehiculo()
            anio = input.anio()
            mes = input.mes()

            serie = df.groupby("FECHA")[vehiculo].sum()
            serie.index.freq = "MS"

            fecha_objetivo = pd.Timestamp(f"{anio}-{mes}-01")
            fecha_final = serie.index[-1]

            model = SARIMAX(serie, order=(1,0,1), seasonal_order=(1,0,1,12)).fit(disp=False)

            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.plot(serie, label="Histórico", color="blue")

            if fecha_objetivo in serie.index:
                valor_real = serie.loc[fecha_objetivo]
                ax.axvline(fecha_objetivo, color="red", linestyle="--", linewidth=1.5)
                ax.text(fecha_objetivo, ax.get_ylim()[1]*0.95, f"{fecha_objetivo.strftime('%Y-%b')}\nReal: {int(valor_real):,}",
                        color="red", rotation=90, verticalalignment='top', horizontalalignment='right', fontsize=8)
            else:
                steps = (fecha_objetivo.year - fecha_final.year) * 12 + (fecha_objetivo.month - fecha_final.month)
                steps = max(1, steps)

                forecast = model.get_forecast(steps=steps)
                pred = forecast.predicted_mean
                print("Predicción cruda:", pred.tail())  # Para inspección
                pred = pred.clip(lower=0)

                ci = forecast.conf_int()
                ci[ci < 0] = 0

                ax.plot(pred, label="Pronóstico", color="green")
                ax.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1], color="blue", alpha=0.2)

                ax.axvline(fecha_objetivo, color="red", linestyle="--", linewidth=1.5)
                ax.text(fecha_objetivo, ax.get_ylim()[1]*0.95, f"{fecha_objetivo.strftime('%Y-%b')}",
                        color="red", rotation=90, verticalalignment='top', horizontalalignment='right', fontsize=8)

            ax.set_title(f"Pronóstico para {nombres_visuales.get(vehiculo, vehiculo)}")
            ax.legend()
            ax.grid(True)

            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y\n%b'))
            ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
            ax.ticklabel_format(useOffset=False, style='plain', axis='y')
            ax.tick_params(axis='x', labelsize=8)

            plt.tight_layout()
            return fig

        except Exception as e:
            print("ERROR en grafico_pronostico:", e)
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.text(0.5, 0.5, "Error al generar gráfico", ha="center", va="center")
            return fig

    @output
    @render.plot
    def grafico_distribucion():
        anio = input.anio_visual()
        tipos = list(input.tipos_visual())
        datos = df[df["AÑO"] == anio][tipos].sum()
        datos = datos.rename(index=nombres_visuales)

        colores = ["#4F81BD", "#C0504D", "#9BBB59", "#FCD116", "#8064A2",
                "#4BACC6", "#F79646", "#C00000", "#00B050", "#7030A0",
                "#FF6666", "#FFCC00", "#00B0F0"]

        fig, ax = plt.subplots(figsize=(6, 3.5))
        bars = ax.bar(datos.index, datos.values, color=colores[:len(datos)])

        ax.set_title(f"Distribución en {anio}")
        ax.set_ylabel("Cantidad")
        ax.grid(axis="y")
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
        ax.ticklabel_format(useOffset=False, style='plain', axis='y')

        # Etiquetas debajo de cada barra
        for bar, label in zip(bars, datos.values):
            ax.text(bar.get_x() + bar.get_width() / 2, -max(datos.values) * 0.05,
                    f"{int(label):,}", ha='center', va='top', fontsize=8)

        ax.set_ylim(bottom=-max(datos.values) * 0.1)
        plt.tight_layout()
        return fig

    @output
    @render.table
    def tabla_datos():
        anio = input.anio_visual()
        tipos = list(input.tipos_visual())
        df_filtrado = df[df["AÑO"] == anio][["AÑO", "MES"] + tipos].copy()
       
        meses_nombres = {
            1: "ENERO", 2: "FEBRERO", 3: "MARZO", 4: "ABRIL",
            5: "MAYO", 6: "JUNIO", 7: "JULIO", 8: "AGOSTO",
            9: "SEPTIEMBRE", 10: "OCTUBRE", 11: "NOVIEMBRE", 12: "DICIEMBRE"
        }

        df_filtrado["MES"] = df_filtrado["MES"].map(meses_nombres)
        df_filtrado.rename(columns=nombres_visuales, inplace=True)
        return df_filtrado

    @output
    @render.text
    def tipo_mayor():
        anio = input.anio_visual()
        suma = df[df["AÑO"] == anio][vehiculos].sum()
        mayor = suma.idxmax()
        return f"Mayor movimiento: {nombres_visuales[mayor]} ({int(suma[mayor]):,})"

    @output
    @render.text
    def tipo_menor():
        anio = input.anio_visual()
        suma = df[df["AÑO"] == anio][vehiculos].sum()
        menor = suma.idxmin()
        return f"Menor movimiento: {nombres_visuales[menor]} ({int(suma[menor]):,})"

# Lanzar app
app = App(app_ui, server)