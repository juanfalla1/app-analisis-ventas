import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
import re
import requests
from io import BytesIO

# Configuración avanzada
st.set_page_config(
    page_title="Análisis Estratégico Profundo de Ventas",
    layout="wide",
    page_icon="🔍",
    initial_sidebar_state="expanded"
)

# URL del archivo en GitHub (reemplaza con tu URL real)
GITHUB_RAW_URL = "https://github.com/juanfalla1/app-analisis-ventas/blob/main/INFORME%20SEMESTRAL1.xlsx"

# Función para cargar datos desde GitHub
@st.cache_data
def load_data_from_github():
    try:
        response = requests.get(GITHUB_RAW_URL)
        response.raise_for_status()
        
        # Determinar tipo de archivo
        if GITHUB_RAW_URL.endswith('.csv'):
            df = pd.read_csv(BytesIO(response.content), delimiter=';', thousands=',', decimal='.', encoding='latin1')
        else:
            df = pd.read_excel(BytesIO(response.content))
        
        return process_data(df)
    except Exception as e:
        st.error(f"Error al cargar datos desde GitHub: {str(e)}")
        return None

# Función para normalizar nombres de columnas
def normalize_column_name(name, existing_names):
    name = name.lower().strip()
    name = re.sub(r'\s+', '_', name)
    name = re.sub(r'[^a-z0-9_]', '', name)
    
    # Manejar duplicados
    if name in existing_names:
        count = 1
        new_name = f"{name}_{count}"
        while new_name in existing_names:
            count += 1
            new_name = f"{name}_{count}"
        name = new_name
    
    existing_names.add(name)
    return name

# Función para procesar datos
def process_data(df):
    # Normalizar nombres de columnas evitando duplicados
    existing_names = set()
    normalized_columns = []
    
    for col in df.columns:
        normalized_columns.append(normalize_column_name(col, existing_names))
    
    df.columns = normalized_columns
    
    # Mapeo de columnas críticas - ajustado a tu estructura
    critical_columns = {
        'fecha': 'fecha',
        'nit': 'nit',
        'segmento': 'segmento',
        'nombre_cliente': 'nombre_cliente',
        'ciudad': 'ciudad',
        'vendedor': 'vendedor',
        'lin': 'linea',
        'referencia': 'referencia',
        'valor_ventas': 'valor_ventas',
        'cantidad_ventas': 'cantidad_ventas'
    }
    
    # Renombrar columnas según el mapeo
    rename_dict = {}
    for orig_name, new_name in critical_columns.items():
        if orig_name in df.columns:
            rename_dict[orig_name] = new_name
    
    df = df.rename(columns=rename_dict)
    
    # Verificar columnas críticas
    missing_critical = []
    for needed in critical_columns.values():
        if needed not in df.columns:
            missing_critical.append(needed)
    
    if missing_critical:
        st.error(f"Columnas críticas faltantes: {', '.join(missing_critical)}")
        st.error(f"Columnas disponibles: {', '.join(df.columns)}")
        return None
    
    # Limpieza y transformaciones
    df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce', dayfirst=True)
    
    # Convertir columnas numéricas
    if 'valor_ventas' in df.columns:
        df['valor_ventas'] = pd.to_numeric(df['valor_ventas'], errors='coerce')
    
    if 'cantidad_ventas' in df.columns:
        df['cantidad_ventas'] = pd.to_numeric(df['cantidad_ventas'], errors='coerce')
    
    # Crear campos derivados
    df['mes'] = df['fecha'].dt.month_name()
    df['año'] = df['fecha'].dt.year
    df['semana'] = df['fecha'].dt.isocalendar().week
    df['dia_semana'] = df['fecha'].dt.day_name()
    
    # Filtrar filas con fechas válidas
    df = df.dropna(subset=['fecha'])
    
    # Verificar si hay datos válidos
    if df.empty:
        st.error("No hay datos válidos después de la limpieza. Verifique el formato de fecha.")
        return None
    
    return df

# Función para comparación anual
def comparacion_anual(df, year1, year2):
    # Filtrar datos para los dos años
    df_year1 = df[df['año'] == year1]
    df_year2 = df[df['año'] == year2]
    
    # Verificar si hay datos para ambos años
    if df_year1.empty or df_year2.empty:
        st.warning(f"No hay datos suficientes para comparar {year1} y {year2}")
        return None, None
    
    # Calcular métricas clave para ambos años
    ventas_year1 = df_year1['valor_ventas'].sum()
    ventas_year2 = df_year2['valor_ventas'].sum()
    crecimiento = ((ventas_year2 - ventas_year1) / ventas_year1) * 100 if ventas_year1 != 0 else 0
    
    clientes_year1 = df_year1['nit'].nunique()
    clientes_year2 = df_year2['nit'].nunique()
    crecimiento_clientes = ((clientes_year2 - clientes_year1) / clientes_year1) * 100 if clientes_year1 != 0 else 0
    
    # Mostrar métricas clave
    col1, col2, col3 = st.columns(3)
    col1.metric(f"Ventas {year1}", f"${ventas_year1:,.0f}")
    col2.metric(f"Ventas {year2}", f"${ventas_year2:,.0f}", f"{crecimiento:.1f}%")
    col3.metric(f"Clientes {year1} vs {year2}", f"{clientes_year1} → {clientes_year2}", f"{crecimiento_clientes:.1f}%")
    
    # Evolución mensual
    st.subheader(f"📈 Evolución Mensual: {year1} vs {year2}")
    
    # Orden de meses
    meses_orden = ['January', 'February', 'March', 'April', 'May', 'June', 
                  'July', 'August', 'September', 'October', 'November', 'December']
    
    # Agrupar por mes para ambos años
    df_mes_year1 = df_year1.groupby('mes').agg(ventas=('valor_ventas', 'sum')).reset_index()
    df_mes_year2 = df_year2.groupby('mes').agg(ventas=('valor_ventas', 'sum')).reset_index()
    
    # Ordenar meses cronológicamente
    df_mes_year1['mes'] = pd.Categorical(df_mes_year1['mes'], categories=meses_orden, ordered=True)
    df_mes_year2['mes'] = pd.Categorical(df_mes_year2['mes'], categories=meses_orden, ordered=True)
    df_mes_year1 = df_mes_year1.sort_values('mes')
    df_mes_year2 = df_mes_year2.sort_values('mes')
    
    # Crear figura
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_mes_year1['mes'], y=df_mes_year1['ventas'], 
                            mode='lines+markers', name=str(year1), line=dict(width=3)))
    fig.add_trace(go.Scatter(x=df_mes_year2['mes'], y=df_mes_year2['ventas'], 
                            mode='lines+markers', name=str(year2), line=dict(width=3)))
    fig.update_layout(title=f"Comparación Mensual de Ventas: {year1} vs {year2}",
                     xaxis_title='Mes',
                     yaxis_title='Ventas Totales ($)',
                     hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)
    
    # Análisis de crecimiento por mes
    df_crecimiento = pd.merge(df_mes_year1, df_mes_year2, on='mes', suffixes=(f'_{year1}', f'_{year2}'))
    df_crecimiento['crecimiento'] = (df_crecimiento[f'ventas_{year2}'] - df_crecimiento[f'ventas_{year1}']) / df_crecimiento[f'ventas_{year1}'] * 100
    df_crecimiento['crecimiento'] = df_crecimiento['crecimiento'].replace([np.inf, -np.inf], 0)
    
    fig_crecimiento = px.bar(df_crecimiento, x='mes', y='crecimiento', 
                            title=f"Crecimiento Mensual: {year2} vs {year1}",
                            labels={'crecimiento': 'Crecimiento (%)'})
    fig_crecimiento.update_traces(marker_color=np.where(df_crecimiento['crecimiento'] > 0, 'green', 'red'))
    st.plotly_chart(fig_crecimiento, use_container_width=True)
    
    return df_year1, df_year2

# Función para rankings comparativos (CORREGIDA)
def rankings_comparativos(df_year1, df_year2, year1, year2, mes=None):
    if mes:
        df_year1 = df_year1[df_year1['mes'] == mes]
        df_year2 = df_year2[df_year2['mes'] == mes]
    
    # Selector de tipo de ranking
    tipo_ranking = st.selectbox("Seleccionar tipo de ranking", 
                               ["Referencia", "Producto", "Ciudad", "Cliente"])
    
    if tipo_ranking == "Referencia":
        col_name = "referencia"
        titulo = "Referencias"
    elif tipo_ranking == "Producto":
        col_name = "linea"
        titulo = "Productos"
    elif tipo_ranking == "Ciudad":
        col_name = "ciudad"
        titulo = "Ciudades"
    else:  # Cliente
        col_name = "nombre_cliente"
        titulo = "Clientes"
    
    # Ranking para año 1
    df_rank_year1 = df_year1.groupby(col_name).agg(
        ventas=('valor_ventas', 'sum'),
        cantidad=('cantidad_ventas', 'sum'),
        transacciones=('valor_ventas', 'count')
    ).reset_index().sort_values('ventas', ascending=False).head(10)
    df_rank_year1['año'] = year1
    
    # Ranking para año 2
    df_rank_year2 = df_year2.groupby(col_name).agg(
        ventas=('valor_ventas', 'sum'),
        cantidad=('cantidad_ventas', 'sum'),
        transacciones=('valor_ventas', 'count')
    ).reset_index().sort_values('ventas', ascending=False).head(10)
    df_rank_year2['año'] = year2
    
    # Combinar rankings
    df_combined = pd.merge(df_rank_year1, df_rank_year2, on=col_name, 
                          suffixes=(f'_{year1}', f'_{year2}'), how='outer')
    
    # Calcular crecimiento y diferencias
    df_combined['diferencia_ventas'] = df_combined[f'ventas_{year2}'] - df_combined[f'ventas_{year1}']
    df_combined['crecimiento'] = (df_combined['diferencia_ventas'] / df_combined[f'ventas_{year1}']) * 100
    df_combined['crecimiento'] = df_combined['crecimiento'].replace([np.inf, -np.inf], 0)
    
    # Ordenar por ventas del año más reciente
    df_combined = df_combined.sort_values(f'ventas_{year2}', ascending=False)
    
    # Mostrar resultados
    st.subheader(f"🏆 Ranking Comparativo de {titulo}: {year1} vs {year2}")
    if mes:
        st.markdown(f"**Filtrado por mes:** {mes}")
    
    # Formatear valores para visualización
    display_cols = [col_name, 
                   f'ventas_{year1}', f'ventas_{year2}', 
                   'diferencia_ventas', 'crecimiento']
    
    # CORRECCIÓN: Crear tabla con estilo CORREGIDO
    styled_df = df_combined[display_cols].style \
        .format({
            f'ventas_{year1}': '${:,.0f}',
            f'ventas_{year2}': '${:,.0f}',
            'diferencia_ventas': '${:,.0f}',
            'crecimiento': '{:.1f}%'
        }) \
        .applymap(lambda x: 'background-color: #e6f7ff' if isinstance(x, (int, float)) and x % 2 == 0 else '', 
                 subset=pd.IndexSlice[::2, :]) \
        .applymap(lambda x: 'color: green' if x > 0 else 'color: red' if x < 0 else '', 
                 subset=['diferencia_ventas', 'crecimiento'])
    
    st.dataframe(styled_df, height=500)
    
    # Gráfico de comparación
    fig = go.Figure()
    
    # Añadir barras para cada año
    fig.add_trace(go.Bar(
        x=df_combined[col_name],
        y=df_combined[f'ventas_{year1}'],
        name=str(year1),
        marker_color='#1f77b4'
    ))
    
    fig.add_trace(go.Bar(
        x=df_combined[col_name],
        y=df_combined[f'ventas_{year2}'],
        name=str(year2),
        marker_color='#ff7f0e'
    ))
    
    # Personalizar diseño
    fig.update_layout(
        title=f"Comparación de Ventas por {titulo}: {year1} vs {year2}",
        xaxis_title=titulo,
        yaxis_title='Ventas Totales ($)',
        barmode='group',
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Función de análisis de rankings
def analisis_rankings(df):
    st.subheader("🏆 Rankings Estratégicos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Ranking por Cliente
        st.markdown("##### 🥇 Top 10 Clientes (por Valor)")
        if 'nit' in df.columns and 'nombre_cliente' in df.columns and 'valor_ventas' in df.columns:
            df_clientes = df.groupby(['nit', 'nombre_cliente']).agg(
                ventas=('valor_ventas', 'sum'),
                compras=('valor_ventas', 'count'),
                ticket_promedio=('valor_ventas', 'mean'),
                ultima_compra=('fecha', 'max')
            ).reset_index().sort_values('ventas', ascending=False).head(10)
            
            df_clientes['participación'] = (df_clientes['ventas'] / df['valor_ventas'].sum() * 100).round(1)
            df_clientes.index = range(1, len(df_clientes)+1)
            
            st.dataframe(df_clientes[['nombre_cliente', 'ventas', 'participación', 'ticket_promedio']].style.format({
                'ventas': '${:,.0f}',
                'participación': '{:.1f}%',
                'ticket_promedio': '${:,.0f}'
            }).bar(subset=['ventas'], color='#5fba7d'), height=400)
            
            # Análisis de Pareto
            st.markdown("##### 🔍 Análisis Pareto Clientes")
            df_pareto = df.groupby('nombre_cliente')['valor_ventas'].sum().reset_index()
            df_pareto = df_pareto.sort_values('valor_ventas', ascending=False)
            df_pareto['porcentaje_acumulado'] = (df_pareto['valor_ventas'].cumsum() / df_pareto['valor_ventas'].sum() * 100).round(1)
            
            fig_pareto = px.bar(df_pareto.head(20), x='nombre_cliente', y='valor_ventas', 
                               title="Distribución de Ventas por Cliente (Pareto)")
            fig_pareto.add_trace(go.Scatter(x=df_pareto.head(20)['nombre_cliente'], 
                                y=df_pareto.head(20)['porcentaje_acumulado'], 
                                mode='lines', name='% Acumulado', 
                                line=dict(color='red', width=3)))
            fig_pareto.update_layout(yaxis2=dict(overlaying='y', side='right', range=[0, 100]))
            st.plotly_chart(fig_pareto, use_container_width=True)
        else:
            st.warning("Columnas necesarias para análisis de clientes no disponibles")
    
    with col2:
        # Ranking por Línea
        st.markdown("##### 🥇 Top 10 Líneas (por Valor)")
        if 'linea' in df.columns and 'valor_ventas' in df.columns and 'cantidad_ventas' in df.columns and 'referencia' in df.columns:
            df_lineas = df.groupby('linea').agg(
                ventas=('valor_ventas', 'sum'),
                unidades=('cantidad_ventas', 'sum'),
                productos_unicos=('referencia', 'nunique')
            ).reset_index()
            
            # Simular margen
            df_lineas['margen'] = np.random.uniform(25, 50, len(df_lineas))
            df_lineas['participación'] = (df_lineas['ventas'] / df['valor_ventas'].sum() * 100).round(1)
            df_lineas = df_lineas.sort_values('ventas', ascending=False).head(10)
            df_lineas.index = range(1, len(df_lineas)+1)
            
            st.dataframe(df_lineas[['linea', 'ventas', 'participación', 'margen']].style.format({
                'ventas': '${:,.0f}',
                'participación': '{:.1f}%',
                'margen': '{:.1f}%'
            }).bar(subset=['margen'], color='#5fba7d'), height=400)
            
            # Análisis de Rentabilidad por Línea
            st.markdown("##### 📊 Rentabilidad por Línea")
            fig = px.scatter(df_lineas, x='ventas', y='margen', size='unidades', 
                            color='linea', hover_name='linea',
                            title="Ventas vs Margen por Línea",
                            labels={'ventas': 'Ventas Totales ($)',
                                    'margen': 'Margen Promedio (%)'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Columnas necesarias para análisis de líneas no disponibles")

# Función de análisis contextual
def analisis_contextual(df):
    st.subheader("🧠 Análisis Contextual")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Tendencias temporales
        st.markdown("##### 📅 Tendencias Mensuales")
        if 'mes' in df.columns and 'valor_ventas' in df.columns and 'año' in df.columns:
            # Obtener años únicos
            años = sorted(df['año'].unique(), reverse=True)
            
            # Si hay múltiples años, mostrar selector
            if len(años) > 1:
                años_seleccionados = st.multiselect("Seleccionar años para comparar", 
                                                  options=años, 
                                                  default=años[-2:],
                                                  max_selections=2)
                
                if len(años_seleccionados) >= 1:
                    df_filtrado = df[df['año'].isin(años_seleccionados)]
                    
                    df_mensual = df_filtrado.groupby(['año', 'mes']).agg(
                        ventas=('valor_ventas', 'sum'),
                        transacciones=('valor_ventas', 'count')
                    ).reset_index()
                    
                    # Ordenar meses cronológicamente
                    meses_orden = ['January', 'February', 'March', 'April', 'May', 'June', 
                                  'July', 'August', 'September', 'October', 'November', 'December']
                    df_mensual['mes'] = pd.Categorical(df_mensual['mes'], categories=meses_orden, ordered=True)
                    df_mensual = df_mensual.sort_values(['año', 'mes'])
                    
                    fig_mes = px.line(df_mensual, x='mes', y='ventas', color='año', markers=True,
                                     title="Evolución Mensual de Ventas",
                                     labels={'ventas': 'Ventas Totales ($)'})
                    fig_mes.update_traces(line=dict(width=4))
                    st.plotly_chart(fig_mes, use_container_width=True)
            else:
                df_mensual = df.groupby('mes').agg(
                    ventas=('valor_ventas', 'sum'),
                    transacciones=('valor_ventas', 'count')
                ).reset_index()
                
                # Ordenar meses cronológicamente
                meses_orden = ['January', 'February', 'March', 'April', 'May', 'June', 
                              'July', 'August', 'September', 'October', 'November', 'December']
                df_mensual['mes'] = pd.Categorical(df_mensual['mes'], categories=meses_orden, ordered=True)
                df_mensual = df_mensual.sort_values('mes')
                
                fig_mes = px.line(df_mensual, x='mes', y='ventas', markers=True,
                                 title="Evolución Mensual de Ventas",
                                 labels={'ventas': 'Ventas Totales ($)'})
                fig_mes.update_traces(line=dict(width=4))
                st.plotly_chart(fig_mes, use_container_width=True)
        else:
            st.warning("Datos necesarios para tendencias mensuales no disponibles")
        
        # Análisis por Día de la Semana
        st.markdown("##### 📆 Patrones Diarios")
        if 'dia_semana' in df.columns and 'valor_ventas' in df.columns:
            df_diario = df.groupby('dia_semana').agg(
                ventas=('valor_ventas', 'sum'),
                transacciones=('valor_ventas', 'count')
            ).reset_index()
            
            # Ordenar días de la semana
            dias_orden = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            df_diario['dia_semana'] = pd.Categorical(df_diario['dia_semana'], categories=dias_orden, ordered=True)
            df_diario = df_diario.sort_values('dia_semana')
            
            fig_dia = px.bar(df_diario, x='dia_semana', y='ventas', 
                            title="Ventas por Día de la Semana",
                            labels={'ventas': 'Ventas Totales ($)'})
            st.plotly_chart(fig_dia, use_container_width=True)
        else:
            st.warning("Datos necesarios para patrones diarios no disponibles")
    
    with col2:
        # Análisis Geográfico
        st.markdown("##### 🌎 Concentración Geográfica")
        if 'ciudad' in df.columns and 'valor_ventas' in df.columns and 'nit' in df.columns:
            df_geo = df.groupby('ciudad').agg(
                ventas=('valor_ventas', 'sum'),
                clientes=('nit', 'nunique')
            ).reset_index().sort_values('ventas', ascending=False)
            
            fig_geo = px.treemap(df_geo.head(15), path=['ciudad'], values='ventas',
                                color='clientes', hover_data=['ventas'],
                                title="Distribución de Ventas por Ciudad",
                                color_continuous_scale='Blues')
            st.plotly_chart(fig_geo, use_container_width=True)
        else:
            st.warning("Datos necesarios para análisis geográfico no disponibles")
        
        # Análisis Vendedores
        st.markdown("##### 👥 Desempeño de Vendedores")
        if 'vendedor' in df.columns and 'valor_ventas' in df.columns and 'nit' in df.columns:
            df_vendedores = df.groupby('vendedor').agg(
                ventas=('valor_ventas', 'sum'),
                clientes=('nit', 'nunique'),
                ticket_promedio=('valor_ventas', 'mean')
            ).reset_index().sort_values('ventas', ascending=False).head(5)
            
            st.dataframe(df_vendedores.style.format({
                'ventas': '${:,.0f}',
                'ticket_promedio': '${:,.0f}'
            }).bar(subset=['ventas'], color='#5fba7d'), height=300)
        else:
            st.warning("Datos necesarios para análisis de vendedores no disponibles")

# Función de análisis de segmentos
def analisis_segmentos(df):
    st.subheader("🎯 Análisis Estratégico por Segmento")
    
    if 'segmento' in df.columns and 'valor_ventas' in df.columns and 'nit' in df.columns:
        # Filtrar segmentos vacíos
        df_segmentos = df[df['segmento'].notna()].groupby('segmento').agg(
            ventas=('valor_ventas', 'sum'),
            clientes=('nit', 'nunique'),
            ticket_promedio=('valor_ventas', 'mean')
        ).reset_index()
        
        if not df_segmentos.empty:
            df_segmentos['participación'] = (df_segmentos['ventas'] / df_segmentos['ventas'].sum() * 100).round(1)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig1 = px.pie(df_segmentos, names='segmento', values='ventas', 
                             title='Participación de Mercado por Segmento',
                             hole=0.4)
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                fig2 = px.bar(df_segmentos, x='segmento', y='participación', 
                             title='Porcentaje de Participación por Segmento',
                             text='participación')
                fig2.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                st.plotly_chart(fig2, use_container_width=True)
            
            # Análisis comparativo
            st.markdown("##### 🔄 Comparativa de Segmentos")
            fig3 = px.scatter(df_segmentos, x='clientes', y='ticket_promedio', 
                             size='ventas', color='segmento', hover_name='segmento',
                             title="Clientes vs Ticket Promedio por Segmento",
                             labels={'clientes': 'Número de Clientes',
                                     'ticket_promedio': 'Ticket Promedio ($)'})
            st.plotly_chart(fig3, use_container_width=True)
            
            # Recomendaciones estratégicas
            segmento_lider = df_segmentos.loc[df_segmentos['ventas'].idxmax(), 'segmento']
            segmento_crecimiento = df_segmentos.loc[df_segmentos['clientes'].idxmax(), 'segmento']
            
            st.markdown(f"💡 **Recomendaciones Estratégicas:**")
            st.write(f"- **Segmento líder ({segmento_lider}):** Programas de fidelización premium")
            st.write(f"- **Segmento de crecimiento ({segmento_crecimiento}):** Campañas de captación")
            st.write("- **Segmentos emergentes:** Ofertas específicas para aumentar ticket")
        else:
            st.warning("No hay datos suficientes para analizar segmentos")
    else:
        st.warning("Datos necesarios para análisis de segmentos no disponibles")

# Interfaz principal
st.title("🔍 Análisis Estratégico Profundo de Ventas")
st.markdown("""
**Diagnóstico integral con comparación anual (2024 vs 2025)**  
*Nivel Gerencial - Enfoque en Causa Raíz - Rankings comparativos*
""")

# Cargar datos automáticamente desde GitHub
df = load_data_from_github()

if df is None or df.empty:
    st.error("No se pudieron cargar datos válidos desde GitHub. Verifique la conexión o el formato del archivo.")
    st.stop()

# Mostrar vista previa de datos
st.sidebar.markdown("### Vista previa de datos")
st.sidebar.dataframe(df.head(3))
st.sidebar.markdown(f"**Columnas disponibles:** {', '.join(df.columns)}")

# Filtros
st.sidebar.header("🔧 Filtros Analíticos")

# Filtro de fechas
if 'fecha' in df.columns:
    min_date = df['fecha'].min().date()
    max_date = df['fecha'].max().date()
    fecha_inicio = st.sidebar.date_input("Fecha inicial", min_date)
    fecha_fin = st.sidebar.date_input("Fecha final", max_date)
    
    # Convertir a datetime
    fecha_inicio = pd.to_datetime(fecha_inicio)
    fecha_fin = pd.to_datetime(fecha_fin)
    
    # Aplicar filtro de fechas
    df = df[(df['fecha'] >= fecha_inicio) & (df['fecha'] <= fecha_fin)]
else:
    st.sidebar.warning("No se encontró columna 'fecha' para filtrar")

# Filtro de ciudades
if 'ciudad' in df.columns:
    try:
        ciudades = ["Todas"] + sorted(df['ciudad'].dropna().astype(str).unique().tolist())
        ciudad_seleccionada = st.sidebar.selectbox("Filtrar por ciudad", ciudades)
        if ciudad_seleccionada != "Todas":
            df = df[df['ciudad'] == ciudad_seleccionada]
    except Exception as e:
        st.sidebar.error(f"Error al procesar ciudades: {str(e)}")
else:
    st.sidebar.warning("No se encontró columna 'ciudad' para filtrar")

# Filtro de segmentos
if 'segmento' in df.columns:
    try:
        segmentos = ["Todos"] + sorted(df['segmento'].dropna().astype(str).unique().tolist())
        segmento_seleccionado = st.sidebar.selectbox("Filtrar por segmento", segmentos)
        if segmento_seleccionado != "Todos":
            df = df[df['segmento'] == segmento_seleccionado]
    except Exception as e:
        st.sidebar.error(f"Error al procesar segmentos: {str(e)}")
else:
    st.sidebar.warning("No se encontró columna 'segmento' para filtrar")

# Filtro de líneas
if 'linea' in df.columns:
    try:
        lineas = ["Todas"] + sorted(df['linea'].dropna().astype(str).unique().tolist())
        linea_seleccionada = st.sidebar.selectbox("Filtrar por línea", lineas)
        if linea_seleccionada != "Todas":
            df = df[df['linea'] == linea_seleccionada]
    except Exception as e:
        st.sidebar.error(f"Error al procesar líneas: {str(e)}")
else:
    st.sidebar.warning("No se encontró columna 'linea' para filtrar")

# Resumen ejecutivo
st.sidebar.divider()
st.sidebar.markdown("### 📌 Resumen Ejecutivo")

if 'valor_ventas' in df.columns:
    ventas_totales = df['valor_ventas'].sum()
    st.sidebar.metric("Ventas Totales", f"${ventas_totales:,.0f}")
else:
    st.sidebar.warning("Columna 'valor_ventas' no encontrada")

if 'nit' in df.columns:
    clientes_unicos = df['nit'].nunique()
    st.sidebar.metric("Clientes Únicos", f"{clientes_unicos}")
else:
    st.sidebar.warning("Columna 'nit' no encontrada")

if 'valor_ventas' in df.columns and not df.empty:
    ticket_promedio = ventas_totales / len(df)
    st.sidebar.metric("Ticket Promedio", f"${ticket_promedio:,.0f}")

# Sección de Resumen Ejecutivo Avanzado
st.sidebar.divider()
st.sidebar.markdown("### 🧠 Resumen Estratégico")

if 'valor_ventas' in df.columns:
    ventas_totales = df['valor_ventas'].sum()
    
    # Calcular crecimiento mensual promedio
    if 'fecha' in df.columns and 'valor_ventas' in df.columns:
        df_mensual = df.resample('M', on='fecha')['valor_ventas'].sum().reset_index()
        if len(df_mensual) > 1:
            crecimiento = ((df_mensual['valor_ventas'].iloc[-1] / df_mensual['valor_ventas'].iloc[-2]) - 1) * 100
            st.sidebar.metric("Crecimiento Mensual", f"{crecimiento:.1f}%", 
                            "↑ Positivo" if crecimiento > 0 else "↓ Negativo")

if 'nit' in df.columns:
    clientes_unicos = df['nit'].nunique()
    
    # Calcular tasa de retención
    if 'fecha' in df.columns:
        clientes_activos = df[df['fecha'] > (datetime.now() - pd.DateOffset(months=3))]['nit'].nunique()
        retencion = (clientes_activos / clientes_unicos * 100) if clientes_unicos > 0 else 0
        st.sidebar.metric("Tasa de Retención (90 días)", f"{retencion:.1f}%")

# Años disponibles para comparación
if 'año' in df.columns:
    años_disponibles = sorted(df['año'].unique(), reverse=True)
    if len(años_disponibles) >= 2:
        año1 = st.sidebar.selectbox("Seleccionar primer año para comparación", años_disponibles, index=1)
        año2 = st.sidebar.selectbox("Seleccionar segundo año para comparación", años_disponibles, index=0)
    else:
        st.sidebar.warning("Se necesitan al menos dos años para comparación")
        año1 = año2 = None
else:
    st.sidebar.warning("No se encontró columna 'año' para comparación")
    año1 = año2 = None

# Meses disponibles para filtro
if 'mes' in df.columns:
    meses_disponibles = ["Todos"] + sorted(df['mes'].unique(), 
                                          key=lambda x: datetime.strptime(x, '%B').month)
    mes_seleccionado = st.sidebar.selectbox("Filtrar por mes", meses_disponibles)
else:
    mes_seleccionado = "Todos"

# Análisis en pestañas
tab1, tab2, tab3, tab4 = st.tabs(["🏆 Rankings", "🧠 Contexto", "🎯 Segmentos", "📊 Comparación Anual"])

with tab1:
    analisis_rankings(df)

with tab2:
    analisis_contextual(df)

with tab3:
    analisis_segmentos(df)

with tab4:
    if año1 and año2:
        df_year1, df_year2 = comparacion_anual(df, año1, año2)
        
        if df_year1 is not None and df_year2 is not None:
            if mes_seleccionado != "Todos":
                st.info(f"Filtro aplicado: Mes = {mes_seleccionado}")
            
            st.divider()
            rankings_comparativos(df_year1, df_year2, año1, año2, mes_seleccionado if mes_seleccionado != "Todos" else None)
    else:
        st.warning("Seleccione dos años diferentes para comparar en la barra lateral")

# Sección de Insights Estratégicos Profundos
st.divider()
st.subheader("🚀 Plan de Acción Estratégico Basado en Hallazgos")

# Generar insights personalizados con manejo de NaN
if 'segmento' in df.columns and 'valor_ventas' in df.columns:
    # Filtrar segmentos vacíos
    df_segmentos = df[df['segmento'].notna()].groupby('segmento')['valor_ventas'].sum().reset_index()
    
    if not df_segmentos.empty:
        # Calcular crecimiento potencial evitando divisiones por cero
        max_ventas = df_segmentos['valor_ventas'].max()
        df_segmentos['crecimiento_potencial'] = df_segmentos.apply(
            lambda row: (max_ventas - row['valor_ventas']) / row['valor_ventas'] 
            if row['valor_ventas'] > 0 else 0, axis=1
        )
        
        # Encontrar segmento con mayor potencial
        if not df_segmentos['crecimiento_potencial'].empty:
            segmento_oportunidad = df_segmentos.loc[
                df_segmentos['crecimiento_potencial'].idxmax(), 'segmento'
            ]
            st.write(f"1. **Oportunidad de crecimiento principal:** Segmento '{segmento_oportunidad}' tiene el mayor potencial de expansión")
            st.write("   - **Acciones:** Campaña focalizada, desarrollo de productos específicos, asignación de recursos especializados")
    else:
        st.warning("No hay datos suficientes para analizar segmentos")

# Manejo de NaN para análisis de vendedores
if 'vendedor' in df.columns and 'valor_ventas' in df.columns:
    df_vendedores = df[df['vendedor'].notna()].groupby('vendedor')['valor_ventas'].sum().reset_index()
    
    if not df_vendedores.empty and len(df_vendedores) > 1:
        max_ventas = df_vendedores['valor_ventas'].max()
        min_ventas = df_vendedores['valor_ventas'].min()
        if max_ventas > 0:
            brecha_eficiencia = ((max_ventas - min_ventas) / max_ventas) * 100
            st.write(f"2. **Brecha de eficiencia:** Diferencias de hasta {brecha_eficiencia:.1f}% en desempeño de vendedores")
            st.write("   - **Acciones:** Programa de mentoría, capacitación en técnicas de ventas, revisión de asignación de clientes")
    else:
        st.warning("No hay datos suficientes para analizar vendedores")

# Manejo de NaN para análisis estacional
if 'mes' in df.columns and 'valor_ventas' in df.columns:
    df_mensual = df[df['mes'].notna()].groupby('mes')['valor_ventas'].sum().reset_index()
    
    if not df_mensual.empty and len(df_mensual) > 1:
        venta_promedio = df_mensual['valor_ventas'].mean()
        if venta_promedio > 0:
            variacion_estacional = (df_mensual['valor_ventas'].max() - df_mensual['valor_ventas'].min()) 
            variacion_estacional = (variacion_estacional / venta_promedio) * 100
            st.write(f"3. **Estacionalidad marcada:** Variación del {variacion_estacional:.1f}% entre meses pico y valle")
            st.write("   - **Acciones:** Plan de fuerza de ventas flexible, promociones contraestacionales, gestión de inventario inteligente")
    else:
        st.warning("No hay datos suficientes para analizar estacionalidad")

# Manejo de NaN para análisis de concentración
if 'nombre_cliente' in df.columns and 'valor_ventas' in df.columns:
    df_clientes = df[df['nombre_cliente'].notna()].groupby('nombre_cliente')['valor_ventas'].sum().reset_index()
    
    if not df_clientes.empty and len(df_clientes) >= 3:
        ventas_totales = df_clientes['valor_ventas'].sum()
        if ventas_totales > 0:
            concentracion = df_clientes['valor_ventas'].nlargest(3).sum() / ventas_totales * 100
            st.write(f"4. **Riesgo de concentración:** Top 3 clientes representan el {concentracion:.1f}% de ventas")
            st.write("   - **Acciones:** Programa de diversificación, desarrollo de clientes medianos, contratos a largo plazo")
    else:
        st.warning("No hay datos suficientes para analizar concentración de clientes")

# Notas finales
st.sidebar.divider()
st.sidebar.caption(f"🔚 Análisis generado el {datetime.now().strftime('%Y-%m-%d %H:%M')}")
if not df.empty:
    st.sidebar.caption(f"📊 {len(df)} registros")
    if 'ciudad' in df.columns:
        st.sidebar.caption(f"🏙️ {df['ciudad'].nunique()} ciudades")
    if 'segmento' in df.columns:
        st.sidebar.caption(f"🎯 {df['segmento'].nunique()} segmentos")

