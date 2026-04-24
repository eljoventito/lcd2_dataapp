"""
Dashboard Analítico — Pokémon Generación I
Lenguaje de Ciencia de Datos II (4364) | CIBERTEC · Ciclo 4 · 2026

Tema 6: Despliegue y Evaluación de Dashboards
Demuestra: diseño UX, visualización estratégica, caché, buenas prácticas.
"""

import time
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path
from typing import Optional

# ── Constantes ───────────────────────────────────────────────────
BASE_URL: str = "https://pokeapi.co/api/v2"
TOTAL_POKEMON: int = 150
CACHE_PATH: Path = Path("data/pokemon_gen1.parquet")
REQUEST_TIMEOUT: int = 10
SLEEP_BETWEEN_REQUESTS: float = 0.05

STATS_COLS: list[str] = [
    "hp", "ataque", "defensa",
    "ataque_especial", "defensa_especial", "velocidad",
]

TIPO_COLORES: dict[str, str] = {
    "fire": "#FF6B35", "water": "#4FC3F7", "grass": "#66BB6A",
    "electric": "#FFEE58", "psychic": "#EC407A", "normal": "#BDBDBD",
    "poison": "#AB47BC", "rock": "#8D6E63", "ground": "#FFA726",
    "flying": "#29B6F6", "bug": "#8BC34A", "ghost": "#5C6BC0",
    "ice": "#80DEEA", "dragon": "#7E57C2", "fighting": "#EF5350",
    "steel": "#B0BEC5", "fairy": "#F48FB1", "dark": "#546E7A",
}

TIPO_ES: dict[str, str] = {
    "fire": "Fuego", "water": "Agua", "grass": "Planta",
    "electric": "Eléctrico", "psychic": "Psíquico", "normal": "Normal",
    "poison": "Veneno", "rock": "Roca", "ground": "Tierra",
    "flying": "Volador", "bug": "Bicho", "ghost": "Fantasma",
    "ice": "Hielo", "dragon": "Dragón", "fighting": "Lucha",
    "steel": "Acero", "fairy": "Hada", "dark": "Siniestro",
}


# ── Carga y transformación (cacheadas) ───────────────────────────

def _obtener_pokemon(pokemon_id: int) -> Optional[dict]:
    """Obtiene los datos de un Pokémon desde la PokeAPI.

    Parameters
    ----------
    pokemon_id : int
        ID del Pokémon (1-150).

    Returns
    -------
    Optional[dict]
        Diccionario con atributos del Pokémon o None si la petición falla.
    """
    try:
        url = f"{BASE_URL}/pokemon/{pokemon_id}"
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()

        stats = {s["stat"]["name"]: s["base_stat"] for s in data["stats"]}
        tipos = [t["type"]["name"] for t in data["types"]]

        return {
            "id": data["id"],
            "nombre": data["name"].capitalize(),
            "tipo_primario": tipos[0] if tipos else None,
            "tipo_secundario": tipos[1] if len(tipos) > 1 else "—",
            "hp": stats.get("hp", 0),
            "ataque": stats.get("attack", 0),
            "defensa": stats.get("defense", 0),
            "ataque_especial": stats.get("special-attack", 0),
            "defensa_especial": stats.get("special-defense", 0),
            "velocidad": stats.get("speed", 0),
            "altura_dm": data["height"],
            "peso_hg": data["weight"],
            "experiencia_base": data.get("base_experience") or 0,
        }
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Pokémon {pokemon_id}: {e}")
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def cargar_dataset() -> pd.DataFrame:
    """Carga el dataset desde caché Parquet o lo descarga de la PokeAPI.

    Returns
    -------
    pd.DataFrame
        DataFrame con los 150 Pokémon de Generación I.
    """
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

    if CACHE_PATH.exists():
        return pd.read_parquet(CACHE_PATH)

    registros = []
    progress = st.progress(0, text="Descargando datos desde PokeAPI...")
    for i in range(1, TOTAL_POKEMON + 1):
        datos = _obtener_pokemon(i)
        if datos:
            registros.append(datos)
        time.sleep(SLEEP_BETWEEN_REQUESTS)
        progress.progress(i / TOTAL_POKEMON, text=f"Descargando Pokémon {i}/{TOTAL_POKEMON}...")

    progress.empty()
    df = pd.DataFrame(registros)
    df.to_parquet(CACHE_PATH, index=False)
    return df


@st.cache_data(show_spinner=False)
def preparar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica transformaciones y feature engineering al dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset crudo de Pokémon.

    Returns
    -------
    pd.DataFrame
        Dataset enriquecido con columnas derivadas.
    """
    df = df.copy()
    df["stat_total"] = df[STATS_COLS].sum(axis=1)
    df["altura_m"] = df["altura_dm"] / 10
    df["peso_kg"] = df["peso_hg"] / 10
    df["tipo_primario_es"] = df["tipo_primario"].map(TIPO_ES).fillna(df["tipo_primario"])
    df["tier"] = pd.cut(
        df["stat_total"],
        bins=[0, 300, 400, 500, 700],
        labels=["Débil", "Normal", "Fuerte", "Legendario"],
    )
    return df


# ── Configuración de página ───────────────────────────────────────

def configurar_pagina() -> None:
    """Configura título, ícono y layout de la página."""
    st.set_page_config(
        page_title="Pokémon Dashboard · LCD II",
        page_icon="🎮",
        layout="wide",
        initial_sidebar_state="expanded",
    )


# ── Sidebar con filtros ───────────────────────────────────────────

def sidebar_filtros(df: pd.DataFrame) -> pd.DataFrame:
    """Renderiza el sidebar con filtros y retorna el DataFrame filtrado.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset completo de Pokémon.

    Returns
    -------
    pd.DataFrame
        Subconjunto del dataset según los filtros aplicados.
    """
    with st.sidebar:
        st.image(
            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/items/poke-ball.png",
            width=60,
        )
        st.title("🎛️ Filtros")
        st.caption("Lenguaje de Ciencia de Datos II · Tema 6")
        st.divider()

        # Filtro por tipo
        tipos_disponibles = sorted(df["tipo_primario"].dropna().unique())
        tipos_es = [TIPO_ES.get(t, t).capitalize() for t in tipos_disponibles]
        tipo_map = dict(zip(tipos_es, tipos_disponibles))

        seleccion_es = st.multiselect(
            "Tipo primario",
            options=tipos_es,
            default=tipos_es,
            help="Selecciona uno o más tipos para filtrar",
        )
        tipos_sel = [tipo_map[t] for t in seleccion_es]

        # Filtro por rango de HP
        hp_min, hp_max = int(df["hp"].min()), int(df["hp"].max())
        rango_hp = st.slider(
            "Rango de HP",
            min_value=hp_min,
            max_value=hp_max,
            value=(hp_min, hp_max),
        )

        # Filtro por stat total
        st_min, st_max = int(df["stat_total"].min()), int(df["stat_total"].max())
        rango_total = st.slider(
            "Stat Total",
            min_value=st_min,
            max_value=st_max,
            value=(st_min, st_max),
        )

        st.divider()
        st.caption("🔗 Datos: [PokeAPI](https://pokeapi.co)")
        st.caption("📚 CIBERTEC · Ciclo 4 · 2026")

    # Aplicar filtros
    df_filtrado = df[
        df["tipo_primario"].isin(tipos_sel)
        & df["hp"].between(rango_hp[0], rango_hp[1])
        & df["stat_total"].between(rango_total[0], rango_total[1])
    ]

    return df_filtrado


# ── Tabs del dashboard ────────────────────────────────────────────

def tab_resumen(df: pd.DataFrame, df_completo: pd.DataFrame) -> None:
    """Renderiza la pestaña de Resumen General.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset filtrado.
    df_completo : pd.DataFrame
        Dataset completo (para calcular deltas de KPIs).
    """
    # ── KPIs ─────────────────────────────────────────────────────
    st.subheader("📌 Indicadores Clave")
    col1, col2, col3, col4 = st.columns(4)

    total = len(df)
    total_completo = len(df_completo)
    delta_total = total - total_completo if total != total_completo else None

    tipo_comun = df["tipo_primario"].value_counts().index[0] if total > 0 else "—"
    tipo_comun_es = TIPO_ES.get(tipo_comun, tipo_comun).capitalize()

    ataque_prom = round(df["ataque"].mean(), 1) if total > 0 else 0
    ataque_completo = round(df_completo["ataque"].mean(), 1)

    stat_total_prom = round(df["stat_total"].mean(), 1) if total > 0 else 0

    col1.metric(
        "Pokémon seleccionados",
        total,
        delta=delta_total,
        delta_color="normal",
    )
    col2.metric("Tipo más común", tipo_comun_es)
    col3.metric(
        "Ataque promedio",
        ataque_prom,
        delta=round(ataque_prom - ataque_completo, 1),
    )
    col4.metric("Stat Total promedio", stat_total_prom)

    if total == 0:
        st.warning("⚠️ Sin Pokémon con los filtros actuales. Ajusta los filtros del sidebar.")
        return

    st.divider()

    # ── Gráfico de distribución por tipo ─────────────────────────
    col_izq, col_der = st.columns([3, 2])

    with col_izq:
        st.subheader("Distribución por Tipo Primario")
        conteo = (
            df["tipo_primario"]
            .value_counts()
            .reset_index()
            .rename(columns={"tipo_primario": "tipo", "count": "cantidad"})
        )
        conteo["tipo_es"] = conteo["tipo"].map(TIPO_ES).fillna(conteo["tipo"])

        fig = px.bar(
            conteo,
            x="cantidad",
            y="tipo_es",
            orientation="h",
            color="tipo",
            color_discrete_map=TIPO_COLORES,
            text="cantidad",
            labels={"cantidad": "Pokémon", "tipo_es": "Tipo"},
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            showlegend=False,
            plot_bgcolor="white",
            yaxis={"categoryorder": "total ascending"},
            height=400,
            margin={"t": 10, "b": 10},
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_der:
        st.subheader("Distribución del Tier")
        tier_counts = df["tier"].value_counts().reset_index()
        tier_counts.columns = ["tier", "cantidad"]
        fig_pie = px.pie(
            tier_counts,
            names="tier",
            values="cantidad",
            color_discrete_sequence=["#EF5350", "#66BB6A", "#4FC3F7", "#7E57C2"],
            hole=0.45,
        )
        fig_pie.update_layout(height=400, margin={"t": 10, "b": 10})
        st.plotly_chart(fig_pie, use_container_width=True)


def tab_comparativo(df: pd.DataFrame) -> None:
    """Renderiza la pestaña de Análisis Comparativo.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset filtrado.
    """
    if len(df) == 0:
        st.warning("⚠️ Sin datos con los filtros actuales.")
        return

    col_izq, col_der = st.columns(2)

    with col_izq:
        st.subheader("⚔️ Ataque vs. Defensa")
        st.caption("Tamaño del punto = HP · Hover para ver el nombre")
        fig_scatter = px.scatter(
            df,
            x="ataque",
            y="defensa",
            color="tipo_primario",
            color_discrete_map=TIPO_COLORES,
            size="hp",
            hover_name="nombre",
            hover_data={
                "tipo_primario": True,
                "stat_total": True,
                "hp": True,
                "velocidad": True,
            },
            labels={
                "ataque": "Ataque Base",
                "defensa": "Defensa Base",
                "tipo_primario": "Tipo",
            },
        )
        fig_scatter.update_layout(
            plot_bgcolor="white",
            height=420,
            showlegend=False,
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_der:
        st.subheader("❤️ Distribución de HP por Tipo")
        st.caption("Cada punto es un Pokémon. La caja muestra mediana e IQR.")
        tipos_validos = df["tipo_primario"].value_counts()
        tipos_validos = tipos_validos[tipos_validos >= 2].index
        df_box = df[df["tipo_primario"].isin(tipos_validos)].copy()
        df_box["tipo_es"] = df_box["tipo_primario"].map(TIPO_ES).fillna(df_box["tipo_primario"])

        fig_box = px.box(
            df_box,
            x="tipo_es",
            y="hp",
            color="tipo_primario",
            color_discrete_map=TIPO_COLORES,
            hover_name="nombre",
            points="all",
            labels={"hp": "HP Base", "tipo_es": "Tipo"},
        )
        fig_box.update_layout(
            showlegend=False,
            plot_bgcolor="white",
            height=420,
            xaxis_tickangle=-35,
        )
        st.plotly_chart(fig_box, use_container_width=True)

    # Gráfico de barras agrupadas: perfil por tipo
    st.divider()
    st.subheader("📊 Perfil de Stats por Tipo")
    st.caption("Promedio de ataque, defensa, velocidad y HP por tipo")

    stats_tipo = (
        df.groupby("tipo_primario")[["ataque", "defensa", "velocidad", "hp"]]
        .mean()
        .round(1)
        .reset_index()
        .sort_values("ataque", ascending=False)
    )
    stats_tipo["tipo_es"] = stats_tipo["tipo_primario"].map(TIPO_ES).fillna(stats_tipo["tipo_primario"])

    fig_bar = px.bar(
        stats_tipo.melt(
            id_vars=["tipo_primario", "tipo_es"],
            value_vars=["ataque", "defensa", "velocidad", "hp"],
        ),
        x="tipo_es",
        y="value",
        color="variable",
        barmode="group",
        labels={"value": "Valor promedio", "tipo_es": "Tipo", "variable": "Stat"},
        color_discrete_map={
            "ataque": "#FF6B35",
            "defensa": "#4FC3F7",
            "velocidad": "#66BB6A",
            "hp": "#EC407A",
        },
    )
    fig_bar.update_layout(plot_bgcolor="white", height=380)
    st.plotly_chart(fig_bar, use_container_width=True)


def tab_explorador(df: pd.DataFrame) -> None:
    """Renderiza la pestaña de Explorador individual de Pokémon.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset filtrado.
    """
    if len(df) == 0:
        st.warning("⚠️ Sin datos con los filtros actuales.")
        return

    col_tabla, col_ficha = st.columns([3, 2])

    with col_tabla:
        st.subheader("📋 Tabla de Pokémon")

        busqueda = st.text_input("🔍 Buscar por nombre", placeholder="ej. Charizard")
        df_tabla = df.copy()
        if busqueda:
            df_tabla = df_tabla[df_tabla["nombre"].str.contains(busqueda, case=False)]

        cols_mostrar = ["id", "nombre", "tipo_primario_es", "hp", "ataque",
                        "defensa", "velocidad", "stat_total", "tier"]
        rename_map = {
            "id": "#", "nombre": "Nombre", "tipo_primario_es": "Tipo",
            "hp": "HP", "ataque": "Ataque", "defensa": "Defensa",
            "velocidad": "Velocidad", "stat_total": "Total", "tier": "Tier",
        }
        st.dataframe(
            df_tabla[cols_mostrar].rename(columns=rename_map),
            use_container_width=True,
            height=420,
            hide_index=True,
        )
        st.caption(f"{len(df_tabla)} Pokémon mostrados")

    with col_ficha:
        st.subheader("🃏 Ficha Individual")

        nombres = sorted(df["nombre"].tolist())
        seleccionado = st.selectbox("Selecciona un Pokémon", nombres)
        poke = df[df["nombre"] == seleccionado].iloc[0]

        # Sprite desde PokeAPI CDN
        sprite_url = (
            f"https://raw.githubusercontent.com/PokeAPI/sprites/master/"
            f"sprites/pokemon/{poke['id']}.png"
        )
        st.image(sprite_url, width=130)

        st.markdown(f"### #{poke['id']:03d} {poke['nombre']}")
        tipo_es = TIPO_ES.get(poke["tipo_primario"], poke["tipo_primario"]).capitalize()
        st.markdown(f"**Tipo:** `{tipo_es}`")
        st.markdown(f"**Altura:** {poke['altura_m']:.1f} m  |  **Peso:** {poke['peso_kg']:.1f} kg")
        st.markdown(f"**Exp. base:** {poke['experiencia_base']}  |  **Tier:** {poke['tier']}")

        # Radar de stats
        stats_vals = [poke[s] for s in STATS_COLS]
        stats_labels = ["HP", "Ataque", "Defensa", "Atk Esp.", "Def Esp.", "Velocidad"]
        color_tipo = TIPO_COLORES.get(poke["tipo_primario"], "#90A4AE")

        fig_radar = go.Figure(go.Scatterpolar(
            r=stats_vals + [stats_vals[0]],
            theta=stats_labels + [stats_labels[0]],
            fill="toself",
            fillcolor=color_tipo + "55",
            line={"color": color_tipo, "width": 2},
            name=poke["nombre"],
        ))
        fig_radar.update_layout(
            polar={"radialaxis": {"visible": True, "range": [0, 160]}},
            showlegend=False,
            height=300,
            margin={"t": 20, "b": 20, "l": 30, "r": 30},
        )
        st.plotly_chart(fig_radar, use_container_width=True)


def tab_correlaciones(df: pd.DataFrame) -> None:
    """Renderiza la pestaña de Correlaciones entre stats.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset filtrado.
    """
    if len(df) < 5:
        st.warning("⚠️ Se necesitan al menos 5 Pokémon para calcular correlaciones.")
        return

    col_heat, col_info = st.columns([3, 2])

    with col_heat:
        st.subheader("🔥 Mapa de Correlación entre Stats")
        st.caption("1.0 = correlación perfecta · 0 = sin relación · -1 = inversa")

        corr = df[STATS_COLS].corr().round(2)
        labels_es = ["HP", "Ataque", "Defensa", "Atk Esp.", "Def Esp.", "Velocidad"]

        fig_heat = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=labels_es,
            y=labels_es,
            colorscale="RdBu_r",
            zmid=0,
            zmin=-1,
            zmax=1,
            text=corr.values.round(2),
            texttemplate="%{text}",
            textfont={"size": 12},
        ))
        fig_heat.update_layout(height=420, margin={"t": 10})
        st.plotly_chart(fig_heat, use_container_width=True)

    with col_info:
        st.subheader("💡 Interpretación")
        st.caption("Pares de stats más correlacionados")

        import numpy as _np
        _mask = _np.triu(_np.ones(corr.shape, dtype=bool), k=1)
        corr_pairs = (
            corr.where(_mask)
            .stack()
            .reset_index()
        )
        corr_pairs.columns = ["stat_a", "stat_b", "correlacion"]
        corr_pairs["stat_a_es"] = corr_pairs["stat_a"].map(
            dict(zip(STATS_COLS, labels_es))
        )
        corr_pairs["stat_b_es"] = corr_pairs["stat_b"].map(
            dict(zip(STATS_COLS, labels_es))
        )
        corr_pairs = corr_pairs.sort_values("correlacion", ascending=False)

        st.markdown("**Mayor correlación positiva:**")
        for _, row in corr_pairs.head(3).iterrows():
            emoji = "🟢" if row["correlacion"] > 0.5 else "🟡"
            st.markdown(
                f"{emoji} **{row['stat_a_es']}** ↔ **{row['stat_b_es']}**: "
                f"`{row['correlacion']:.2f}`"
            )

        st.markdown("**Mayor correlación negativa:**")
        for _, row in corr_pairs.tail(3).iterrows():
            emoji = "🔴" if row["correlacion"] < -0.3 else "🟠"
            st.markdown(
                f"{emoji} **{row['stat_a_es']}** ↔ **{row['stat_b_es']}**: "
                f"`{row['correlacion']:.2f}`"
            )

        st.divider()
        st.subheader("🏆 Top 5 — Stat Total")
        top5 = df.nlargest(5, "stat_total")[["nombre", "tipo_primario_es", "stat_total"]]
        top5.columns = ["Pokémon", "Tipo", "Total"]
        st.dataframe(top5, hide_index=True, use_container_width=True)


# ── Main ──────────────────────────────────────────────────────────

def main() -> None:
    """Punto de entrada de la aplicación Streamlit."""
    configurar_pagina()

    # Header principal
    st.markdown("## 🎮 Dashboard Analítico — Pokémon Generación I")
    st.caption(
        "Lenguaje de Ciencia de Datos II (4364) · CIBERTEC · Ciclo 4 · 2026  |  "
        "Datos: [PokeAPI](https://pokeapi.co)"
    )
    st.divider()

    # Carga de datos con spinner
    with st.spinner("⏳ Cargando datos desde PokeAPI..."):
        df_raw = cargar_dataset()

    if df_raw.empty:
        st.error("[ERROR] No se pudieron cargar los datos. Verifica tu conexión a internet.")
        st.stop()

    df = preparar_datos(df_raw)
    st.success(f"✅ {len(df)} Pokémon cargados correctamente.")

    # Filtros del sidebar
    df_filtrado = sidebar_filtros(df)

    # Indicador de filtros activos
    if len(df_filtrado) < len(df):
        st.info(
            f"🔍 Filtros activos: mostrando **{len(df_filtrado)}** de **{len(df)}** Pokémon."
        )

    # Tabs principales
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Resumen General",
        "⚔️ Análisis Comparativo",
        "🔎 Explorador",
        "🔥 Correlaciones",
    ])

    with tab1:
        tab_resumen(df_filtrado, df)
    with tab2:
        tab_comparativo(df_filtrado)
    with tab3:
        tab_explorador(df_filtrado)
    with tab4:
        tab_correlaciones(df_filtrado)


if __name__ == "__main__":
    main()
