from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import textwrap

# -----------------------------------------------------------------------
# Project imports
# -----------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.common.aqi import aqi_category, aqi_color  # noqa: E402
from ml.common.cities import CITIES  # noqa: E402


# -----------------------------------------------------------------------
# App identity
# -----------------------------------------------------------------------

APP_NAME = "Pearl AQI Predictor"
APP_TAGLINE = "Get air quality data where you live."

HERO_IMAGE_URL = (
    "https://images.unsplash.com/photo-1758101662980-31d7b4e5ed99"
    "?auto=format&fit=crop&w=1800&q=85"
)

NAV_ITEMS = [
    "Air Quality Trend",
    "Why This Forecast",
    "Model Diagnostics",
]

GITHUB_ICON_DATA = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='%23ffffff'%3E%3Cpath fill-rule='evenodd' clip-rule='evenodd' d='M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.53 1.032 1.53 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z'/%3E%3C/svg%3E"

LINKEDIN_ICON_DATA = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='%23ffffff'%3E%3Cpath d='M19 3a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h14m-.5 15.5v-5.3a3.26 3.26 0 0 0-3.26-3.26c-.85 0-1.84.52-2.28 1.3v-1.11h-2.79v8.37h2.79v-4.93c0-.77.62-1.4 1.39-1.4a1.4 1.4 0 0 1 1.4 1.4v4.93h2.75M6.88 8.56a1.68 1.68 0 0 0 1.68-1.68c0-.93-.75-1.69-1.68-1.69a1.69 1.69 0 0 0-1.69 1.69c0 .93.76 1.68 1.69 1.68m1.39 9.94v-8.37H5.5v8.37h2.77z'/%3E%3C/svg%3E"

INSTAGRAM_ICON_DATA = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='%23ffffff'%3E%3Cpath d='M12 2.163c3.204 0 3.584.012 4.85.07 3.252.148 4.771 1.691 4.919 4.919.058 1.265.069 1.645.069 4.849 0 3.205-.012 3.584-.069 4.849-.149 3.225-1.664 4.771-4.919 4.919-1.266.058-1.644.07-4.85.07-3.204 0-3.584-.012-4.849-.07-3.26-.149-4.771-1.699-4.919-4.92-.058-1.265-.07-1.644-.07-4.849 0-3.204.013-3.583.07-4.849.149-3.227 1.664-4.771 4.919-4.919 1.266-.057 1.645-.069 4.849-.069zM12 0C8.741 0 8.333.014 7.053.072 2.695.272.273 2.69.073 7.052.014 8.333 0 8.741 0 12s.014 3.668.072 4.948c.2 4.358 2.618 6.78 6.98 6.98C8.333 23.986 8.741 24 12 24s3.668-.014 4.948-.072c4.354-.2 6.782-2.618 6.979-6.98.059-1.28.073-1.689.073-4.948 0-3.259-.014-3.667-.072-4.947-.196-4.354-2.617-6.78-6.979-6.98C15.668.014 15.259 0 12 0zm0 5.838a6.162 6.162 0 100 12.324 6.162 6.162 0 000-12.324zM12 16a4 4 0 110-8 4 4 0 010 8zm6.406-11.845a1.44 1.44 0 100 2.881 1.44 1.44 0 000-2.881z'/%3E%3C/svg%3E"


# -----------------------------------------------------------------------
# Streamlit configuration
# -----------------------------------------------------------------------

st.set_page_config(
    page_title="Pearl AQI Predictor",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# -----------------------------------------------------------------------
# Design tokens
# -----------------------------------------------------------------------

WHITE = "#FFFFFF"
PAGE_BG = "#F3F3F3"
CONTENT_BG = "#FFFFFF"

NAV_BLUE = "#075B97"
NAV_BLUE_DARK = "#074574"

HERO_BLUE = "#0056A6"
HERO_BLUE_DARK = "#003E87"

TEXT = "#171717"
TEXT_SOFT = "#555555"
TEXT_LIGHT = "#FFFFFF"

BORDER = "#C9C9C9"

AIRNOW_RED = "#A71930"
CARD_RED = "#B21F2D"

GOOD_GREEN = "#55B947"
MODERATE_YELLOW = "#F5D547"
UNHEALTHY_ORANGE = "#F28C28"
UNHEALTHY_RED = "#E64B4B"
VERY_UNHEALTHY_PURPLE = "#9B51B6"
HAZARDOUS_MAROON = "#7A1F1F"

ACCENT = "#0A65A8"


# -----------------------------------------------------------------------
# AQI scale
# -----------------------------------------------------------------------

AQI_BREAKPOINTS = [
    (0, GOOD_GREEN),
    (50, MODERATE_YELLOW),
    (100, UNHEALTHY_ORANGE),
    (150, UNHEALTHY_RED),
    (200, VERY_UNHEALTHY_PURPLE),
    (300, HAZARDOUS_MAROON),
    (500, HAZARDOUS_MAROON),
]


# -----------------------------------------------------------------------
# CSS
# -----------------------------------------------------------------------

def _inject_css() -> None:
    st.html(f"""
        <style>

        /* ============================================================
           GLOBAL
           ============================================================ */

        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Roboto+Condensed:wght@400;500;600;700&display=swap');

        html,
        body,
        [class*="css"] {{
            font-family: Arial, Helvetica, sans-serif;
            color: {TEXT};
        }}

        html {{
            scroll-behavior: smooth;
        }}

        body {{
            margin: 0;
            padding: 0;
            background: {PAGE_BG};
        }}

        .stApp {{
            background: {PAGE_BG};
        }}

        .main, [data-testid="stMain"], section.main {{
            padding: 0 !important;
            margin: 0 !important;
        }}

        .block-container {{
            padding-top: 0 !important;
            padding-bottom: 2rem !important;
            padding-left: 0 !important;
            padding-right: 0 !important;
            max-width: 100% !important;
            width: 100% !important;
            margin: 0 !important;
        }}

        #MainMenu,
        footer,
        [data-testid="collapsedControl"] {{
            display: none !important;
        }}

        [data-testid="stHeader"] {{
            display: none !important;
        }}

        /* ============================================================
           TOP NAVIGATION
           ============================================================ */

        .top-nav {{
            width: 100%;
            min-height: 72px;
            background: {NAV_BLUE};

            display: flex;
            align-items: center;

            box-sizing: border-box;
            padding: 0 24px;

            color: white;
            margin: 0 !important;
            border: none !important;
        }}

        .nav-inner {{
            width: 1110px;
            max-width: calc(100% - 30px);
            margin: 0 auto;

            display: flex;
            align-items: center;
            justify-content: space-between;

            gap: 20px;
        }}

        /* Pearls logo */

        .pearls-logo {{
            position: relative;

            width: 185px;
            height: 62px;

            display: flex;
            align-items: center;
            justify-content: center;

            flex-shrink: 0;
        }}

        .pearls-logo-ring {{
            position: absolute;

            width: 78px;
            height: 48px;

            border: 2px solid white;
            border-right-color: transparent;
            border-radius: 50%;

            transform: rotate(-8deg);
        }}

        .pearls-logo-ring-2 {{
            position: absolute;

            width: 98px;
            height: 54px;

            border: 2px solid rgba(255,255,255,0.85);
            border-left-color: transparent;
            border-radius: 50%;

            transform: rotate(9deg);
        }}

        .pearls-logo-text {{
            position: relative;
            z-index: 2;

            font-family: Arial, Helvetica, sans-serif;
            font-size: 25px;
            font-weight: 400;
            letter-spacing: -1px;

            color: white;
        }}

        .pearls-logo-text span {{
            color: #D72A42;
            font-weight: 700;
        }}

        /* Navigation links */

        .nav-links {{
            flex: 1;

            display: flex;
            align-items: center;
            justify-content: center;

            gap: 27px;
        }}

        .nav-link {{
            color: white !important;
            text-decoration: none !important;

            font-size: 13px;
            font-weight: 600;

            white-space: nowrap;

            opacity: 0.98;
        }}

        .nav-link:hover {{
            color: #E5F3FF !important;
            text-decoration: underline !important;
        }}

        .nav-right {{
            display: flex;
            align-items: center;
            gap: 16px;

            flex-shrink: 0;
        }}

        .social-links {{
            display: flex;
            align-items: center;
            gap: 12px;
        }}

        .social-link {{
             display: inline-flex;
    align-items: center;
    justify-content: center;
    color: #FFFFFF !important;
    opacity: 0.92;
    transition: opacity 0.2s ease, transform 0.2s ease;
    text-decoration: none !important;
        }}
        .social-link svg{{
     width: 22px;
    height: 22px;
    display: block;
    fill: #FFFFFF;
        }}

        .social-link:hover {{
            opacity: 1;
    transform: scale(1.15);
    color: #FFFFFF !important;
        }}

        .language {{
            display: flex;
            align-items: center;
            gap: 7px;

            color: white;
            font-size: 12px;
            font-weight: 600;
        }}

        .globe {{
            font-size: 22px;
            line-height: 1;
        }}

        .search-icon {{
            font-size: 25px;
            color: white;
            line-height: 1;
        }}

        /* ============================================================
           HERO
           ============================================================ */

        .hero {{
            position: relative;

            width: 100%;
            height: 380px;

            overflow: hidden;

            background-image:
                linear-gradient(
                    180deg,
                    rgba(0, 63, 135, 0.18),
                    rgba(0, 58, 135, 0.72)
                ),
                url("{HERO_IMAGE_URL}");

            background-size: cover;
            background-position: center;

            display: flex;
            flex-direction: column;
            align-items: center;

            box-sizing: border-box;

            padding-top: 55px;
            margin: 0 !important;
            border: none !important;
        }}

        .hero-search-placeholder {{
            width: 340px;
            max-width: calc(100% - 40px);
            height: 42px;
        }}

        /*
         * Streamlit selectbox positioned right over the hero placeholder
         */

        div[data-testid="stSelectbox"] {{
            width: 340px !important;
            max-width: calc(100% - 40px) !important;

            margin-left: auto !important;
            margin-right: auto !important;
            
            margin-top: -325px !important;
            margin-bottom: 283px !important;

            position: relative;
            z-index: 10;
        }}

        div[data-testid="stSelectbox"] label {{
            display: none !important;
        }}

        div[data-testid="stSelectbox"] > div {{
            width: 100% !important;
        }}

        div[data-testid="stSelectbox"] > div > div {{
            background: rgba(255,255,255,0.97) !important;

            border: 1px solid #FFFFFF !important;
            border-radius: 3px !important;

            min-height: 40px !important;

            box-shadow: 0 2px 8px rgba(0,0,0,0.2) !important;
        }}

        div[data-testid="stSelectbox"] [data-baseweb="select"] {{
            background: transparent !important;
        }}

        div[data-testid="stSelectbox"] [data-baseweb="select"] > div {{
            min-height: 38px !important;

            background: transparent !important;

            border: none !important;
            box-shadow: none !important;
        }}

        div[data-testid="stSelectbox"] input {{
            font-size: 16px !important;
        }}

        div[data-testid="stSelectbox"] [data-testid="stMarkdownContainer"] {{
            color: #222222 !important;
            font-size: 15px !important;
            font-weight: 500 !important;
        }}

        .hero-tagline {{
            color: white;

            font-size: 22px;
            font-weight: 500;

            margin-top: 18px;

            text-align: center;

            text-shadow: 0 1px 3px rgba(0,0,0,0.35);
        }}

        .hero-bottom {{
            position: absolute;

            bottom: 24px;
            left: 25px;
            right: 25px;

            display: flex;
            align-items: flex-end;
            justify-content: space-between;
        }}

        .hero-credit {{
            color: rgba(255,255,255,0.95);

            font-size: 15px;
            font-weight: 700;
        }}

        .epa-style {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .epa-symbol {{
            width: 19px;
            height: 19px;

            border-radius: 50%;

            background: white;

            color: {NAV_BLUE};

            display: flex;
            align-items: center;
            justify-content: center;

            font-size: 11px;
            font-weight: 700;
        }}

        /* ============================================================
           EXPLORE SECTION & DASHBOARD CONTENT (CENTERED)
           ============================================================ */

        .explore-section {{
            background: #F7F7F7;
            padding: 21px 20px 34px 20px;
            max-width: 1110px;
            margin: 0 auto;
            box-sizing: border-box;
        }}

        .explore-title {{
            text-align: center;

            font-family: "Roboto Condensed", Arial, sans-serif;

            font-size: 31px;
            font-weight: 400;

            color: #222;

            margin: 0 0 17px 0;
        }}

        .explore-grid {{
            display: grid;

            grid-template-columns: repeat(3, 1fr);

            gap: 17px;
        }}

        .explore-card {{
            min-height: 162px;

            background: white;

            border: 1px solid #C8C8C8;

            box-sizing: border-box;

            padding: 15px 15px 22px;

            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: flex-start;

            text-decoration: none !important;

            transition:
                border-color 0.2s ease,
                box-shadow 0.2s ease;
        }}

        .explore-card:hover {{
            border-color: {NAV_BLUE};

            box-shadow:
                0 3px 10px rgba(0,0,0,0.08);
        }}

        .explore-label {{
            font-family: Arial, Helvetica, sans-serif;

            font-size: 19px;
            font-weight: 700;

            color: #202020;

            text-align: center;

            margin-bottom: 22px;
        }}

        .explore-icon-circle {{
            width: 63px;
            height: 63px;

            border-radius: 50%;

            background: {CARD_RED};

            display: flex;
            align-items: center;
            justify-content: center;

            color: white;

            font-size: 32px;

            line-height: 1;
        }}

        /* Center dashboard content below hero */
        div[data-testid="stHorizontalBlock"],
        div[data-testid="stElementContainer"]:has(.eyebrow),
        div[data-testid="stElementContainer"]:has(.alert-banner),
        div[data-testid="stElementContainer"]:has(.forecast-grid),
        div[data-testid="stElementContainer"]:has(.stPlotlyChart),
        div[data-testid="stElementContainer"]:has(.stDataFrame),
        div[data-testid="stElementContainer"]:has(.site-footer),
        div[data-testid="stElementContainer"]:has(.stCaption),
        div[data-testid="stElementContainer"]:has(.stAlert) {{
            max-width: 1110px !important;
            margin-left: auto !important;
            margin-right: auto !important;
            padding-left: 20px !important;
            padding-right: 20px !important;
            box-sizing: border-box !important;
        }}

        .eyebrow {{
            font-family: Arial, Helvetica, sans-serif;

            color: #171717;

            font-size: 20px;
            font-weight: 700;

            margin-top: 28px;
            margin-bottom: 8px;
        }}

        .section-title {{
            font-family: Arial, Helvetica, sans-serif;

            color: #171717;

            font-size: 20px;
            font-weight: 700;

            margin: 0 0 6px 0;
        }}

        .section-description {{
            color: {TEXT_SOFT};

            font-size: 14px;

            margin-bottom: 14px;
        }}

        /* ============================================================
           AQI STATION READOUT
           ============================================================ */

        .station-heading {{
            font-family: "Roboto Condensed", Arial, sans-serif;

            font-size: 31px;
            font-weight: 500;

            color: #222;

            margin: 0;
        }}

        .station-status {{
            font-size: 16px;
            font-weight: 700;
        }}

        .station-update {{
            color: {TEXT_SOFT};
            font-size: 13px;
        }}

        /* ============================================================
           METRIC CARDS
           ============================================================ */

        .metric-grid {{
            display: grid;

            grid-template-columns: repeat(3, 1fr);

            gap: 12px;

            margin-top: 17px;
        }}

        .metric-card {{
            background: #FAFAFA;

            border: 1px solid #D3D3D3;

            padding: 15px;

            min-height: 88px;

            box-sizing: border-box;
        }}

        .metric-label {{
            color: #666;

            font-size: 11px;
            font-weight: 700;

            text-transform: uppercase;
            letter-spacing: 0.08em;
        }}

        .metric-value {{
            color: #171717;

            font-size: 27px;
            font-weight: 600;

            margin-top: 7px;
        }}

        .metric-unit {{
            color: #666;

            font-size: 12px;
            font-weight: 400;
        }}

        /* ============================================================
           ALERT BANNER
           ============================================================ */

        .alert-banner {{
            background: #FFF4F1;

            border: 1px solid #E4B9AD;
            border-left: 5px solid #B21F2D;

            padding: 13px 16px;

            margin: 18px 0;

            color: #222;

            font-size: 14px;
            line-height: 1.5;
        }}

        /* ============================================================
           FORECAST
           ============================================================ */

        .forecast-grid {{
            display: grid;

            grid-template-columns: repeat(3, 1fr);

            gap: 12px;

            margin-bottom: 25px;
        }}

        .forecast-card {{
            background: white;

            border: 1px solid #CFCFCF;

            padding: 15px;

            box-sizing: border-box;
        }}

        .forecast-day {{
            color: #666;

            font-size: 11px;
            font-weight: 700;

            text-transform: uppercase;
            letter-spacing: 0.08em;
        }}

        .forecast-value {{
            color: #171717;

            font-size: 29px;
            font-weight: 600;

            margin-top: 6px;
        }}

        .forecast-delta {{
            color: #666;

            font-size: 12px;

            margin-top: 3px;
        }}

        /* ============================================================
           PLOTLY
           ============================================================ */

        [data-testid="stPlotlyChart"] {{
            border: 1px solid #D4D4D4;

            background: white;

            padding: 3px;

            box-sizing: border-box;
        }}

        /* ============================================================
           DATAFRAME
           ============================================================ */

        [data-testid="stDataFrame"] {{
            border: 1px solid #D0D0D0;
        }}

        /* ============================================================
           FOOTER
           ============================================================ */

        .site-footer {{
            margin-top: 35px;

            border-top: 1px solid #D4D4D4;

            padding: 24px 0 32px;

            color: #666;

            font-size: 12px;

            text-align: center;

            letter-spacing: 0.03em;

            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 14px;
        }}

        .footer-socials {{
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 22px;
        }}

        .footer-social-link {{
            display: inline-flex;
            align-items: center;
            gap: 7px;
            color: {NAV_BLUE} !important;
            text-decoration: none !important;
            font-weight: 600;
            font-size: 13px;
            transition: color 0.2s ease, transform 0.2s ease;
        }}

        .footer-social-link:hover {{
            color: {NAV_BLUE_DARK} !important;
            text-decoration: underline !important;
            transform: translateY(-1px);
        }}

        /* ============================================================
           STREAMLIT SPACING CLEANUP
           ============================================================ */

        div[data-testid="stVerticalBlock"] {{
            gap: 0 !important;
        }}

        div[data-testid="stElementContainer"] {{
            margin: 0 !important;
            padding: 0 !important;
        }}

        .stMarkdown {{
            margin-bottom: 0 !important;
        }}

        .stCaption {{
            color: #666 !important;
        }}

        /* ============================================================
           RESPONSIVE
           ============================================================ */

        @media (max-width: 950px) {{

            .nav-links {{
                gap: 12px;
            }}

            .nav-link {{
                font-size: 11px;
            }}

            .pearls-logo {{
                width: 140px;
            }}

            .explore-grid {{
                grid-template-columns: 1fr;
            }}

            .explore-card {{
                min-height: 145px;
            }}
        }}

        @media (max-width: 720px) {{

            .top-nav {{
                min-height: 64px;
                padding: 0 12px;
            }}

            .nav-inner {{
                max-width: calc(100% - 10px);
            }}

            .nav-links {{
                display: none;
            }}

            .nav-right {{
                margin-left: auto;
            }}

            .hero {{
                height: 340px;
                padding-top: 45px;
            }}

            .hero-tagline {{
                font-size: 18px;
                margin-top: 14px;
            }}

            div[data-testid="stSelectbox"] {{
                margin-top: -295px !important;
                margin-bottom: 253px !important;
            }}

            .hero-bottom {{
                bottom: 18px;
                flex-direction: column;
                align-items: flex-start;
                gap: 6px;
            }}

            .metric-grid {{
                grid-template-columns: 1fr;
            }}

            .forecast-grid {{
                grid-template-columns: 1fr;
            }}
        }}

        </style>
""")


def _html(content: str) -> None:
    st.html(content)


def _polar(
    cx: float,
    cy: float,
    r: float,
    angle_deg: float,
) -> tuple[float, float]:

    a = math.radians(angle_deg)

    return (
        cx + r * math.cos(a),
        cy + r * math.sin(a),
    )


def _arc_path(
    cx: float,
    cy: float,
    r: float,
    a0: float,
    a1: float,
) -> str:

    x0, y0 = _polar(cx, cy, r, a0)
    x1, y1 = _polar(cx, cy, r, a1)

    large_arc = 1 if (a1 - a0) > 180 else 0

    return (
        f"M {x0:.2f} {y0:.2f} "
        f"A {r} {r} 0 {large_arc} 1 {x1:.2f} {y1:.2f}"
    )


def _dial_svg(
    aqi_value: float,
    color: str,
    category: str,
) -> str:

    cx, cy, r = 130, 128, 96

    start = 135
    end = 405

    span = end - start

    segments = []

    for (v0, c0), (v1, _c1) in zip(
        AQI_BREAKPOINTS[:-1],
        AQI_BREAKPOINTS[1:],
    ):

        a0 = start + (v0 / 500) * span
        a1 = start + (v1 / 500) * span

        segments.append(
            f"""
            <path
                d="{_arc_path(cx, cy, r, a0, a1)}"
                stroke="{c0}"
                stroke-width="18"
                fill="none"
            />
            """
        )

    pct = max(
        0.0,
        min(1.0, aqi_value / 500),
    )

    needle_angle = start + pct * span

    nx, ny = _polar(
        cx,
        cy,
        r - 8,
        needle_angle,
    )

    return f"""
    <svg
        viewBox="0 0 260 240"
        width="260"
        height="240"
        xmlns="http://www.w3.org/2000/svg"
    >

        {''.join(segments)}

        <line
            x1="{cx}"
            y1="{cy}"
            x2="{nx:.2f}"
            y2="{ny:.2f}"
            stroke="#202020"
            stroke-width="3"
            stroke-linecap="round"
        />

        <circle
            cx="{cx}"
            cy="{cy}"
            r="6"
            fill="#202020"
        />

        <text
            x="{cx}"
            y="{cy - 14}"
            text-anchor="middle"
            font-family="Arial"
            font-size="32"
            font-weight="700"
            fill="{color}"
        >
            {int(round(aqi_value))}
        </text>

        <text
            x="{cx}"
            y="{cy + 9}"
            text-anchor="middle"
            font-family="Arial"
            font-size="11"
            letter-spacing="0.08em"
            fill="#666666"
        >
            AQI
        </text>

        <text
            x="{cx}"
            y="{cy + 32}"
            text-anchor="middle"
            font-family="Arial"
            font-size="14"
            font-weight="600"
            fill="{color}"
        >
            {category}
        </text>

    </svg>
    """


# -----------------------------------------------------------------------
# Hopsworks connection
# -----------------------------------------------------------------------

@st.cache_resource
def _connect():
    """
    Connect to Hopsworks (cached).
    Returns None if unavailable.
    """

    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parents[1] / ".env")

        api_key = os.getenv("HOPSWORKS_API_KEY", "").strip()
        if not api_key:
            try:
                api_key = st.secrets["HOPSWORKS_API_KEY"]
            except Exception:
                pass

        if not api_key:
            return None

        import hopsworks

        project_name = os.getenv("HOPSWORKS_PROJECT_NAME", "").strip()
        if project_name:
            project = hopsworks.login(
                project=project_name,
                api_key_value=api_key,
            )
        else:
            project = hopsworks.login(api_key_value=api_key)

        return project.get_feature_store()
    except Exception:
        return None


# -----------------------------------------------------------------------
# Feature group reader
# -----------------------------------------------------------------------

@st.cache_data(ttl=60)
def _read_fg(
    name: str,
    version: int = 1,
) -> pd.DataFrame:

    """
    Read a Feature Group.

    Hopsworks is attempted first.
    Local parquet is used as fallback.
    """

    fs = _connect()

    if fs is not None:

        try:

            fg = fs.get_feature_group(
                name,
                version,
            )

            df = fg.read()

            df.columns = [
                c.lower()
                for c in df.columns
            ]

            if not df.empty:
                return df

        except Exception:
            pass

    local_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / f"{name}.parquet"
    )

    if local_path.exists():

        try:

            df = pd.read_parquet(
                local_path
            )

            df.columns = [
                c.lower()
                for c in df.columns
            ]

            return df

        except Exception:
            pass

    return pd.DataFrame()


# -----------------------------------------------------------------------
# Start application
# -----------------------------------------------------------------------

_inject_css()


# =======================================================================
# LOAD DATA
# =======================================================================

names = {
    c.id: c.name
    for c in CITIES
}

features = _read_fg(
    "aqi_features"
)


# -----------------------------------------------------------------------
# No data state
# -----------------------------------------------------------------------

if features.empty:

    st.html("""
        <div class="site-content">
            <div style="
                padding:60px 30px;
                text-align:center;
                background:white;
            ">
                <h2>No feature data found</h2>
                <p>
                    Run the data pipelines first.
                </p>
            </div>
        </div>
        """)

    st.stop()


# =======================================================================
# LATEST CITY DATA
# =======================================================================

latest_all = (
    features
    .sort_values(
        "event_time",
        ascending=False,
    )
    .groupby(
        "city_id",
        as_index=False,
    )
    .first()
)


options = [
    c
    for c in latest_all["city_id"].tolist()
    if c in names
]


if not options:

    st.html("""
        <div class="site-content">
            <div style="
                padding:60px 30px;
                text-align:center;
                background:white;
            ">
                <h2>No forecast data found</h2>
                <p>
                    Check the configured cities and feature data.
                </p>
            </div>
        </div>
        """)

    st.stop()


# =======================================================================
# ALERT DATA
# =======================================================================

alerts_all = _read_fg(
    "alerts"
)

alert_count = (
    0
    if alerts_all.empty
    else len(alerts_all)
)


# =======================================================================
# HEADER: TOP NAVIGATION + HERO
# =======================================================================

nav_links_html = ""

nav_links_html += (
    '<a class="nav-link" href="#trend">'
    'Air Quality Trend'
    '</a>'
)

nav_links_html += (
    '<a class="nav-link" href="#forecast-drivers">'
    'Why This Forecast'
    '</a>'
)

nav_links_html += (
    '<a class="nav-link" href="#model-diagnostics">'
    'Model Diagnostics'
    '</a>'
)


st.html(f"""
    <div class="top-nav">
        <div class="nav-inner">

            <div class="pearls-logo">
                <div class="pearls-logo-ring"></div>
                <div class="pearls-logo-ring-2"></div>

                <div class="pearls-logo-text">
                    Pearl<span>AQI</span>
                </div>
            </div>

            <div class="nav-links">
                {nav_links_html}
            </div>

            <div class="nav-right">

                <a href="https://github.com/Rabia605" target="_blank" rel="noopener noreferrer" class="social-link" title="GitHub (@Rabia605)">
                    <img src="{GITHUB_ICON_DATA}" width="22" height="22" alt="GitHub" />
                </a>

                <a href="https://www.linkedin.com/in/rabianoreen" target="_blank" rel="noopener noreferrer" class="social-link" title="LinkedIn (rabianoreen)">
                    <img src="{LINKEDIN_ICON_DATA}" width="22" height="22" alt="LinkedIn" />
                </a>

                

            </div>


        </div>
    </div>

    <div class="hero">

        <div class="hero-search-placeholder"></div>

        <div class="hero-tagline">
            {APP_TAGLINE}
        </div>

        <div class="hero-bottom">

            <div class="hero-credit epa-style">
                <span class="epa-symbol">P</span>
                <span>Pearl AQI Predictor</span>
            </div>

            <div class="hero-credit">
                AIR QUALITY INFORMATION
            </div>

        </div>

    </div>
    """)


# -----------------------------------------------------------------------
# City selector
# -----------------------------------------------------------------------

city_id = st.selectbox(
    "City",
    options,
    format_func=lambda c: names.get(
        c,
        c,
    ),
    index=0,
    label_visibility="collapsed",
)


# =======================================================================
# CURRENT DATA
# =======================================================================

current = latest_all[
    latest_all["city_id"] == city_id
].iloc[0]


# =======================================================================
# FORECAST
# =======================================================================

forecast = _read_fg(
    "predictions"
)


if not forecast.empty:

    forecast = forecast[
        forecast["city_id"] == city_id
    ].sort_values(
        "horizon_h"
    )


# =======================================================================
# CURRENT AQI
# =======================================================================

aqi_now = float(
    current["aqi"]
)

category = aqi_category(
    aqi_now
)

color = aqi_color(
    aqi_now
)

last_updated = (
    pd.to_datetime(
        current["event_time"]
    ).strftime(
        "%d %b, %H:%M UTC"
    )
)


# =======================================================================
# EXPLORE
# =======================================================================

st.html("""
    <div class="explore-section">

        <div class="explore-title">
            Explore
        </div>

        <div class="explore-grid">

            <a
                class="explore-card"
                href="#trend"
            >

                <div class="explore-label">
                    Air Quality Trend
                </div>

                <div class="explore-icon-circle">
                    📈
                </div>

            </a>


            <a
                class="explore-card"
                href="#forecast-drivers"
            >

                <div class="explore-label">
                    Why This Forecast
                </div>

                <div class="explore-icon-circle">
                    🧭
                </div>

            </a>


            <a
                class="explore-card"
                href="#model-diagnostics"
            >

                <div class="explore-label">
                    Model Diagnostics
                </div>

                <div class="explore-icon-circle">
                    📊
                </div>

            </a>

        </div>

    </div>
    """)


# =======================================================================
# DASHBOARD (Removed invalid HTML wrapper)
# =======================================================================


# =======================================================================
# STATION READOUT
# =======================================================================

st.html("""
    <div class="eyebrow">
        Station Readout
    </div>
    """)


dial_col, info_col = st.columns(
    [1, 2],
    gap="large",
)


# -----------------------------------------------------------------------
# AQI dial
# -----------------------------------------------------------------------

with dial_col:

    st.html(
        _dial_svg(
            aqi_now,
            color,
            category,
        )
    )


# -----------------------------------------------------------------------
# Current information
# -----------------------------------------------------------------------

with info_col:

    st.html(f"""
        <h1 class="station-heading">
            {names[city_id]}
        </h1>
        """)

    st.html(f"""
        <div>

            <span
                class="station-status"
                style="color:{color};"
            >
                {category}
            </span>

            <span class="station-update">
                · updated {last_updated}
            </span>

        </div>
        """)


    # ---------------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------------

    st.html(f"""
        <div class="metric-grid">

            <div class="metric-card">

                <div class="metric-label">
                    PM2.5
                </div>

                <div class="metric-value">
                    {current['pm25']:.0f}
                    <span class="metric-unit">
                        µg/m³
                    </span>
                </div>

            </div>


            <div class="metric-card">

                <div class="metric-label">
                    Temperature
                </div>

                <div class="metric-value">
                    {current['temp_c']:.0f}°
                    <span class="metric-unit">
                        C
                    </span>
                </div>

            </div>


            <div class="metric-card">

                <div class="metric-label">
                    Wind
                </div>

                <div class="metric-value">
                    {current['wind_speed']:.1f}
                    <span class="metric-unit">
                        m/s
                    </span>
                </div>

            </div>

        </div>
        """)


# =======================================================================
# CITY ALERT
# =======================================================================

if not alerts_all.empty:

    alert = alerts_all[
        alerts_all["city_id"] == city_id
    ]

    if not alert.empty:

        a = alert.iloc[0]

        st.html(f"""
            <div class="alert-banner">

                <strong>
                    Air reaching {a['category']}
                    (AQI {a['peak_aqi']:.0f})
                    in about
                    {int(a['starts_in_h'])}
                    hours.
                </strong>

                {a['advice']}

                <br>

                <span style="color:#666;">
                    Most affected:
                    {a['affects']}
                </span>

            </div>
            """)


# =======================================================================
# 3-DAY OUTLOOK
# =======================================================================

if not forecast.empty:

    st.html("""
        <div class="eyebrow">
            3-Day Outlook
        </div>

        <div class="section-description">
            Predicted air quality for the coming days.
        </div>
        """)


    cards = []

    for row in forecast.itertuples():

        day = int(
            row.horizon_h // 24
        )

        delta = (
            row.predicted_aqi
            - aqi_now
        )

        arrow = (
            "▲"
            if delta > 0
            else (
                "▼"
                if delta < 0
                else "•"
            )
        )

        cards.append(
            f"""
            <div class="forecast-card">

                <div class="forecast-day">
                    In {day}
                    day{'s' if day != 1 else ''}
                </div>

                <div class="forecast-value">
                    {int(round(row.predicted_aqi))}
                </div>

                <div class="forecast-delta">
                    {arrow}
                    {delta:+.0f}
                    vs now ·
                    {row.category}
                </div>

            </div>
            """
        )


    st.html(f"""
        <div class="forecast-grid">
            {"".join(cards)}
        </div>
        """)


# =======================================================================
# AIR QUALITY TREND
# =======================================================================

st.html("""
    <div
        class="eyebrow"
        id="trend"
    >
        Air Quality Trend
    </div>
    """)


history = features[
    (features["city_id"] == city_id)
    &
    (
        features["event_time"]
        >= (
            pd.Timestamp.now(tz="UTC")
            - pd.Timedelta(days=7)
        )
    )
].sort_values(
    "event_time"
)


if not history.empty:

    fig = go.Figure()


    # ---------------------------------------------------------------
    # Measured AQI
    # ---------------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=history["event_time"],
            y=history["aqi"],

            mode="lines",

            name="Measured",

            line=dict(
                color=ACCENT,
                width=2.5,
            ),

            hovertemplate=(
                "%{x|%d %b, %H:%M}"
                "<br>AQI %{y:.0f}"
                "<extra>Measured</extra>"
            ),
        )
    )


    # ---------------------------------------------------------------
    # Forecast
    # ---------------------------------------------------------------

    if not forecast.empty:

        fc = forecast.rename(
            columns={
                "forecast_time": "time",
                "predicted_aqi": "Forecast",
            }
        )


        bridge_x = pd.concat(
            [
                pd.Series(
                    [
                        history[
                            "event_time"
                        ].iloc[-1]
                    ]
                ),
                fc["time"],
            ]
        )


        bridge_y = pd.concat(
            [
                pd.Series(
                    [
                        history[
                            "aqi"
                        ].iloc[-1]
                    ]
                ),
                fc["Forecast"],
            ]
        )
        


        fig.add_trace(
            go.Scatter(
                x=bridge_x,
                y=bridge_y,

                mode="lines",

                name="Forecast",

                line=dict(
                    color="#B54834",
                    width=2.5,
                    dash="dash",
                ),

                hovertemplate=(
                    "%{x|%d %b, %H:%M}"
                    "<br>AQI %{y:.0f}"
                    "<extra>Forecast</extra>"
                ),
            )
        )


    # ---------------------------------------------------------------
    # Chart styling
    # ---------------------------------------------------------------

    fig.update_layout(

        height=340,

        margin=dict(
            l=10,
            r=10,
            t=15,
            b=10,
        ),

        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",

        font=dict(
            family="Arial, sans-serif",
            color="#222222",
            size=12,
        ),

        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),

        xaxis=dict(
            showgrid=False,
            linecolor="#D0D0D0",
        ),

        yaxis=dict(
            showgrid=True,
            gridcolor="#E4E4E4",
            title="AQI",
        ),

        hovermode="x unified",
    )


    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": False
        },
    )


else:

    st.info(
        "Not enough history to display the trend chart."
    )


# =======================================================================
# TWO-COLUMN ANALYSIS
# =======================================================================

left, right = st.columns(
    2,
    gap="large",
)


# =======================================================================
# WHY THIS FORECAST
# =======================================================================

with left:

    st.html("""
        <div
            class="eyebrow"
            id="forecast-drivers"
        >
            Why This Forecast
        </div>
        """)

    st.caption(
        "SHAP how much each factor moved tomorrow's prediction."
    )


    drivers = _read_fg(
        "forecast_drivers"
    )


    if not drivers.empty:

        drivers = drivers[
            drivers["city_id"] == city_id
        ]


    if drivers.empty:

        st.info(
            "Explanations refresh with the daily training run."
        )


    else:

        drivers = (
            drivers
            .sort_values(
                "contribution",
                key=abs,
                ascending=False,
            )
            .head(5)
        )

        drivers = drivers.sort_values(
            "contribution"
        )


        bar_colors = [
            "#B54834"
            if v > 0
            else ACCENT
            for v in drivers[
                "contribution"
            ]
        ]


        fig = go.Figure(
            go.Bar(
                x=drivers[
                    "contribution"
                ],

                y=drivers[
                    "label"
                ],

                orientation="h",

                marker_color=bar_colors,

                hovertemplate=(
                    "%{y}: %{x:+.2f}"
                    "<extra></extra>"
                ),
            )
        )


        fig.update_layout(

            height=260,

            margin=dict(
                l=10,
                r=10,
                t=10,
                b=10,
            ),

            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",

            font=dict(
                family="Arial, sans-serif",
                color="#222222",
                size=12,
            ),

            xaxis=dict(
                showgrid=True,
                gridcolor="#E4E4E4",
                zeroline=True,
                zerolinecolor="#CCCCCC",
            ),

            yaxis=dict(
                showgrid=False
            ),
        )


        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                "displayModeBar": False
            },
        )


        st.caption(
            "Positive values raise the predicted AQI; negative lower it."
        )


# =======================================================================
# WHAT'S IN THE AIR
# =======================================================================

with right:

    st.html("""
        <div
            class="eyebrow"
            id="pollutants"
        >
            What's in the Air
        </div>
        """)

    st.caption(
        "Current concentrations vs. WHO guideline (µg/m³)."
    )


    pollutants = pd.DataFrame(
        {
            "Pollutant": [
                "PM2.5",
                "PM10",
                "O₃",
                "NO₂",
                "SO₂",
                "CO",
            ],

            "Value": [
                current["pm25"],
                current["pm10"],
                current["o3"],
                current["no2"],
                current["so2"],
                current["co"],
            ],

            "WHO guideline": [
                15,
                45,
                100,
                25,
                40,
                4000,
            ],
        }
    )


    bar_colors = [
        "#B54834"
        if v > g
        else ACCENT

        for v, g in zip(
            pollutants["Value"],
            pollutants["WHO guideline"],
        )
    ]


    fig = go.Figure(
        go.Bar(

            x=pollutants["Value"],

            y=pollutants[
                "Pollutant"
            ],

            orientation="h",

            marker_color=bar_colors,

            hovertemplate=(
                "%{y}: %{x:.1f} µg/m³"
                "<extra></extra>"
            ),
        )
    )


    fig.add_trace(
        go.Scatter(

            x=pollutants[
                "WHO guideline"
            ],

            y=pollutants[
                "Pollutant"
            ],

            mode="markers",

            name="WHO guideline",

            marker=dict(
                symbol="line-ns",
                size=22,
                line=dict(
                    color="#555555",
                    width=2,
                ),
            ),

            hovertemplate=(
                "WHO guideline: %{x}"
                "<extra></extra>"
            ),
        )
    )


    fig.update_layout(

        height=260,

        margin=dict(
            l=10,
            r=10,
            t=10,
            b=10,
        ),

        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",

        font=dict(
            family="Arial, sans-serif",
            color="#222222",
            size=12,
        ),

        xaxis=dict(
            showgrid=True,
            gridcolor="#E4E4E4",
        ),

        yaxis=dict(
            showgrid=False
        ),

        showlegend=False,
    )


    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": False
        },
    )


    st.caption(
        "Tick mark shows the WHO annual guideline for each pollutant."
    )


# =======================================================================
# MODEL DIAGNOSTICS
# =======================================================================

st.html("""
    <div
        class="eyebrow"
        id="model-diagnostics"
    >
        Model Diagnostics
    </div>
    """)


scores = _read_fg(
    "model_registry"
)


if scores.empty:

    st.info(
        "No training runs recorded yet."
    )


else:

    latest_run = (
        scores
        .sort_values(
            "trained_at",
            ascending=False,
        )["run_id"]
        .iloc[0]
    )


    display = (
        scores[
            scores["run_id"]
            == latest_run
        ][
            [
                "model_name",
                "horizon_h",
                "rmse",
                "mae",
                "r2",
            ]
        ]

        .sort_values(
            [
                "horizon_h",
                "rmse",
            ]
        )

        .rename(
            columns={
                "model_name": "Model",
                "horizon_h": "Horizon (h)",
                "rmse": "RMSE",
                "mae": "MAE",
                "r2": "R²",
            }
        )

        .round(3)
    )


    try:

        styled = (
            display.style
            .background_gradient(
                subset=[
                    "RMSE",
                    "MAE",
                ],
                cmap="Reds",
            )
            .background_gradient(
                subset=["R²"],
                cmap="Greens",
            )
            .format(
                precision=3
            )
        )

        st.dataframe(
            styled,
            hide_index=True,
            use_container_width=True,
        )

    except Exception:

        st.dataframe(
            display,
            hide_index=True,
            use_container_width=True,
        )


# =======================================================================
# FOOTER
# =======================================================================

st.html(f"""
    <div class="site-footer">
        {APP_NAME.upper()} · © 2026 Rabia Noreen · All Rights Reserved
    </div>
    """)
