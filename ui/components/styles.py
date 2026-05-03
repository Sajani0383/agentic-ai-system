import streamlit as st
from ui.config import PALETTE

def inject_styles():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background:
                radial-gradient(circle at top left, rgba(77,163,255,0.12), transparent 28%),
                linear-gradient(180deg, #09111b 0%, {PALETTE["bg"]} 100%);
        }}
        .stApp,
        .stApp > div,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        [data-testid="stVerticalBlock"],
        [data-testid="stBlock"],
        [data-testid="stTabs"] {{
            opacity: 1 !important;
            filter: none !important;
            backdrop-filter: none !important;
        }}
        [data-testid="stStatusWidget"],
        [data-testid="stDecoration"],
        [data-testid="stToolbar"],
        [data-testid="stDeployButton"],
        [data-testid="stModal"],
        [data-testid="stDialog"],
        div[role="dialog"],
        div[data-baseweb="modal"] {{
            backdrop-filter: none !important;
        }}
        div[data-baseweb="modal"] > div {{
            background: transparent !important;
        }}
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #1c202b 0%, #222632 100%);
            border-right: 1px solid rgba(255,255,255,0.06);
        }}
        .hero-title {{
            font-size: 3.35rem;
            line-height: 1.02;
            font-weight: 800;
            color: {PALETTE["text"]};
            letter-spacing: -0.04em;
            margin-bottom: 0.45rem;
        }}
        .hero-sub {{
            color: {PALETTE["muted"]};
            font-size: 1.03rem;
            max-width: 56rem;
        }}
        .event-banner {{
            margin: 1rem 0 0.9rem 0;
            padding: 1rem 1.2rem;
            border-radius: 18px;
            border: 1px solid rgba(255,255,255,0.08);
            background: linear-gradient(135deg, rgba(77,163,255,0.16), rgba(216,200,110,0.10));
            box-shadow: 0 18px 50px rgba(0,0,0,0.16);
        }}
        .event-title {{
            color: {PALETTE["text"]};
            font-size: 1.3rem;
            font-weight: 700;
            margin-bottom: 0.3rem;
        }}
        .event-copy {{
            color: {PALETTE["muted"]};
            font-size: 0.96rem;
        }}
        .signal-grid {{
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.9rem;
            margin: 0.7rem 0 1.1rem;
        }}
        .signal-card {{
            background: linear-gradient(180deg, rgba(18,31,47,0.94), rgba(11,22,34,0.94));
            border: 1px solid {PALETTE["border"]};
            border-radius: 18px;
            padding: 1rem 1.05rem;
            box-shadow: 0 16px 50px rgba(0,0,0,0.22);
        }}
        .signal-label {{
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: {PALETTE["muted"]};
            margin-bottom: 0.35rem;
        }}
        .signal-value {{
            font-size: 1.45rem;
            font-weight: 700;
            color: {PALETTE["text"]};
            line-height: 1.15;
        }}
        .signal-note {{
            margin-top: 0.42rem;
            color: {PALETTE["muted"]};
            font-size: 0.9rem;
        }}
        .story-grid {{
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.95rem;
            margin: 0.9rem 0 1.1rem;
        }}
        .story-card {{
            background: linear-gradient(180deg, rgba(16,27,43,0.96), rgba(10,18,29,0.96));
            border: 1px solid {PALETTE["border"]};
            border-radius: 22px;
            padding: 1rem 1.05rem;
            box-shadow: 0 16px 44px rgba(0,0,0,0.24);
        }}
        .story-title {{
            color: {PALETTE["muted"]};
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.16em;
            margin-bottom: 0.45rem;
        }}
        .story-value {{
            color: {PALETTE["text"]};
            font-size: 1.7rem;
            font-weight: 800;
            line-height: 1.05;
            margin-bottom: 0.45rem;
        }}
        .story-copy {{
            color: {PALETTE["muted"]};
            font-size: 0.95rem;
            line-height: 1.55;
        }}
        .zone-grid {{
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 0.85rem;
            margin: 0.6rem 0 1rem;
        }}
        .zone-card {{
            background: linear-gradient(180deg, rgba(14,24,37,0.96), rgba(8,16,26,0.96));
            border: 1px solid rgba(255,255,255,0.07);
            border-radius: 20px;
            padding: 0.95rem;
        }}
        .zone-name {{
            color: {PALETTE["text"]};
            font-size: 1.02rem;
            font-weight: 700;
            margin-bottom: 0.28rem;
        }}
        .zone-meta {{
            color: {PALETTE["muted"]};
            font-size: 0.84rem;
            margin-bottom: 0.7rem;
        }}
        .util-track {{
            width: 100%;
            height: 9px;
            background: rgba(255,255,255,0.08);
            border-radius: 999px;
            overflow: hidden;
            margin: 0.55rem 0 0.35rem;
        }}
        .util-fill {{
            height: 100%;
            border-radius: 999px;
        }}
        .zone-stat-row {{
            display: flex;
            justify-content: space-between;
            gap: 0.5rem;
            color: {PALETTE["muted"]};
            font-size: 0.83rem;
            margin-top: 0.35rem;
        }}
        .chip {{
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.28rem 0.62rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 700;
            border: 1px solid transparent;
        }}
        .chip-blue {{
            background: rgba(77,163,255,0.12);
            color: #8ec8ff;
            border-color: rgba(77,163,255,0.25);
        }}
        .chip-green {{
            background: rgba(75,211,138,0.12);
            color: #81e6aa;
            border-color: rgba(75,211,138,0.25);
        }}
        .chip-gold {{
            background: rgba(216,200,110,0.12);
            color: #ebdf91;
            border-color: rgba(216,200,110,0.25);
        }}
        .chip-coral {{
            background: rgba(255,116,108,0.12);
            color: #ff9e98;
            border-color: rgba(255,116,108,0.25);
        }}
        .feature-callout {{
            background: linear-gradient(135deg, rgba(110,223,246,0.10), rgba(159,140,255,0.08));
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 20px;
            padding: 1rem 1.05rem;
            margin: 0.5rem 0 1rem;
        }}
        .section-kicker {{
            color: {PALETTE["muted"]};
            text-transform: uppercase;
            letter-spacing: 0.14em;
            font-size: 0.74rem;
            margin-bottom: 0.28rem;
        }}
        .section-copy {{
            color: {PALETTE["muted"]};
            margin-top: -0.2rem;
            margin-bottom: 0.75rem;
            font-size: 0.93rem;
        }}
        .status-bar {{
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.65rem;
            margin: 0.7rem 0 0.45rem;
            padding: 0.85rem;
            border-radius: 8px;
            border: 1px solid rgba(77,163,255,0.22);
            background: rgba(12,23,35,0.95);
        }}
        .status-bar div {{
            min-width: 0;
            border-right: 1px solid rgba(255,255,255,0.07);
            padding-right: 0.65rem;
        }}
        .status-bar div:last-child {{ border-right: 0; }}
        .status-bar span {{
            display: block;
            color: {PALETTE["muted"]};
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.22rem;
        }}
        .status-bar strong {{
            display: block;
            color: {PALETTE["text"]};
            font-size: 1.05rem;
            line-height: 1.2;
            overflow-wrap: anywhere;
        }}
        .status-note {{
            color: {PALETTE["muted"]};
            font-size: 0.9rem;
            margin: 0 0 0.9rem;
        }}
        div[data-testid="stMetric"] {{
            background: linear-gradient(180deg, rgba(15,27,41,0.95), rgba(10,18,29,0.95));
            border: 1px solid {PALETTE["border"]};
            padding: 1rem 1rem 0.9rem 1rem;
            border-radius: 18px;
        }}
        div[data-testid="stMetricLabel"] {{ color: {PALETTE["muted"]}; }}
        div[data-testid="stMetricValue"] {{ color: {PALETTE["text"]}; }}
        div[data-testid="stDataFrame"], div[data-testid="stJson"] {{
            border: 1px solid {PALETTE["border"]};
            border-radius: 18px;
            overflow: hidden;
        }}
        .stTabs [data-baseweb="tab-list"] {{
            gap: 0.7rem;
            flex-wrap: nowrap;
            overflow-x: auto;
        }}
        .stTabs [data-baseweb="tab"] {{
            border-radius: 999px;
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.06);
            padding: 0.45rem 0.9rem;
        }}
        .stTabs [aria-selected="true"] {{
            background: rgba(77,163,255,0.12);
            border-color: rgba(77,163,255,0.35);
        }}
        .stTabs [data-baseweb="tab-panel"] {{
            padding-top: 1rem;
            min-height: 0 !important;
        }}
        .stTabs [data-baseweb="tab-panel"][hidden] {{
            display: none !important;
            height: 0 !important;
            overflow: hidden !important;
        }}
        @media (max-width: 1000px) {{
            .hero-title {{ font-size: 2.4rem; }}
            .signal-grid {{ grid-template-columns: 1fr; }}
            .status-bar {{ grid-template-columns: 1fr; }}
            .status-bar div {{ border-right: 0; border-bottom: 1px solid rgba(255,255,255,0.07); padding-bottom: 0.5rem; }}
            .status-bar div:last-child {{ border-bottom: 0; }}
            .story-grid {{ grid-template-columns: 1fr; }}
            .zone-grid {{ grid-template-columns: 1fr; }}
        }}
        .hero-badge {{
            display: inline-block;
            background: rgba(75, 211, 138, 0.15);
            color: #81e6aa;
            border: 1px solid rgba(75, 211, 138, 0.4);
            padding: 0.25rem 0.75rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.5rem;
        }}
        .decision-summary-box {{
            background: linear-gradient(135deg, rgba(77,163,255,0.1), rgba(16,27,43,0.8));
            border: 1px solid rgba(77,163,255,0.3);
            border-radius: 12px;
            padding: 1.2rem;
            margin-top: 0.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 8px 30px rgba(0,0,0,0.2);
        }}
        .flash-highlight {{
            animation: flash 2s ease-out;
        }}
        @keyframes flash {{
            0% {{ box-shadow: 0 0 0 rgba(75,211,138,0); border-color: rgba(77,163,255,0.3); }}
            10% {{ box-shadow: 0 0 20px rgba(75,211,138,0.8); border-color: rgba(75,211,138,0.8); }}
            100% {{ box-shadow: 0 0 0 rgba(75,211,138,0); border-color: rgba(77,163,255,0.3); }}
        }}
        .llm-insight-box {{
            background: rgba(159,140,255,0.08);
            border-left: 4px solid #9f8cff;
            padding: 1rem;
            border-radius: 4px 12px 12px 4px;
            margin-bottom: 1rem;
        }}
        .realtime-badge {{
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.35rem 0.72rem;
            border-radius: 999px;
            background: rgba(75, 211, 138, 0.12);
            border: 1px solid rgba(75, 211, 138, 0.28);
            color: #8ae8b0;
            font-size: 0.76rem;
            font-weight: 700;
            margin-bottom: 0.65rem;
        }}
        .realtime-dot {{
            width: 9px;
            height: 9px;
            border-radius: 999px;
            background: #4bd38a;
            box-shadow: 0 0 16px rgba(75, 211, 138, 0.8);
            animation: realtimePulse 1.4s ease-in-out infinite;
        }}
        .focus-grid {{
            display: grid;
            grid-template-columns: minmax(0, 1.45fr) minmax(280px, 0.9fr);
            gap: 1rem;
            margin: 0.75rem 0 1.2rem;
        }}
        .focus-card {{
            background: linear-gradient(135deg, rgba(18,31,47,0.98), rgba(9,18,29,0.98));
            border: 1px solid rgba(77,163,255,0.28);
            border-radius: 22px;
            padding: 1.25rem 1.25rem 1.1rem;
            box-shadow: 0 22px 54px rgba(0,0,0,0.24);
        }}
        .focus-card.accent {{
            border-color: rgba(75,211,138,0.26);
            background: linear-gradient(135deg, rgba(14,35,29,0.96), rgba(9,18,29,0.98));
        }}
        .focus-kicker {{
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            color: {PALETTE["muted"]};
            margin-bottom: 0.45rem;
        }}
        .focus-title {{
            color: {PALETTE["text"]};
            font-size: 2.7rem;
            line-height: 1;
            font-weight: 800;
            margin-bottom: 0.45rem;
        }}
        .focus-route {{
            color: #9ed4ff;
            font-size: 1.1rem;
            font-weight: 700;
            margin-bottom: 0.75rem;
        }}
        .focus-reason {{
            color: #dce8f8;
            font-size: 1rem;
            line-height: 1.6;
        }}
        .focus-impact {{
            color: #f1f6ff;
            font-size: 1.05rem;
            line-height: 1.55;
        }}
        .focus-stat-grid {{
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.7rem;
            margin-top: 1rem;
        }}
        .focus-stat {{
            padding: 0.75rem 0.8rem;
            border-radius: 16px;
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.06);
        }}
        .focus-stat span {{
            display: block;
            color: {PALETTE["muted"]};
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.24rem;
        }}
        .focus-stat strong {{
            color: {PALETTE["text"]};
            font-size: 1.35rem;
            line-height: 1.1;
        }}
        .quota-panel {{
            padding: 0.9rem 1rem;
            border-radius: 16px;
            border: 1px solid rgba(255,184,77,0.28);
            background: rgba(255,184,77,0.08);
            margin: 0.4rem 0 1rem;
        }}
        .quota-panel strong {{
            color: #ffd08a;
            display: block;
            margin-bottom: 0.22rem;
        }}
        .quota-panel span {{
            color: #d8e2ef;
            font-size: 0.92rem;
            line-height: 1.55;
        }}
        .llm-detail-row {{
            padding: 0.78rem 0.95rem;
            border-radius: 12px;
            border: 1px solid rgba(138,216,255,0.16);
            background: rgba(255,255,255,0.035);
            color: #d8e2ef;
            line-height: 1.55;
            white-space: normal;
            overflow-wrap: anywhere;
            margin: 0.45rem 0;
        }}
        .llm-detail-row strong {{
            color: #ffffff;
        }}
        .slot-board {{
            display: grid;
            grid-template-columns: minmax(260px, 0.82fr) minmax(0, 1.18fr);
            gap: 1rem;
            margin: 0.6rem 0 1rem;
        }}
        .slot-selector-card,
        .slot-detail-card {{
            background: linear-gradient(180deg, rgba(15,27,41,0.95), rgba(10,18,29,0.95));
            border: 1px solid {PALETTE["border"]};
            border-radius: 20px;
            padding: 1rem;
        }}
        .slot-selector-grid {{
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.65rem;
            margin-top: 0.85rem;
        }}
        .slot-timestamp {{
            color: {PALETTE["muted"]};
            font-size: 0.84rem;
            margin-top: 0.18rem;
        }}
        .slot-grid-dash {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(42px, 1fr));
            gap: 0.45rem;
            margin-top: 0.95rem;
        }}
        .slot-cell {{
            min-height: 42px;
            border-radius: 10px;
            border: 1px solid rgba(255,255,255,0.08);
            background: rgba(255,255,255,0.03);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 0.12rem;
            font-size: 0.9rem;
        }}
        .slot-cell.filled-car {{
            background: rgba(127, 217, 255, 0.12);
            border-color: rgba(127, 217, 255, 0.35);
        }}
        .slot-cell.filled-bike {{
            background: rgba(255, 182, 110, 0.12);
            border-color: rgba(255, 182, 110, 0.35);
        }}
        .slot-cell.empty {{
            color: #6d829e;
        }}
        .slot-cell small {{
            color: inherit;
            font-size: 0.62rem;
        }}
        .slot-legend {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.8rem;
            margin-top: 0.85rem;
            color: {PALETTE["muted"]};
            font-size: 0.8rem;
        }}
        .slot-legend span {{
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
        }}
        .slot-swatch {{
            width: 11px;
            height: 11px;
            border-radius: 999px;
            display: inline-block;
        }}
        .slot-swatch.car {{ background: #7fd9ff; }}
        .slot-swatch.bike {{ background: #ffb66e; }}
        .slot-swatch.free {{ background: rgba(255,255,255,0.12); }}
        .learning-banner {{
            padding: 0.9rem 1rem;
            border-radius: 16px;
            border: 1px solid rgba(75,211,138,0.22);
            background: rgba(75,211,138,0.08);
            margin: 0.45rem 0 1rem;
        }}
        .learning-banner strong {{
            display: block;
            color: #8ee6b0;
            margin-bottom: 0.24rem;
        }}
        .learning-banner span {{
            color: #d8e2ef;
            font-size: 0.93rem;
            line-height: 1.5;
        }}
        .metric-card {{
            min-height: 138px;
            border-radius: 18px;
            padding: 1rem;
            border: 1px solid rgba(255,255,255,0.08);
            background: linear-gradient(180deg, rgba(15,27,41,0.96), rgba(8,16,27,0.96));
            box-shadow: 0 16px 42px rgba(0,0,0,0.2);
        }}
        .metric-card span,
        .metric-card small {{
            display: block;
            color: {PALETTE["muted"]};
            font-size: 0.74rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
        }}
        .metric-card strong {{
            display: block;
            color: {PALETTE["text"]};
            font-size: 1.65rem;
            line-height: 1.08;
            margin: 0.38rem 0;
            overflow-wrap: anywhere;
        }}
        .metric-card.danger-card {{
            border-color: rgba(255,116,108,0.32);
            background: linear-gradient(180deg, rgba(74,21,28,0.34), rgba(8,16,27,0.96));
            box-shadow: 0 0 28px rgba(255,116,108,0.12);
        }}
        .metric-card.success-card {{
            border-color: rgba(75,211,138,0.32);
            background: linear-gradient(180deg, rgba(19,62,43,0.34), rgba(8,16,27,0.96));
            box-shadow: 0 0 28px rgba(75,211,138,0.12);
        }}
        .metric-card.route-card {{
            border-color: rgba(77,163,255,0.36);
            background:
                repeating-linear-gradient(90deg, transparent 0 24px, rgba(255,255,255,0.08) 24px 36px, transparent 36px 56px),
                linear-gradient(180deg, rgba(18,47,78,0.48), rgba(8,16,27,0.96));
            position: relative;
            overflow: hidden;
        }}
        .metric-card.route-card::after {{
            content: "";
            position: absolute;
            left: -22%;
            right: auto;
            top: 72%;
            width: 36px;
            height: 10px;
            border-radius: 999px;
            background: #8ad8ff;
            box-shadow: 0 0 16px rgba(138,216,255,0.9);
            animation: dashboardRouteMove 1.2s ease-in-out infinite;
        }}
        @keyframes dashboardRouteMove {{
            0% {{ left: -20%; opacity: 0; }}
            15% {{ opacity: 1; }}
            85% {{ opacity: 1; }}
            100% {{ left: 105%; opacity: 0; }}
        }}
        @keyframes realtimePulse {{
            0%, 100% {{ opacity: 0.55; transform: scale(0.95); }}
            50% {{ opacity: 1; transform: scale(1.05); }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
