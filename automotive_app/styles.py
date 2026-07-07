import streamlit as st


def apply_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --surface: #ffffff;
            --ink: #111827;
            --muted: #64748b;
            --line: #e5e7eb;
            --accent: #0f766e;
            --accent-2: #2563eb;
        }

        [data-testid="stAppViewContainer"] {
            background: #f6f7fb;
        }

        [data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid var(--line);
        }

        [data-testid="stHeader"] {
            background: rgba(246, 247, 251, 0.88);
            backdrop-filter: blur(10px);
        }

        .block-container {
            max-width: 1240px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        h1, h2, h3 {
            letter-spacing: 0;
            color: var(--ink);
        }

        .hero {
            padding: 1.3rem 1.4rem;
            border: 1px solid var(--line);
            border-radius: 8px;
            background: linear-gradient(135deg, #ffffff 0%, #eef7f6 52%, #eef2ff 100%);
            margin-bottom: 1.1rem;
        }

        .hero h1 {
            margin: 0;
            font-size: clamp(2rem, 4vw, 3.4rem);
            line-height: 1.05;
        }

        .hero p {
            margin: .55rem 0 0 0;
            color: var(--muted);
            font-size: 1.02rem;
        }

        [data-testid="stMetric"] {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: .9rem 1rem;
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid var(--line);
            border-radius: 8px;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: .35rem;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            padding: .45rem .75rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

