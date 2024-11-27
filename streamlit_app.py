
# Daten laden und verarbeiten
import pandas as pd
import numpy as np

# Für die Diagramme
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import seaborn as sns  # Wird hier und da für Visualisierungen verwendet

# Für die Streamlit-App
import streamlit as st

import matplotlib.ticker as mticker

# Machine Learning und spezifische Aufgaben
from matplotlib.ticker import FuncFormatter  # Für benutzerdefinierte Achseneinstellungen
from sklearn.ensemble import RandomForestRegressor  # Für Preisvorhersage
from sklearn.preprocessing import LabelEncoder  # Für Kategorien in Zahlen umwandeln
from sklearn.model_selection import train_test_split  # Datenaufteilung in Trainings- und Testset
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error  # Fehlerberechnung


###########################################################################################################


# Titel-Design: Groß, zentriert und unterstrichen
# Farben werden mit HEX-Codes definiert
st.markdown(
    """
    <h1 style='text-align: center; font-size: 60px; text-decoration: underline; color: #FFFFFF;'>
        Automotive Insights
    </h1>
    """,
    unsafe_allow_html=True  # Ermöglicht HTML im Markdown
)

# CSS für das Seiten- und Sidebar-Layout
# Hier ein dezenter dunkelblauer Farbverlauf und eine violette Sidebar

page_bg_css = """
<style>
/* Hintergrund für die Hauptseite: Dunkelblauer Verlauf und ein dezentes Muster */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(to bottom right, #283289, #0f1a44), 
                url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='20' height='20'><rect width='10' height='10' fill='%23c0c0c0' opacity='0.2'/><rect x='10' y='10' width='10' height='10' fill='%23c0c0c0' opacity='0.2'/></svg>");
    background-size: cover, 40px 40px; /* Hintergrundgröße und Muster wiederholen */
    background-position: center;      /* Muster wird zentriert */
    background-repeat: no-repeat, repeat; /* Verlauf fixiert, Muster wiederholt */
}

/* Sidebar: Ein sattes Violett für einen klaren Kontrast */
[data-testid="stSidebar"] {
    background-color: #26112b;
}

/* Header transparent: Der obere Bereich wird nahtlos integriert */
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}

/* Toolbar-Position leicht anpassen */
[data-testid="stToolbar"] {
    right: 2rem;
}
</style>
"""


###########################################################################################################


# HTML Formatierung anwenden (Hintergrund und Sidebar)
st.markdown(page_bg_css, unsafe_allow_html=True)

# Banner anzeigen: Automotive Insights, Python und Neu-/Gebrauchtwagenanalyse
st.image("banner1.jpg", use_column_width=True)

# Sidebar-Inhaltsverzeichnis
st.sidebar.title("Inhaltsverzeichnis")

# Neuzulassungen
st.sidebar.markdown("## 🚗 Neuwagenzulassungen")
st.sidebar.markdown("[🔍 Neuwagenzulassungen der Länder in Europa](#neuzulassungen-eines-spezifischen-landes-anzeigen)")
st.sidebar.markdown("[🌍 Neuwagenzulassungen für ganz Europa nach Antriebsart](#neuzulassungen-in-europa-anzeigen)")

# Durchschnittsverbrauch
st.sidebar.markdown("## ⛽ Durchschnittsverbrauch Liter pro 100km")
st.sidebar.markdown("[📅 Durchschnittsverbrauch über die Jahre](#durchschnittsverbrauch-nach-jahr)")

# Kraftstoffarten
st.sidebar.markdown("## 🛢️ Kraftstoffarten")
st.sidebar.markdown("[📊 PKW-Bestand nach Kraftstoffart](#bestand-an-pkw-nach-kraftstoffart-und-jahr)")

# Ladesäulen
st.sidebar.markdown("## ⚡ Ladesäulen")
st.sidebar.markdown("[🔋 Ladesäulen-Entwicklung und Prognose](#ladesäulen-entwicklung-und-prognose-nach-bundesland)")

# Gebrauchtwagen Analyse
st.sidebar.markdown("## 🚙 Gebrauchtwagen Analyse")
st.sidebar.markdown("[🔍 Gebrauchtwagenanalyse nach Marke und Modell](#gebrauchtwagen-analyse-nach-marke-und-modell)")

# Erweiterte Gebrauchtwagen Analyse
st.sidebar.markdown("## 📈 Erweiterte Gebrauchtwagen Analyse")
st.sidebar.markdown("[🔧 Datenvisualisierung nach Auswahl](#datenvisualisierung-nach-auswahl)")

# Elektro- und Hybridfahrzeuge Analyse
st.sidebar.markdown("## 🌱 Elektro- und Hybridfahrzeuge")
st.sidebar.markdown("[🏆 Top-10 Modelle für Elektro- und Hybridfahrzeuge](#top-10-modelle-fuer-elektro-und-hybridfahrzeuge)")
st.sidebar.markdown("[📋 Detaillierte Informationen zu Elektro- oder Hybridmodellen](#detaillierte-informationen-zu-elektro-oder-hybridmodellen)")

# Machine Learning für Preisvorhersage
st.sidebar.markdown("## 🤖 Machine Learning für Preisvorhersage")
st.sidebar.markdown("[🔮 Preisvorhersage nach Fahrzeugmerkmalen](#preisvorhersage-nach-fahrzeugmerkmalen)")

# Quellenangaben
st.sidebar.markdown("## 📚 Quellenverzeichnis")
st.sidebar.markdown("[🔗 Quellen](#quellen)", unsafe_allow_html=True)


###########################################################################################################


# Neuzulassungen

# Cachen der Daten für effizientes Laden
@st.cache_data
def load_neuzulassung_data():
    # Daten laden
    data = pd.read_csv('neuzulassung.csv', delimiter=';')

    # Europa-Spalte berechnen
    data['Europe'] = data.loc[:, 'Belgium':'Kosovo*'] \
        .apply(lambda x: pd.to_numeric(x.astype(str).str.replace('.', ''), errors='coerce')).sum(axis=1)
    return data

# Daten laden (gecached)
data_neuzulassung = load_neuzulassung_data()

# Custom y-Achsenformatierung
def custom_formatter(x, pos):
    return f'{int(x / 1_000_000)} Mio' if x >= 1_000_000 else f'{int(x)}'

# Streamlit Titel und Bild
st.title('Neuwagenzulassungen der Länder in Europa')
st.image("neuwagen.jpg", caption="Neuwagen (AI)", use_column_width=True)

# Länder- und Antriebsartenliste
laender = list(data_neuzulassung.columns[2:-1])  # Ignoriert 'Antrieb', 'Jahr' und 'Europe'
antriebe = data_neuzulassung['Antrieb'].unique()

# Abschnitt: Neuzulassungen zwischen 2013 und 2023
with st.expander("Neuwagenzulassungen in Europa (2013–2023)", expanded=False):
    st.subheader("Neuwagenzulassungen zwischen 2013 und 2023")

    # Land auswählen, Standard: Germany
    selected_country = st.selectbox('Wähle ein Land aus', laender, index=laender.index("Germany"))

    # Daten filtern basierend auf dem ausgewählten Land
    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=data_neuzulassung, 
        x="Jahr", 
        y=pd.to_numeric(data_neuzulassung[selected_country].astype(str).str.replace('.', ''), errors='coerce').fillna(0),
        hue="Antrieb", 
        marker="o"
    )
    plt.title(f"Neuwagenzulassungen zwischen 2013 und 2023")
    plt.xlabel("Jahr")
    plt.ylabel("Anzahl der Neuzulassungen")
    plt.grid(True)
    plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(custom_formatter))
    st.pyplot(plt)

# Abschnitt: Neuwagenzulassungen nach Kraftstoffart
with st.expander("Neuwagenzulassungen nach Kraftstoffart (2013–2023)", expanded=False):
    st.subheader("Neuwagenzulassungen Gesamt Europa zwischen 2013 und 2023 nach Kraftstoffart")

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=data_neuzulassung, x="Jahr", y="Europe", hue="Antrieb", style="Antrieb", markers=True)
    plt.title("Neuwagenzulassungen Gesamt Europa zwischen 2013 und 2023 nach Kraftstoffart")
    plt.xlabel("Jahr")
    plt.ylabel("Anzahl der Neuzulassungen")
    plt.grid(True)
    plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(custom_formatter))
    st.pyplot(plt)


###########################################################################################################



# Durchschnittlicher Verbrauch inkusive Effizienz

# Cachen der Daten für effizientes Laden
@st.cache_data
def load_durchschnitt_verbrauch_data():
    # Daten laden
    data = pd.read_csv('durchschnitt_verbrauch.csv', sep=';')
    
    # Konvertiere Verbrauchsspalten zu numerischen Werten
    for column in data.columns[1:]:
        data[column] = pd.to_numeric(
            data[column].astype(str).str.replace(',', '.'), errors='coerce'
        )
    return data

# Daten laden (gecached)
data_durchschnitt_verbrauch = load_durchschnitt_verbrauch_data()

# Streamlit Titel
st.title('Durchschnittsverbrauch über die Jahre')

# Funktion zur Berechnung der Effizienz
def calculate_efficiency(data, column):
    start_value = data[column].iloc[0]
    end_value = data[column].iloc[-1]
    reduction_percent = ((start_value - end_value) / start_value) * 100 if start_value != 0 else 0
    return reduction_percent

# Abschnitt mit Checkboxen und Effizienzberechnung
st.markdown("<a id='verbrauch-checkbox-effizienz'></a>", unsafe_allow_html=True)
with st.expander("Benzin und Diesel Verbrauch", expanded=False):
    # Checkboxen für Benzin und Diesel
    show_benzin = st.checkbox("Benzin anzeigen", value=True)
    show_diesel = st.checkbox("Diesel anzeigen", value=True)

    # Effizienzberechnung für Benzin und Diesel
    if show_benzin:
        benzin_efficiency = calculate_efficiency(data_durchschnitt_verbrauch, "Benzin")
        st.metric(label="Effizienz Benzin", value=f"{benzin_efficiency:.2f}%")
    
    if show_diesel:
        diesel_efficiency = calculate_efficiency(data_durchschnitt_verbrauch, "Diesel")
        st.metric(label="Effizienz Diesel", value=f"{diesel_efficiency:.2f}%")

    # Plot basierend auf Checkboxen
    plt.figure(figsize=(12, 6))
    if show_benzin:
        sns.lineplot(data=data_durchschnitt_verbrauch, x="Jahr", y="Benzin", label="Benzin")
    if show_diesel:
        sns.lineplot(data=data_durchschnitt_verbrauch, x="Jahr", y="Diesel", label="Diesel")
    
    plt.title("Verbrauch für Benzin und Diesel auf 100km")
    plt.xlabel("Jahr")
    plt.ylabel("Verbrauch")
    plt.legend(title="Kraftstofftyp")
    plt.grid(True)

    # Plot anzeigen
    st.pyplot(plt)



###########################################################################################################

# Bestandliste an Fahrzeugen nach Kraftstoffarten

# Cachen der Daten für effizientes Laden
@st.cache_data
def load_kraftstoff_data():
    return pd.read_csv('pkw_bestand_kraftstoffart_neu.csv', delimiter=';')

# Daten laden (gecached)
data_kraftstoff = load_kraftstoff_data()

# Titel und Bild
st.title('Bestand an PKW nach Kraftstoffart und Jahr')
st.image("refuel.jpg", caption="Tankstellen", use_column_width=True)

# Plot-Funktion für verschiedene Diagrammtypen
def plot_kraftstoff_data(data, jahre, antriebe, selected_antriebe, diagramm_typ):
    plt.figure(figsize=(12, 8))
    
    if diagramm_typ == "Liniendiagramm":
        for antrieb in selected_antriebe:
            sns.lineplot(x=jahre, y=data[antrieb], label=antrieb)
        plt.title('PKW-Bestand nach Kraftstoffart über die Jahre')
        plt.xlabel('Jahr')
        plt.ylabel('Bestand')
        plt.legend(title='Antriebsarten')
        plt.grid(True)
        
        # Custom y-Achse für bessere Lesbarkeit
        def custom_mio_formatter(x, pos):
            return f'{int(x / 1_000_000)} Mio' if x >= 1_000_000 else str(int(x))
        plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(custom_mio_formatter))
    
    elif diagramm_typ == "Kuchendiagramm":
        latest_data = data[selected_antriebe].iloc[-1]
        plt.pie(latest_data, labels=selected_antriebe, autopct='%1.1f%%', startangle=90)
        plt.title(f"Anteile der Antriebsarten im Jahr {data['Jahr'].iloc[-1]}")

    elif diagramm_typ == "Balkendiagramm":
        latest_data = data[selected_antriebe].iloc[-1]
        plt.bar(selected_antriebe, latest_data)
        plt.title(f"Bestand der Antriebsarten im Jahr {data['Jahr'].iloc[-1]}")
        plt.xlabel('Antriebsart')
        plt.ylabel('Bestand')
        
        # Custom y-Achse für bessere Lesbarkeit
        def custom_mio_formatter(x, pos):
            return f'{int(x / 1_000_000)} Mio' if x >= 1_000_000 else str(int(x))
        plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(custom_mio_formatter))
    
    st.pyplot(plt)

# Abschnitt für Visualisierung
st.markdown("<a id='bestand-an-pkw-nach-kraftstoffart-und-jahr'></a>", unsafe_allow_html=True)
with st.expander("Datenvisualisierung", expanded=False):
    jahre = data_kraftstoff['Jahr']
    antriebe = data_kraftstoff.columns[1:]  # Ignoriert erste Spalte ('Jahr')

    # Deaktivierter Punkt: "PKW gesamt"
    antriebe = [antrieb for antrieb in antriebe if antrieb != "PKW gesamt"]

    # Vorauswahl: Benzin, Diesel und Elektro (falls vorhanden)
    default_antriebe = [antrieb for antrieb in ["Benzin", "Diesel", "Elektro (BEV)"] if antrieb in antriebe]
    
    # Multiselect für Antriebsarten
    selected_antriebe = st.multiselect(
        'Wähle die Antriebsarten aus',
        options=antriebe,
        default=default_antriebe
    )

    # Auswahl des Diagrammtyps
    diagramm_typ = st.radio("Diagrammtyp auswählen", ["Liniendiagramm", "Kuchendiagramm", "Balkendiagramm"])

    # Diagramm erstellen, wenn Antriebsarten ausgewählt sind
    if selected_antriebe:
        plot_kraftstoff_data(data_kraftstoff, jahre, antriebe, selected_antriebe, diagramm_typ)
    else:
        st.write("Bitte wähle mindestens eine Antriebsart zur Anzeige aus.")

        

###########################################################################################################

# Entwicklung der Ladesäulen über Jahre
  
# Cachen der Daten für effizientes Laden
@st.cache_data
def load_ladesaeulen_data():
    return pd.read_csv('ladesaeulen.csv')

# Funktion: Machine Learning-Modell trainieren und Vorhersage erstellen
def predict_future(data, column, start_year, end_year):
    # Jahre und Werte für Training
    years = data['Jahr'].values.reshape(-1, 1)
    values = data[column].values

    # Modell trainieren
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(years, values)

    # Vorhersage für die Zukunft
    future_years = np.arange(start_year, end_year + 1).reshape(-1, 1)
    future_predictions = model.predict(future_years)

    # R²-Score berechnen
    y_pred_train = model.predict(years)
    r2 = r2_score(values, y_pred_train)

    return future_years, future_predictions, r2

# Funktion: Plot der Daten und Prognosen
def plot_ladesaeulen(data, columns, selected_column, plot_all, future_years=None, future_predictions=None):
    plt.figure(figsize=(18, 12))
    
    if plot_all:
        for column in columns:
            sns.lineplot(data=data, x='Jahr', y=column, label=column)
        start_avg = data[columns].iloc[0].mean()
        end_avg = data[columns].iloc[-1].mean()
        reduction_percent = ((end_avg - start_avg) / start_avg) * 100
    else:
        sns.lineplot(data=data, x='Jahr', y=selected_column, label=selected_column)
        
        # Prognose plotten, falls verfügbar
        if future_years is not None and future_predictions is not None:
            plt.plot(future_years, future_predictions, '--', label=f'{selected_column} (Prognose)')
    
    # Custom y-Achse für bessere Lesbarkeit
    def custom_mio_formatter(x, pos):
        return f'{int(x / 1_000_000)} Mio' if x >= 1_000_000 else str(int(x))
    plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(custom_mio_formatter))

    # Grafiktitel und Achsenbeschriftungen
    plt.title('Ladesäulen-Entwicklung und Prognose nach Jahr')
    plt.xlabel('Jahr')
    plt.ylabel('Anzahl Ladesäulen')
    plt.legend(title='Bundesland')
    plt.grid(True)

    st.pyplot(plt)

# Daten laden (gecached)
data_ladesaeulen = load_ladesaeulen_data()

# Titel und Bild
st.title('Ladesäulen-Entwicklung und Prognose nach Bundesland')
st.image("Deutschlandkarte1.jpg", caption="Ladeinfrastruktur Deutschland", use_column_width=True)

# Visualisierung und Prognose
st.markdown("<a id='ladesaeulen-entwicklung-und-prognose-nach-bundesland'></a>", unsafe_allow_html=True)
with st.expander("Datenvisualisierung", expanded=False):
    columns = list(data_ladesaeulen.columns[1:])  # Ignoriert 'Jahr'
    
    # Session-State initialisieren
    if 'selected_column' not in st.session_state:
        st.session_state.selected_column = columns

    # Button: Alle Spalten auswählen/abwählen
    if st.button("Alle Spalten auswählen / abwählen", key="toggle_all_columns"):
        if len(st.session_state.selected_column) == len(columns):
            st.session_state.selected_column = []
        else:
            st.session_state.selected_column = columns

    # Selectbox und Checkbox
    selected_column = st.selectbox('Wähle eine Spalte zum Plotten aus', columns)
    plot_all = st.checkbox('Alle Spalten plotten', value=len(st.session_state.selected_column) == len(columns), key="plot_all_columns")

    # Prognose nur für eine Spalte
    future_years, future_predictions, r2 = None, None, None
    if not plot_all:
        future_years, future_predictions, r2 = predict_future(data_ladesaeulen, selected_column, 2024, 2035)

    # Diagramm erstellen
    plot_ladesaeulen(data_ladesaeulen, columns, selected_column, plot_all, future_years, future_predictions)

    # Anzeige von R²-Score und Wachstumsraten
    if not plot_all:
        st.metric(label="R²-Score des Modells", value=f"{r2:.2f}")

###########################################################################################################


# Gebrauchtwagenanalyse mit Machine Learning

# Cachen der Gebrauchtwagendaten
@st.cache_data
def load_gebrauchtwagen_data():
    return pd.read_csv('gebrauchtwagen.csv')

# Funktion: Balkendiagramm erstellen
def plot_gebrauchtwagen_models(data, marke, top_n=15):
    # Filtert Daten nach Marke
    data_filtered = data[data['brand'] == marke]
    
    # Berechnet die Anzahl der Modelle und wählt die Top-N aus
    model_counts = data_filtered['model'].value_counts().reset_index().head(top_n)
    model_counts.columns = ['model', 'count']

    # Balkendiagramm
    plt.figure(figsize=(12, 8))
    sns.barplot(data=model_counts, x='model', y='count', palette='viridis')
    plt.title(f'Anzahl der Modelle für {marke} (Top {top_n})')
    plt.xlabel('Modell')
    plt.ylabel('Anzahl der Modelle')
    plt.xticks(rotation=45)
    plt.grid(True)

    st.pyplot(plt)

# Daten laden (gecached)
data_gebrauchtwagen = load_gebrauchtwagen_data()

# Streamlit-Titel
st.markdown("<a id='gebrauchtwagenanalyse-nach-marke-und-modell'></a>", unsafe_allow_html=True)
st.title('Gebrauchtwagen Analyse nach Marke und Modell')

# Datenvisualisierung
with st.expander("Datenvisualisierung", expanded=False):
    # Marken für die Auswahlbox
    marken = data_gebrauchtwagen['brand'].unique()
    selected_marke = st.selectbox("Wähle eine Marke aus", marken)

    # Eingabefeld für Anzahl der Top-Modelle
    top_n = st.number_input("Anzahl der Top-Modelle anzeigen", min_value=1, max_value=50, value=15, step=1)

    # Diagramm erstellen
    plot_gebrauchtwagen_models(data_gebrauchtwagen, selected_marke, top_n)


##############################


# Cachen der Datenvorbereitung
@st.cache_data
def prepare_gebrauchtwagen_data(data):
    data['price_in_euro'] = pd.to_numeric(data['price_in_euro'], errors='coerce')
    data['mileage_in_km'] = pd.to_numeric(data['mileage_in_km'], errors='coerce')
    data['power_ps'] = pd.to_numeric(data['power_ps'], errors='coerce')
    return data

# Dynamische Farbpallette basierend auf den Farbwerten
def dynamic_color_palette(data, column):
    unique_colors = data[column].unique()
    return {color: color for color in unique_colors if color.startswith('#')}

# Funktion: Diagramme erstellen
def plot_histogram(data, column, title, xlabel, ylabel, color='blue', kde=True):
    plt.figure(figsize=(10, 6))
    sns.histplot(data[column], kde=kde, color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    st.pyplot(plt)

def plot_countplot(data, column, title, xlabel, ylabel, palette=None):
    plt.figure(figsize=(10, 6))
    sns.countplot(data=data, x=column, palette=palette)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    st.pyplot(plt)


##############################


# Funktion: Farbverlauf erstellen (Grün -> Rot)
def create_color_gradient(data, column, cmap_name="RdYlGn_r"):
    # Normalisieren der Werte (für den Farbverlauf)
    normalized_values = (data[column] - data[column].min()) / (data[column].max() - data[column].min())
    cmap = cm.get_cmap(cmap_name)
    return [cmap(value) for value in normalized_values]

# Funktion: Histogramm mit Farbverlauf
def plot_histogram(data, column, title, xlabel, ylabel, gradient=False, kde=True):
    plt.figure(figsize=(10, 6))
    if gradient:
        # Berechnung des Farbverlaufs
        colors = create_color_gradient(data, column)
        values, bins, bars = plt.hist(data[column], bins=30, color="grey", edgecolor="black", alpha=0.7)
        # Wende Farbverlauf an
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        norm = plt.Normalize(bin_centers.min(), bin_centers.max())
        cmap = cm.get_cmap("RdYlGn_r")
        for bar, value in zip(bars, bin_centers):
            bar.set_facecolor(cmap(norm(value)))
    else:
        sns.histplot(data[column], kde=kde, color="skyblue")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    st.pyplot(plt)

# Funktion: Countplot bleibt gleich
def plot_countplot(data, column, title, xlabel, ylabel, palette=None):
    plt.figure(figsize=(10, 6))
    sns.countplot(data=data, x=column, palette=palette)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    st.pyplot(plt)

# Daten vorbereiten (gecached)
@st.cache_data
def prepare_gebrauchtwagen_data(data):
    data['price_in_euro'] = pd.to_numeric(data['price_in_euro'], errors='coerce')
    data['mileage_in_km'] = pd.to_numeric(data['mileage_in_km'], errors='coerce')
    data['power_ps'] = pd.to_numeric(data['power_ps'], errors='coerce')
    return data

data_gebrauchtwagen = prepare_gebrauchtwagen_data(pd.read_csv('gebrauchtwagen.csv'))

# Streamlit-Titel
st.markdown("<a id='datenvisualisierung-nach-auswahl'></a>", unsafe_allow_html=True)
st.title('Erweiterte Gebrauchtwagen Analyse')

st.image("autoscout24logo.png", use_column_width=True)

# Visualisierung
with st.expander("Datenvisualisierung nach Auswahl", expanded=False):
    # Auswahl von Marke und Modell
    selected_brand = st.selectbox("Wähle eine Marke aus", data_gebrauchtwagen['brand'].unique(), key="brand_select")
    filtered_data_by_brand = data_gebrauchtwagen[data_gebrauchtwagen['brand'] == selected_brand]

    selected_model = st.selectbox("Wähle ein Modell aus", filtered_data_by_brand['model'].unique(), key="model_select")
    filtered_data = filtered_data_by_brand[filtered_data_by_brand['model'] == selected_model]

    # Kilometerstände mit Farbverlauf
    plot_histogram(filtered_data, 'mileage_in_km', f'Kilometerstände für {selected_model}',
                   'Kilometerstand', 'Anzahl', gradient=True)

    # Farben-Diagramm mit dynamischer Palette
    def dynamic_color_palette(data, column):
        unique_colors = data[column].unique()
        predefined_colors = {
            "red": "red",
            "silver": "silver",
            "black": "black",
            "grey": "grey",
            "beige": "beige",
            "blue": "blue",
            "green": "green",
            "brown": "brown",
            "yellow": "yellow",
            "gold": "gold",
            "orange": "orange"
        }
        # Für Hex-Codes oder Standardfarben, alles andere wird zu "grey"
        return {color: predefined_colors.get(color, "grey") if not str(color).startswith('#') else color for color in unique_colors}

    color_palette = dynamic_color_palette(filtered_data, 'color')
    plot_countplot(filtered_data, 'color', f'Farben für {selected_model}', 'Farbe', 'Anzahl', palette=color_palette)

    # Kraftstoffarten
    plot_countplot(filtered_data, 'fuel_type', f'Kraftstoffarten für {selected_model}',
                   'Kraftstoff', 'Anzahl', palette='husl')

    # Getriebetypen
    plot_countplot(filtered_data, 'transmission_type', f'Getriebetypen für {selected_model}',
                   'Getriebe', 'Anzahl', palette='Set1')

    # Preise mit Farbverlauf
    plot_histogram(filtered_data, 'price_in_euro', f'Preise für {selected_model}',
                   'Preis (€)', 'Anzahl', gradient=True)

##############################


# Cachen der Datenvorbereitung
@st.cache_data
def prepare_electric_hybrid_data(data):
    data['price_in_euro'] = pd.to_numeric(data['price_in_euro'], errors='coerce')
    data['mileage_in_km'] = pd.to_numeric(data['mileage_in_km'], errors='coerce')
    data['power_ps'] = pd.to_numeric(data['power_ps'], errors='coerce')
    data['fuel_consumption_g_km'] = data['fuel_consumption_g_km'].str.extract(r'(\d+)').astype(float)
    # Filtert nur Elektro- und Hybridfahrzeuge
    return data[data['fuel_type'].isin(['Electric', 'Hybrid'])]

# Funktion: Top-N Modelle plotten
def plot_top_models(data, fuel_type, brand, top_n=10, palette='Blues_r'):
    models = data[data['fuel_type'] == fuel_type]['model'].value_counts().head(top_n)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=models.values, y=models.index, palette=palette)
    plt.title(f'Anzahl der {fuel_type.lower()} Modelle für {brand}')
    plt.xlabel('Anzahl')
    plt.ylabel(f'{fuel_type} Modelle')
    st.pyplot(plt)


##############################



# Cachen der Datenvorbereitung
@st.cache_data
def prepare_electric_hybrid_data(data):
    data['price_in_euro'] = pd.to_numeric(data['price_in_euro'], errors='coerce')
    data['mileage_in_km'] = pd.to_numeric(data['mileage_in_km'], errors='coerce')
    data['power_ps'] = pd.to_numeric(data['power_ps'], errors='coerce')
    data['fuel_consumption_g_km'] = data['fuel_consumption_g_km'].str.extract(r'(\d+)').astype(float)
    # Filtert nur Elektro- und Hybridfahrzeuge
    return data[data['fuel_type'].isin(['Electric', 'Hybrid'])]

# Funktion: Farbverlauf für Balken mit Überprüfung
def create_color_gradient_for_bars(values, cmap_name="RdYlGn_r"):
    if len(values) == 0:  # Überprüfen, ob Werte vorhanden sind
        return ["grey"]  # Standardfarbe für leere Werte
    norm = mcolors.Normalize(vmin=min(values), vmax=max(values))
    cmap = cm.get_cmap(cmap_name)
    return [mcolors.to_hex(cmap(norm(value))) for value in values]

# Funktion: Top-N Modelle plotten mit Fehlerbehandlung
def plot_top_models(data, fuel_type, brand, top_n=10, cmap_name="RdYlGn_r"):
    models = data[data['fuel_type'] == fuel_type]['model'].value_counts().head(top_n)
    if models.empty:  # Überprüfen, ob Daten vorhanden sind
        st.write(f"Keine {fuel_type.lower()}-Modelle für {brand} verfügbar.")
        return
    colors = create_color_gradient_for_bars(models.values, cmap_name)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=models.values, y=models.index, palette=colors)
    plt.title(f'Anzahl der {fuel_type.lower()} Modelle für {brand}')
    plt.xlabel('Anzahl')
    plt.ylabel(f'{fuel_type} Modelle')
    st.pyplot(plt)

# Dynamische Farbpalette für Farben
def dynamic_color_palette(data, column):
    unique_colors = data[column].unique()
    predefined_colors = {
        "red": "red",
        "silver": "silver",
        "black": "black",
        "grey": "grey",
        "beige": "beige",
        "blue": "blue",
        "green": "green",
        "brown": "brown",
        "yellow": "yellow",
        "gold": "gold",
        "orange": "orange"
    }
    return {color: predefined_colors.get(color, "grey") if not str(color).startswith('#') else color for color in unique_colors}

# Funktion: Histogramme mit Farbverlauf und optionalem color-Argument
def plot_histogram(data, column, title, xlabel, ylabel, gradient=False, cmap_name="RdYlGn_r", kde=True, color=None):
    if data[column].dropna().empty:  # Überprüfen, ob Werte vorhanden sind
        st.write(f"Keine Daten für {column} verfügbar.")
        return
    plt.figure(figsize=(10, 6))
    if gradient:
        colors = create_color_gradient_for_bars(data[column].dropna(), cmap_name)
        sns.histplot(data[column].dropna(), kde=kde, color=colors[0])
    else:
        sns.histplot(data[column].dropna(), kde=kde, color=color if color else "skyblue")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    st.pyplot(plt)

# Funktion: Countplot für Farben mit Fehlerbehandlung
def plot_color_countplot(data, column, title, xlabel, ylabel):
    if data[column].dropna().empty:  # Überprüfen, ob Werte vorhanden sind
        st.write(f"Keine Daten für {column} verfügbar.")
        return
    palette = dynamic_color_palette(data, column)
    plt.figure(figsize=(10, 6))
    sns.countplot(data=data, x=column, palette=palette)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    st.pyplot(plt)

# Daten vorbereiten
data_electric_hybrid = prepare_electric_hybrid_data(pd.read_csv('gebrauchtwagen.csv'))

# Streamlit-Titel
st.title("Analyse für Elektro- und Hybridfahrzeuge")

# Top-10 Modelle Abschnitt
st.markdown("<a id='top-10-modelle-fuer-elektro-und-hybridfahrzeuge'></a>", unsafe_allow_html=True)
with st.expander("Top-10 Modelle für Elektro- und Hybridfahrzeuge", expanded=False):
    # Auswahl der Marke
    marken_electric_hybrid = data_electric_hybrid['brand'].unique()
    selected_brand = st.selectbox("Wähle eine Marke für Elektro- oder Hybridfahrzeuge aus", marken_electric_hybrid)

    # Filtert Daten basierend auf der ausgewählten Marke
    filtered_data = data_electric_hybrid[data_electric_hybrid['brand'] == selected_brand]

    # Anzahl der Top-Modelle anpassbar
    top_n = st.number_input("Anzahl der Top-Modelle anzeigen", min_value=1, max_value=50, value=10, step=1)

    # Diagramme für elektrische und hybride Modelle
    if not filtered_data.empty:
        st.subheader(f"Top-{top_n} elektrische Modelle für {selected_brand}")
        plot_top_models(filtered_data, 'Electric', selected_brand, top_n=top_n, cmap_name="Greens_r")

        st.subheader(f"Top-{top_n} Hybrid-Modelle für {selected_brand}")
        plot_top_models(filtered_data, 'Hybrid', selected_brand, top_n=top_n, cmap_name="Blues_r")
    else:
        st.write(f"Keine Daten für {selected_brand} verfügbar.")

# Detaillierte Informationen zu Modellen
st.markdown("<a id='detaillierte-informationen-zu-elektro-oder-hybridmodellen'></a>", unsafe_allow_html=True)
with st.expander("Detaillierte Informationen zu Elektro- oder Hybridmodellen", expanded=False):
    # Filtert Marken mit Elektro- oder Hybridmodellen
    brands_with_electric_or_hybrid = data_electric_hybrid['brand'].unique()
    selected_brand_details = st.selectbox("Wähle eine Marke für Detailinformationen", brands_with_electric_or_hybrid)

    # Filtert Modelle basierend auf der Marke
    models_with_electric_or_hybrid = data_electric_hybrid[data_electric_hybrid['brand'] == selected_brand_details]['model'].unique()
    selected_model = st.selectbox("Wähle ein elektrisches oder hybrides Modell aus", models_with_electric_or_hybrid)

    # Daten für das ausgewählte Modell
    model_data = data_electric_hybrid[
        (data_electric_hybrid['brand'] == selected_brand_details) &
        (data_electric_hybrid['model'] == selected_model)
    ]

    # Preise mit Farbverlauf
    st.subheader(f"Preise für {selected_model}")
    plot_histogram(model_data, 'price_in_euro', f'Preise für {selected_model}',
                   'Preis (€)', 'Anzahl', gradient=True, cmap_name="Oranges")

    # Kilometerstände mit Farbverlauf
    st.subheader(f"Kilometerstände für {selected_model}")
    plot_histogram(model_data, 'mileage_in_km', f'Kilometerstände für {selected_model}',
                   'Kilometerstand', 'Anzahl', gradient=True, cmap_name="Blues")

    # Farben mit dynamischer Palette
    st.subheader(f"Farben für {selected_model}")
    plot_color_countplot(model_data, 'color', f'Farben für {selected_model}',
                         'Farbe', 'Anzahl')

    # PS (Leistung)
    st.subheader(f"PS-Werte für {selected_model}")
    plot_histogram(model_data, 'power_ps', f'PS-Werte für {selected_model}',
                   'PS', 'Anzahl', color='purple')

    # Reichweite
    st.subheader(f"Reichweite für {selected_model}")
    plot_histogram(model_data, 'fuel_consumption_g_km', f'Reichweite für {selected_model}',
                   'Reichweite (km)', 'Anzahl', color='green')
    


##############################


# Machine Learning für die Preisbestimmung der Gebrauchtwagen


# Cachen der Datenvorbereitung und Modellerstellung
@st.cache_data
def prepare_ml_data(data):
    # Konvertieren relevanter Spalten
    data['price_in_euro'] = pd.to_numeric(data['price_in_euro'], errors='coerce')
    data['year'] = pd.to_numeric(data['year'], errors='coerce')
    data['mileage_in_km'] = pd.to_numeric(data['mileage_in_km'], errors='coerce')
    data['power_ps'] = pd.to_numeric(data['power_ps'], errors='coerce')

    # Entferne ungültige Einträge (z. B. negative Werte)
    data = data[(data['price_in_euro'] > 0) & (data['mileage_in_km'] > 0) & (data['power_ps'] > 0)]

    return data

# Funktion: ML-Modell trainieren und vorhersagen
def train_and_predict(data, features, target, year_range):
    # Kodierung der kategorischen Variablen
    encoders = {}
    for col in ['color', 'transmission_type', 'fuel_type']:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        encoders[col] = le

    # Entferne Zeilen mit NaN-Werten in den relevanten Spalten
    data = data.dropna(subset=features + [target])

    # Trainings- und Testdaten erstellen
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modelltraining
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Historische Werte und Vorhersagen
    historical_data = data[['year', target]].groupby('year').mean().reset_index()
    year_data = pd.DataFrame({'year': year_range})
    year_data['mileage_in_km'] = data['mileage_in_km'].mean()
    year_data['power_ps'] = data['power_ps'].mean()

    for col in ['color', 'transmission_type', 'fuel_type']:
        mode_value = data[col].mode()[0]
        year_data[col] = encoders[col].transform([mode_value])[0] if mode_value in encoders[col].classes_ else 0

    predicted_prices = model.predict(year_data)

    # Fehlermetriken
    y_pred = model.predict(X_test)
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred)
    }

    return model, predicted_prices, historical_data, year_data, metrics

# Daten vorbereiten
data_gebrauchtwagen = prepare_ml_data(pd.read_csv('gebrauchtwagen.csv'))

# Streamlit Abschnitt
st.header("Machine Learning für Gebrauchtwagen Preisvorhersage")
st.image("ml_forestcar.jpg", caption="Used Cars Random Forest (AI)", use_column_width=True)

st.markdown("<a id='preisvorhersage-nach-fahrzeugmerkmalen'></a>", unsafe_allow_html=True)
with st.expander("Preisvorhersage nach Fahrzeugmerkmalen", expanded=False):
    # Filter für Marke und Modell
    selected_brand = st.selectbox("Wähle eine Marke aus", data_gebrauchtwagen['brand'].unique(), key="unique_brand_select")
    filtered_data = data_gebrauchtwagen[data_gebrauchtwagen['brand'] == selected_brand]

    selected_model = st.selectbox("Wähle ein Modell aus", filtered_data['model'].unique(), key="unique_model_select")
    model_data = filtered_data[filtered_data['model'] == selected_model]

    # Features und Ziel
    features = ['year', 'mileage_in_km', 'power_ps', 'color', 'transmission_type', 'fuel_type']
    target = 'price_in_euro'

    # Vorhersage
    year_range = np.arange(model_data['year'].min(), model_data['year'].max() + 1)
    model, predicted_prices, historical_data, year_data, metrics = train_and_predict(model_data, features, target, year_range)

    # Plot der Vorhersagen
    st.subheader(f"Preis und Vorhersage für {selected_model} nach Baujahr")
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=historical_data, x='year', y=target, label='Historische Preise', color='blue', marker='o')
    sns.lineplot(x=year_data['year'], y=predicted_prices, label='Vorhergesagte Preise', color='red', linestyle='--')
    plt.title(f'Preisverlauf für {selected_model} nach Jahr')
    plt.xlabel('Baujahr')
    plt.ylabel('Durchschnittlicher Preis (€)')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

    # Fehlermetriken
    st.subheader("Modell-Fehlermetriken")
    st.write(f"RMSE: {metrics['rmse']:.2f}")
    st.write(f"MAE: {metrics['mae']:.2f}")
    st.write(f"R²-Score: {metrics['r2']:.2f}")



##############################


# Quellenverzeichnis in der Hauptansicht
st.header("Quellenverzeichnis")
st.markdown("<a id='quellen'></a>", unsafe_allow_html=True)

with st.expander("Quellen", expanded=False):
    st.markdown("### Datenquellen")
    st.markdown(
        """
        - [Deutschlandatlas](https://www.deutschlandatlas.bund.de/DE/Karten/Wie-wir-uns-bewegen/111/_node.html#_t11lwjbxk)
        - [Eurostat Road Transport Data](https://ec.europa.eu/eurostat/databrowser/view/road_eqr_carpda__custom_13451775/default/table?lang=en)
        - [Bundesnetzagentur - E-Mobilität](https://www.bundesnetzagentur.de/DE/Fachthemen/ElektrizitaetundGas/E-Mobilitaet/start.html)
        - [Umweltbundesamt - Durchschnittlicher Kraftstoffverbrauch](https://www.umweltbundesamt.de/bild/durchschnittlicher-kraftstoffverbrauch-von-pkw)
        - [Umweltbundesamt - PKW-Neuzulassungen](https://www.umweltbundesamt.de/bild/entwicklung-der-pkw-neuzulassungen-nach)
        - [Umweltbundesamt - PKW-Bestand nach Kraftstoffart](https://www.umweltbundesamt.de/bild/entwicklung-der-pkw-im-bestand-nach-kraftstoffart)
        - [Kaggle - Germany Used Cars Dataset](https://www.kaggle.com/datasets/wspirat/germany-used-cars-dataset-2023/data)
        """
    )

    st.markdown("### Bildquellen")
    st.markdown(
        """
        - [Bundesnetzagentur - Deutschlandkarte](https://www.bundesnetzagentur.de/DE/Fachthemen/ElektrizitaetundGas/E-Mobilitaet/Deutschlandkarte1.jpg?__blob=publicationFile&v=9)
        - [AutoScout24 Logo](https://www.autoscout24.de/cms-content-assets/1tkbXrmTEPPaTFel6UxtLr-c0eb4849caa00accfa44b32e8da0a2ff-AutoScout24_primary_solid.png)
        - [Pixabay - Tankstellenbild](https://pixabay.com/de/photos/tanken-zapfsäule-tankstelle-diesel-1629074/)
        """
    )
