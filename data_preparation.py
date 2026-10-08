import pandas as pd
from cassandra_connect import connect_cassandra
from cassandra.query import BatchStatement
from cassandra import ConsistencyLevel
from datetime import datetime

def clean_data(df):
    """
    Nettoie le dataframe en sélectionnant uniquement les colonnes utiles,
    renommage, suppression des NaN, et conversion en int.
    """
    df = df[['location', 'date', 'total_cases', 'total_deaths']]
    df = df.rename(columns={'location': 'country'})
    df = df.dropna(subset=['country', 'date', 'total_cases', 'total_deaths'])
    df['total_cases'] = df['total_cases'].astype(int)
    df['total_deaths'] = df['total_deaths'].astype(int)
    return df

def insert_data_batch(session, df, batch_size=100):
    """
    Insère les données dans Cassandra par batchs de taille définie.
    """
    insert_stmt = session.prepare("""
        INSERT INTO stats (country, date, total_cases, total_deaths)
        VALUES (?, ?, ?, ?)
    """)
    insert_stmt.consistency_level = ConsistencyLevel.ONE

    batch = BatchStatement(consistency_level=ConsistencyLevel.ONE)
    count = 0

    for _, row in df.iterrows():
        date_obj = datetime.strptime(row['date'], '%Y-%m-%d').date()
        batch.add(insert_stmt, (row['country'], date_obj, row['total_cases'], row['total_deaths']))
        count += 1

        if count % batch_size == 0:
            session.execute(batch)
            batch.clear()
            print(f"✅ {count} lignes insérées...")

    if len(batch) > 0:
        session.execute(batch)
        print(f"✅ {count} lignes insérées (finalisation du dernier batch)...")

    print("✅ Insertion terminée avec méthode batch.")

if __name__ == '__main__':
    df = pd.read_csv('owid-covid-data.csv')
    df_clean = clean_data(df)
    session = connect_cassandra()
    insert_data_batch(session, df_clean, batch_size=100)  # batch_size ajustable


