from cassandra.cluster import Cluster

def connect_cassandra():
    """
    Connexion au cluster Cassandra local (127.0.0.1).
    Retourne la session connectée au keyspace 'covid'.
    """
    cluster = Cluster(['127.0.0.1'])
    session = cluster.connect()
    session.set_keyspace('covid')
    return session
