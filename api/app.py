import sys
import os
import logging
from datetime import datetime
from flask import Flask, jsonify, request
from flask_cors import CORS
from cassandra.cluster import Cluster
from cassandra import ConsistencyLevel
import re

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

app = Flask(__name__)
CORS(app)  # Permet les requêtes cross-origin

class DatabaseConnection:
    def __init__(self):
        self.session = None
        self.cluster = None
        self.connect()
    
    def connect(self):
        try:
            self.cluster = Cluster(['127.0.0.1'])
            self.session = self.cluster.connect()
            self.session.set_keyspace('covid')
            logger.info("Connexion à Cassandra établie")
        except Exception as e:
            logger.error(f"Erreur de connexion à Cassandra: {e}")
            raise
    
    def get_session(self):
        if not self.session:
            self.connect()
        return self.session
    
    def close(self):
        if self.cluster:
            self.cluster.shutdown()

# Instance globale de connexion
db = DatabaseConnection()

def validate_country_name(country):
    """Valide et nettoie le nom du pays"""
    if not country:
        return None
    
    # Supprime les caractères spéciaux dangereux
    country = re.sub(r'[^\w\s\-\.]', '', country)
    country = country.strip()
    
    if len(country) < 2 or len(country) > 100:
        return None
    
    return country

def validate_date_range(start_date=None, end_date=None):
    """Valide les dates optionnelles"""
    try:
        if start_date:
            start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        if end_date:
            end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
        
        if start_date and end_date and start_date > end_date:
            return None, None, "Date de début postérieure à la date de fin"
        
        return start_date, end_date, None
    except ValueError:
        return None, None, "Format de date invalide. Utilisez YYYY-MM-DD"

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint non trouvé'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Erreur interne: {error}")
    return jsonify({'error': 'Erreur interne du serveur'}), 500

@app.route("/")
def get_countries():
    """Récupère la liste des pays disponibles"""
    try:
        session = db.get_session()
        rows = session.execute("SELECT DISTINCT country FROM stats")
        countries = sorted([row.country for row in rows if row.country])
        
        logger.info(f"Récupération de {len(countries)} pays")
        return jsonify({
            'countries': countries,
            'total': len(countries)
        })
    
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des pays: {e}")
        return jsonify({'error': 'Erreur lors de la récupération des pays'}), 500

@app.route("/stats/<country>")
def get_stats(country):
    """Récupère les statistiques pour un pays donné"""
    try:
        # Validation du pays
        clean_country = validate_country_name(country)
        if not clean_country:
            return jsonify({'error': 'Nom de pays invalide'}), 400
        
        # Paramètres optionnels de date
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        limit = request.args.get('limit', type=int)
        
        # Validation des dates
        start_date, end_date, date_error = validate_date_range(start_date, end_date)
        if date_error:
            return jsonify({'error': date_error}), 400
        
        # Construction de la requête
        session = db.get_session()
        query = "SELECT date, total_cases, total_deaths FROM stats WHERE country=%s"
        params = [clean_country]
        
        if start_date and end_date:
            query += " AND date >= %s AND date <= %s"
            params.extend([start_date, end_date])
        
        query += " ORDER BY date ASC"
        
        if limit and limit > 0:
            query += f" LIMIT {min(limit, 10000)}"  # Limite maximale de sécurité
        
        rows = session.execute(query, params)
        data = []
        
        for row in rows:
            data.append({
                "date": str(row.date),
                "total_cases": row.total_cases or 0,
                "total_deaths": row.total_deaths or 0
            })
        
        if not data:
            return jsonify({'error': f'Aucune donnée trouvée pour {clean_country}'}), 404
        
        logger.info(f"Récupération de {len(data)} enregistrements pour {clean_country}")
        
        return jsonify({
            'country': clean_country,
            'data': data,
            'total_records': len(data),
            'date_range': {
                'start': data[0]['date'] if data else None,
                'end': data[-1]['date'] if data else None
            }
        })
    
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des stats pour {country}: {e}")
        return jsonify({'error': 'Erreur lors de la récupération des données'}), 500

@app.route("/stats/<country>/summary")
def get_country_summary(country):
    """Récupère un résumé des statistiques pour un pays"""
    try:
        clean_country = validate_country_name(country)
        if not clean_country:
            return jsonify({'error': 'Nom de pays invalide'}), 400
        
        session = db.get_session()
        
        # Statistiques agrégées
        summary_query = """
            SELECT 
                MAX(total_cases) as max_cases,
                MAX(total_deaths) as max_deaths,
                MIN(date) as first_date,
                MAX(date) as last_date
            FROM stats 
            WHERE country=%s
        """
        
        result = session.execute(summary_query, [clean_country]).one()
        
        if not result or result.max_cases is None:
            return jsonify({'error': f'Aucune donnée trouvée pour {clean_country}'}), 404
        
        # Données récentes (derniers 30 jours)
        recent_query = """
            SELECT date, total_cases, total_deaths 
            FROM stats 
            WHERE country=%s 
            ORDER BY date DESC 
            LIMIT 30
        """
        
        recent_rows = session.execute(recent_query, [clean_country])
        recent_data = [
            {
                "date": str(row.date),
                "total_cases": row.total_cases or 0,
                "total_deaths": row.total_deaths or 0
            }
            for row in recent_rows
        ]
        
        summary = {
            'country': clean_country,
            'max_cases': result.max_cases or 0,
            'max_deaths': result.max_deaths or 0,
            'first_date': str(result.first_date) if result.first_date else None,
            'last_date': str(result.last_date) if result.last_date else None,
            'recent_data': recent_data[:10]  # 10 derniers points
        }
        
        return jsonify(summary)
    
    except Exception as e:
        logger.error(f"Erreur lors de la récupération du résumé pour {country}: {e}")
        return jsonify({'error': 'Erreur lors de la récupération du résumé'}), 500

@app.route("/health")
def health_check():
    """Point de contrôle de santé de l'API"""
    try:
        session = db.get_session()
        # Test simple de connexion
        session.execute("SELECT COUNT(*) FROM stats LIMIT 1")
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'database': 'connected'
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'timestamp': datetime.now().isoformat(),
            'database': 'disconnected',
            'error': str(e)
        }), 500

if __name__ == "__main__":
    try:
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        logger.info("Arrêt de l'application")
    finally:
        db.close()