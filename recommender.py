import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
import json
import os

# Ruta del archivo de historial de compras global
HISTORY_FILE = 'purchase_history.json'
MAX_HISTORY_SIZE = 300

def load_global_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as file:
            return json.load(file)
    return []

def save_global_history(history):
    # Limitar el tamaño del historial
    if len(history) > MAX_HISTORY_SIZE:
        history = history[-MAX_HISTORY_SIZE:]  # Mantener solo los últimos 300 registros
    with open(HISTORY_FILE, 'w') as file:
        json.dump(history, file, indent=4)  # Usar indent para un formato más legible

def recommend_categories(purchase_history):
    global_history = load_global_history()
    
    # Extraer categorías de las compras y actualizar el historial global
    new_categories = [purchase['category'] for purchase in purchase_history]
    global_history.extend(new_categories)
    save_global_history(global_history)
    
    if len(global_history) < 2:
        return list(set(new_categories))
    
    df = pd.DataFrame(global_history, columns=['category'])
    
    # Preprocesamiento
    le = LabelEncoder()
    df['category_encoded'] = le.fit_transform(df['category'])
    
    # Crear un DataFrame ficticio para ajustar el modelo
    df['dummy_price'] = 0
    df['dummy_quantity'] = 1
    X = df[['dummy_price', 'dummy_quantity', 'category_encoded']]
    
    # Modelado
    n_neighbors = min(3, len(X))  # Ajustar n_neighbors a no más del número de muestras disponibles
    model = NearestNeighbors(n_neighbors=n_neighbors)
    model.fit(X)
    
    # Generar recomendaciones para las nuevas categorías
    new_categories_encoded = le.transform(new_categories)
    new_data = pd.DataFrame({
        'dummy_price': [0] * len(new_categories_encoded),
        'dummy_quantity': [1] * len(new_categories_encoded),
        'category_encoded': new_categories_encoded
    })
    
    distances, indices = model.kneighbors(new_data)
    
    recommended_indices = indices.flatten()
    recommended_categories = df.iloc[recommended_indices]['category'].unique().tolist()
    
    # Limitar el número de categorías recomendadas a 2
    recommended_categories = recommended_categories[:2]
    
    # Considerar las categorías más compradas
    category_totals = df['category'].value_counts()
    top_categories = category_totals.nlargest(2).index.tolist()
    
    # Combinar las recomendaciones con las categorías más compradas, asegurando que sean únicas
    final_recommendations = list(set(recommended_categories + top_categories))[:2]
    
    return final_recommendations
