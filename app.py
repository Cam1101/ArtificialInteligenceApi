import os
from flask import Flask, request, jsonify
from recommender import recommend_categories

app = Flask(__name__)

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        purchase_history = data.get('purchase_history', [])
        if not purchase_history:
            return jsonify({"error": "Purchase history is empty"}), 400

        recommended_categories = recommend_categories(purchase_history)
        return jsonify({"recommended_categories": recommended_categories})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=True)
