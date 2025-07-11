
from flask import Flask, render_template, request, redirect, url_for, send_file
import os
import pandas as pd
from model_utils import run_all_models, save_model_comparison, save_anomalies
from plot_utils import generate_charts

app = Flask(__name__, static_folder='static', template_folder='templates')
UPLOAD_FOLDER = 'data'
OUTPUT_FOLDER = 'outputs'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Store results globally for navigation
app_state = {'results': None, 'best_model': None}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate')
def generate_data():
    import numpy as np
    app_state['results'] = None
    app_state['best_model'] = None

    np.random.seed(42)
    n_normal, n_ddos = 100, 40

    normal = pd.DataFrame({
        'duration': np.random.exponential(1.0, n_normal),
        'src_bytes': np.random.randint(100, 500, n_normal),
        'dst_bytes': np.random.randint(100, 500, n_normal),
        'count': np.random.randint(1, 50, n_normal),
        'srv_count': np.random.randint(1, 50, n_normal),
        'label': ['normal'] * n_normal
    })

    ddos = pd.DataFrame({
        'duration': np.random.exponential(0.2, n_ddos),
        'src_bytes': np.random.randint(1000, 5000, n_ddos),
        'dst_bytes': np.random.randint(0, 100, n_ddos),
        'count': np.random.randint(50, 100, n_ddos),
        'srv_count': np.random.randint(50, 100, n_ddos),
        'label': ['ddos'] * n_ddos
    })

    df = pd.concat([normal, ddos]).sample(frac=1).reset_index(drop=True)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'synthetic_network.csv')
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    df.to_csv(filepath, index=False)

    results, best_model = run_all_models(df)
    generate_charts(results, df)
    save_model_comparison(results)
    save_anomalies(df)

    app_state['results'] = results
    app_state['best_model'] = best_model

    return '''<script>window.location.replace("/results?success=1");</script>'''

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if file.filename != '':
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(filepath)

        df = pd.read_csv(filepath)
        results, best_model = run_all_models(df)
        generate_charts(results, df)
        save_model_comparison(results)
        save_anomalies(df)

        app_state['results'] = results
        app_state['best_model'] = best_model

        return '''<script>window.location.replace("/results?success=upload");</script>'''
    return redirect('/')

@app.route('/results')
def results():
    return render_template('results.html', results=app_state['results'], best_model=app_state['best_model'])

@app.route('/visualizations')
def visualizations():
    return render_template('visualizations.html')

@app.route('/features')
def features():
    return render_template('features.html')

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(OUTPUT_FOLDER, filename), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
