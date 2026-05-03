from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from joblib import Parallel, delayed
import time, random

app = Flask(__name__)
model = {}

def load_data():
    print("Loading data...")
    ratings = pd.read_csv('ratings.dat', sep='::', header=None,
                          names=['user_id','movie_id','rating','timestamp'], engine='python')
    movies  = pd.read_csv('movies.dat',  sep='::', header=None,
                          names=['movie_id','title','genres'], engine='python', encoding='latin-1')
    movie_map = dict(zip(movies['movie_id'], movies['title']))
    df = ratings.merge(movies[['movie_id','title']], on='movie_id')

    mc = df['movie_id'].value_counts()
    df = df[df['movie_id'].isin(mc[mc >= 50].index)]
    uc = df['user_id'].value_counts()
    df = df[df['user_id'].isin(uc[uc >= 50].index)]
    top300 = df['movie_id'].value_counts().head(300).index
    df = df[df['movie_id'].isin(top300)]

    uim  = df.pivot_table(index='user_id', columns='movie_id', values='rating').fillna(0)
    norm = uim.copy().astype(float)
    norm = norm.sub(norm.replace(0, np.nan).mean(axis=1), axis=0).fillna(0)
    sim  = cosine_similarity(norm)
    sim_df = pd.DataFrame(sim, index=uim.index, columns=uim.index)

    model.update({
        'uim': uim, 'sim_df': sim_df, 'movie_map': movie_map,
        'popular': list(df['movie_id'].value_counts().head(10).index),
        'matrix_np': norm.values,
        'all_users': list(uim.index),
        'n_users': len(uim), 'n_movies': len(uim.columns), 'n_ratings': len(df),
    })
    print(f"Ready! Users:{len(uim)} Movies:{len(uim.columns)}")

def recommend_movies(user_id, top_n=5):
    user_id = int(user_id)
    uim     = model['uim']
    scores  = model['sim_df'][user_id].sort_values(ascending=False).iloc[1:6]
    rec     = set()
    for u in scores.index:
        items = uim.loc[u]
        rec.update(items[items >= 3].index)
    if not rec:
        for u in scores.index:
            items = uim.loc[u]
            rec.update(items[items > 0].index)
    rated = set(uim.loc[user_id][uim.loc[user_id] > 0].index)
    final = list(rec - rated)
    if not final:
        final = [m for m in model['popular'] if m not in rated]
    return [model['movie_map'].get(m, str(m)) for m in final[:top_n]]

def process_chunk(chunk):
    return cosine_similarity(chunk, model['matrix_np'])

@app.route('/')
def index():
    return render_template('index.html',
        users=[str(u) for u in model['all_users'][:300]],
        n_users=f"{model['n_users']:,}",
        n_movies=f"{model['n_movies']:,}",
        n_ratings=f"{model['n_ratings']:,}")

@app.route('/recommend', methods=['POST'])
def recommend():
    data   = request.json
    uid    = data.get('user_id')
    top_n  = int(data.get('top_n', 5))
    M      = model['matrix_np']
    n      = len(M)
    cs     = n // 4
    chunks = [M[i*cs:(i+1)*cs] for i in range(4)]

    t0 = time.time()
    [process_chunk(c) for c in chunks]
    seq_t = time.time() - t0

    t0 = time.time()
    Parallel(n_jobs=4, prefer='threads')(delayed(process_chunk)(c) for c in chunks)
    par_t = time.time() - t0

    try:
        recs = recommend_movies(uid, top_n)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

    sp  = round(seq_t / par_t, 2) if par_t > 0 else 0
    eff = round((sp / 4) * 100, 1)
    return jsonify({'recommendations': recs,
                    'seq_time': round(seq_t*1000),
                    'par_time': round(par_t*1000),
                    'speedup': sp, 'efficiency': eff})

@app.route('/random_user')
def random_user():
    users_300 = model['all_users'][:300]
    uid = random.choice(users_300)
    return jsonify({'user_id': str(uid)})
if __name__ == '__main__':
    load_data()
    app.run(host='0.0.0.0', port=5000, debug=False)
