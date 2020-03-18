import os
import json
import networkx as nx
import numpy as np
from random import random
from flask import Flask, request

app = Flask(__name__)

port = os.environ.get("PORT", 8080)


def maybe_add_node(G, node):
    if not G.has_node(node):
        G.add_node(node)


def wa(scores):
    total_weights = sum([x[1] for x in scores])

    if total_weights == 0.0:
        return None

    return sum([x[0] * x[1] for x in scores]) / total_weights


def weight(G, node, path=[]):
    if node in path:
        return 0

    edges = list(G.in_edges([node], data=True))

    if len(path) > 5 or len(edges) == 0:
        return 0

    scores = [
        np.log(1 + np.exp(x[2]["weight"])) * weight(G, x[0], path + [x[0]])
        for x in edges
    ]
    scores = [x for x in scores if x > 0]

    return np.prod(np.array(scores))


def score(G, node):
    edges = list(G.in_edges([node], data=True))

    if len(edges) == 0:
        return None

    scores = [[x[2]["weight"], weight(G, x[0])] for x in edges]
    return wa(scores)


def dropout(G, rate):
    for edge in list(G.edges()):
        if random() < rate:
            G.remove_edge(*edge)


@app.route("/calculate", methods=["POST"])
def calculate():
    data = request.json["review_matrix"]
    iterations = request.json["iterations"]
    dropout_rate = request.json["dropout"]
    # This matrix is of the form:
    #   [{reviewee_id -> [{reviewer_id -> latest_review_score}, ...}, ...]

    G = nx.DiGraph()

    for reviewee_id in data:
        maybe_add_node(G, reviewee_id)

        for reviewer_id in data[reviewee_id]:
            maybe_add_node(G, reviewer_id)

            G.add_edge(reviewer_id, reviewee_id, weight=data[reviewee_id][reviewer_id])

    scores = {}

    for i in range(iterations):
        G_current = G.copy()

        if dropout_rate > 0.0:
            dropout(G_current, dropout_rate)

        for node in G_current.nodes():
            s = score(G_current, node)
            if s is not None:
                if node not in scores:
                    scores[node] = []
                scores[node].append(s)

    print("Found %d nodes and %d edges" % (len(scores), len(G.edges())))

    ranked = sorted(scores.items(), key=lambda x: np.mean(x[1]))

    results = {}

    for x in ranked:
        score_hist = np.histogram(x[1], bins=np.arange(0, 1.1, 0.1))
        score_mean = np.mean(x[1])
        score_sd = np.std(x[1])
        results[x[0]] = {
            "score_mean": score_mean,
            "score_sd": score_sd,
            "score_hist": [
                {"x": score_hist[1][i], "y": int(score_hist[0][i])}
                for i in range(len(score_hist[0]))
            ],
            "indegree": G.in_degree(x[0]),
            "outdegree": G.out_degree(x[0]),
        }

    return {
        "ok": True,
        "results": {
            "mean": np.mean([np.mean(ix[1]) for ix in ranked]),
            "std": np.std([np.mean(ix[1]) for ix in ranked]),
            "median": np.median([np.mean(ix[1]) for ix in ranked]),
            "iters": iterations,
            "dropout": dropout_rate,
            "employees": results,
        },
    }


app.run(
    host="0.0.0.0", port=port, debug=True, use_reloader=True,
)
