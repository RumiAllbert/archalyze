import streamlit as st
from collections import Counter
import matplotlib.pyplot as plt
import spacy
import networkx as nx
from itertools import combinations
import streamlit.components.v1 as components
from pyvis.network import Network

def visualize_character_network(text, window_size):
    G = generate_character_network(text, window_size)
    
    # Convert networkx graph to pyvis network
    nt = Network(notebook=True)
    nt.from_nx(G)
    
    # Save to HTML and display within Streamlit
    tmp_file = "tmp_graph.html"
    nt.save_graph(tmp_file)
    components.html(open(tmp_file).read(), width=800, height=800)


def generate_character_network(text, window_size=150):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    
    # Extract characters
    characters = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    
    # Create a co-occurrence matrix
    co_occurrences = Counter()
    for i in range(len(characters) - 1):
        for j in range(i + 1, len(characters)):
            distance = abs(i - j)
            if distance <= window_size:
                pair = tuple(sorted([characters[i], characters[j]]))
                co_occurrences[pair] += 1
    
    # Create a network graph
    G = nx.Graph()
    for (char1, char2), freq in co_occurrences.items():
        if freq > 5:  # Only add significant relationships
            G.add_edge(char1, char2, weight=freq)

    return G


# def visualize_character_network(text, window_size=150):
#     G = generate_character_network(text, window_size)

#     # Draw the network
#     plt.figure(figsize=(10, 10))
#     pos = nx.spring_layout(G)
#     nx.draw_networkx_nodes(G, pos)
#     nx.draw_networkx_labels(G, pos)
#     nx.draw_networkx_edges(G, pos, width=[d["weight"] for (u, v, d) in G.edges(data=True)])
#     st.pyplot(plt)
