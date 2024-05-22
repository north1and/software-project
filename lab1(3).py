import sys
import re
from collections import defaultdict
import random
import heapq
import networkx as nx

lastword=''

import matplotlib.pyplot as plt

def draw_graph(graph):
    # 创建一个新的图形，设置图形的大小
    plt.figure(figsize=(12, 12))

    G = nx.DiGraph()

    for word1 in graph:
        for word2 in graph[word1]:
            G.add_edge(word1, word2)

    # 使用 spring_layout，可以通过 'k' 参数来调整节点之间的距离
    pos = nx.spring_layout(G, k=0.15)  # k 值越大，节点之间的距离越大

    nx.draw(G, pos, with_labels=True, node_size=1000, font_size=12)

    plt.show()


def build_directed_graph(file_path):
    graph = defaultdict(lambda: defaultdict(int))

    with open(file_path, 'r') as file:
        text = file.read()
        words = re.findall(r'\w+', text.lower())
        for i in range(len(words) - 1):
            graph[words[i]][words[i + 1]] += 1

        global lastword
        lastword=words[i+1]


    return graph

def find_bridge_words(graph, word1, word2):
    if word1 not in graph or (word2 not in graph and word2 != lastword):
        print("No word1 or word2 in the graph!")
        return

    bridge_words = []
    for bridge_word in graph[word1]:
        if bridge_word in graph and word2 in graph[bridge_word]:
            bridge_words.append(bridge_word)

    if not bridge_words:
        print("No bridge words from word1 to word2!")
    else:
        print(f"The bridge words from {word1} to {word2} are: {', '.join(bridge_words)}.")

def insert_bridge_words(graph, sentence):
    words = re.findall(r'\w+', sentence.lower())
    new_words = [words[0]]

    for i in range(1, len(words)):
        word1 = words[i - 1]
        word2 = words[i]

        bridge_words = []
        for bridge_word in graph[word1]:
            if bridge_word in graph and word2 in graph[bridge_word]:
                bridge_words.append(bridge_word)

        if bridge_words:
            new_words.append(random.choice(bridge_words))

        new_words.append(word2)

    new_sentence = ' '.join(new_words)
    return new_sentence

def dijkstra_shortest_path(graph, start_word, end_word):
    if start_word not in graph or end_word not in graph:
        print("No word1 or word2 in the graph!")
        return None, float('inf')

    pq = [(0, start_word, [start_word])]
    visited = set()

    while pq:
        weight, word, path = heapq.heappop(pq)

        if word == end_word:
            return path, weight

        if word not in visited:
            visited.add(word)
            for neighbor, edge_weight in graph[word].items():
                heapq.heappush(pq, (weight + edge_weight, neighbor, path + [neighbor]))

    return None, float('inf')


def random_walk(graph, start_node=None):
    if not start_node:
        start_node = random.choice(list(graph.keys()))

    path = [start_node]
    edges = []

    while True:
        if path[-1] not in graph or not graph[path[-1]]:
            break

        next_node = random.choice(list(graph[path[-1]].keys()))
        edge = (path[-1], next_node)

        if edge in edges:
            break

        path.append(next_node)
        edges.append(edge)

    return path, edges


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    directed_graph = build_directed_graph(file_path)

    # 需求 1
    # print("Directed Graph:")
    # for node, edges in directed_graph.items():
    #     print(f"{node}: {edges}")

    # 需求 2
    # draw_graph(directed_graph)

    # 需求 3
    # word1 = input("Enter word1: ").lower()
    # word2 = input("Enter word2: ").lower()
    # find_bridge_words(directed_graph, word1, word2)


    # 需求 4
    # sentence = input("Enter a sentence: ").lower()
    # new_sentence = insert_bridge_words(directed_graph, sentence)

    # print(f"Original sentence: {sentence}")
    # print(f"New sentence: {new_sentence}")


    # 需求 5
    # word1 = input("Enter word1: ").lower()
    # word2 = input("Enter word2: ").lower()

    # shortest_path, path_length = dijkstra_shortest_path(directed_graph, word1, word2)

    # if shortest_path:
    #     print(f"The shortest path from {word1} to {word2} is:")
    #     print("->".join(shortest_path))
    #     print(f"Length of the path: {path_length}")
    # else:
    #     print("No path exists between the given words!")

    # 需求 6
    path, edges = random_walk(directed_graph)
    print(f"Random walk path: {' -> '.join(path)}")
