#!/usr/bin/env python3
"""
Steiner Tree en grafos no dirigidos ponderados.
Incluye una solución exacta por enumeración de subconjuntos de nodos de Steiner
para instancias pequeñas y una heurística por clausura métrica + MST.

Formato de entrada:
    n m
    u v w
    ... (m aristas, nodos 0-indexados)
    k t1 t2 ... tk

Salida:
    exact_cost exact_ms heuristic_cost heuristic_ms
"""
from __future__ import annotations
import sys
import time
import heapq
from dataclasses import dataclass
from typing import List, Tuple, Set, Dict

INF = 10**18

@dataclass(frozen=True)
class Edge:
    u: int
    v: int
    w: int
    idx: int

class DSU:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


def read_instance() -> Tuple[int, List[Edge], List[int]]:
    data = sys.stdin.read().strip().split()
    if not data:
        raise ValueError("Entrada vacía")
    it = iter(data)
    n, m = int(next(it)), int(next(it))
    edges = []
    for i in range(m):
        u, v, w = int(next(it)), int(next(it)), int(next(it))
        edges.append(Edge(u, v, w, i))
    k = int(next(it))
    terminals = [int(next(it)) for _ in range(k)]
    return n, edges, terminals


def mst_cost_induced(n: int, edges_sorted: List[Edge], selected: Set[int]) -> int:
    dsu = DSU(n)
    cost = 0
    used = 0
    target_edges = len(selected) - 1
    for e in edges_sorted:
        if e.u in selected and e.v in selected and dsu.union(e.u, e.v):
            cost += e.w
            used += 1
            if used == target_edges:
                break
    if used != target_edges:
        return INF
    # Verifica que todos los nodos seleccionados estén en el mismo componente.
    root = None
    for v in selected:
        if root is None:
            root = dsu.find(v)
        elif dsu.find(v) != root:
            return INF
    return cost


def exact_steiner(n: int, edges: List[Edge], terminals: List[int]) -> int:
    terminal_set = set(terminals)
    optional = [v for v in range(n) if v not in terminal_set]
    edges_sorted = sorted(edges, key=lambda e: e.w)
    best = INF
    total_masks = 1 << len(optional)
    for mask in range(total_masks):
        selected = set(terminal_set)
        for i, v in enumerate(optional):
            if mask & (1 << i):
                selected.add(v)
        cost = mst_cost_induced(n, edges_sorted, selected)
        if cost < best:
            best = cost
    return best


def dijkstra(n: int, adj: List[List[Tuple[int, int, int]]], source: int):
    dist = [INF] * n
    parent = [-1] * n
    parent_edge = [-1] * n
    dist[source] = 0
    pq = [(0, source)]
    while pq:
        d, u = heapq.heappop(pq)
        if d != dist[u]:
            continue
        for v, w, idx in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                parent[v] = u
                parent_edge[v] = idx
                heapq.heappush(pq, (nd, v))
    return dist, parent, parent_edge


def heuristic_steiner(n: int, edges: List[Edge], terminals: List[int]) -> int:
    adj = [[] for _ in range(n)]
    for e in edges:
        adj[e.u].append((e.v, e.w, e.idx))
        adj[e.v].append((e.u, e.w, e.idx))

    k = len(terminals)
    all_dist, all_parent, all_parent_edge = [], [], []
    for t in terminals:
        dist, parent, parent_edge = dijkstra(n, adj, t)
        all_dist.append(dist)
        all_parent.append(parent)
        all_parent_edge.append(parent_edge)

    closure_edges = []
    for i in range(k):
        for j in range(i + 1, k):
            closure_edges.append((all_dist[i][terminals[j]], i, j))
    closure_edges.sort()

    dsu_terms = DSU(k)
    chosen_pairs = []
    for d, i, j in closure_edges:
        if d >= INF:
            continue
        if dsu_terms.union(i, j):
            chosen_pairs.append((i, j))
            if len(chosen_pairs) == k - 1:
                break
    if len(chosen_pairs) != k - 1:
        return INF

    used_edge_ids = set()
    for i, j in chosen_pairs:
        source_terminal = terminals[i]
        target = terminals[j]
        current = target
        parent = all_parent[i]
        parent_edge = all_parent_edge[i]
        while current != source_terminal:
            eid = parent_edge[current]
            if eid == -1:
                return INF
            used_edge_ids.add(eid)
            current = parent[current]

    # El subgrafo unión puede tener ciclos; se calcula MST sobre sus aristas.
    sub_edges = [e for e in edges if e.idx in used_edge_ids]
    sub_edges.sort(key=lambda e: e.w)
    dsu = DSU(n)
    cost = 0
    for e in sub_edges:
        if dsu.union(e.u, e.v):
            cost += e.w

    root = dsu.find(terminals[0])
    if any(dsu.find(t) != root for t in terminals):
        return INF
    return cost


def average_time(func, repetitions: int) -> Tuple[int, float]:
    # Una ejecución de calentamiento evita medir inicialización de estructuras/cachés.
    result = func()
    start = time.perf_counter()
    for _ in range(repetitions):
        result = func()
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / repetitions
    return result, elapsed_ms


def main():
    repetitions = 1
    if len(sys.argv) >= 2:
        repetitions = max(1, int(sys.argv[1]))
    n, edges, terminals = read_instance()
    exact_cost, exact_ms = average_time(lambda: exact_steiner(n, edges, terminals), repetitions)
    heuristic_cost, heuristic_ms = average_time(lambda: heuristic_steiner(n, edges, terminals), repetitions)
    print(f"{exact_cost} {exact_ms:.6f} {heuristic_cost} {heuristic_ms:.6f}")

if __name__ == "__main__":
    main()
