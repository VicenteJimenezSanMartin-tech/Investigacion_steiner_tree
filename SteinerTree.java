import java.io.*;
import java.util.*;

public class SteinerTree {
    static final long INF = Long.MAX_VALUE / 4;

    static class Edge {
        int u, v, w, idx;
        Edge(int u, int v, int w, int idx) { this.u = u; this.v = v; this.w = w; this.idx = idx; }
    }

    static class DSU {
        int[] p, r;
        DSU(int n) { p = new int[n]; r = new int[n]; for (int i = 0; i < n; i++) p[i] = i; }
        int find(int x) { while (p[x] != x) { p[x] = p[p[x]]; x = p[x]; } return x; }
        boolean union(int a, int b) {
            a = find(a); b = find(b); if (a == b) return false;
            if (r[a] < r[b]) { int tmp = a; a = b; b = tmp; }
            p[b] = a; if (r[a] == r[b]) r[a]++;
            return true;
        }
    }

    static class AdjEdge {
        int to, w, idx;
        AdjEdge(int to, int w, int idx) { this.to = to; this.w = w; this.idx = idx; }
    }

    static class DijkstraResult {
        long[] dist;
        int[] parent, parentEdge;
        DijkstraResult(int n) { dist = new long[n]; parent = new int[n]; parentEdge = new int[n]; }
    }

    static class FastScanner {
        private final InputStream in;
        private final byte[] buffer = new byte[1 << 16];
        private int ptr = 0, len = 0;
        FastScanner(InputStream is) { in = is; }
        int read() throws IOException {
            if (ptr >= len) { len = in.read(buffer); ptr = 0; if (len <= 0) return -1; }
            return buffer[ptr++];
        }
        int nextInt() throws IOException {
            int c; do { c = read(); } while (c <= ' ' && c != -1);
            int sign = 1; if (c == '-') { sign = -1; c = read(); }
            int val = 0;
            while (c > ' ') { val = val * 10 + (c - '0'); c = read(); }
            return val * sign;
        }
    }

    static long mstCostInduced(int n, List<Edge> sorted, List<Integer> selectedList, boolean[] selected) {
        DSU dsu = new DSU(n);
        long cost = 0; int used = 0, target = selectedList.size() - 1;
        for (Edge e : sorted) {
            if (selected[e.u] && selected[e.v] && dsu.union(e.u, e.v)) {
                cost += e.w; used++;
                if (used == target) break;
            }
        }
        if (used != target) return INF;
        int root = dsu.find(selectedList.get(0));
        for (int v : selectedList) if (dsu.find(v) != root) return INF;
        return cost;
    }

    static long exactSteiner(int n, List<Edge> edges, int[] terminals) {
        boolean[] isTerminal = new boolean[n];
        for (int t : terminals) isTerminal[t] = true;
        ArrayList<Integer> optional = new ArrayList<>();
        for (int v = 0; v < n; v++) if (!isTerminal[v]) optional.add(v);
        ArrayList<Edge> sorted = new ArrayList<>(edges);
        sorted.sort(Comparator.comparingInt(e -> e.w));
        long best = INF;
        int totalMasks = 1 << optional.size();
        for (int mask = 0; mask < totalMasks; mask++) {
            boolean[] selected = new boolean[n];
            ArrayList<Integer> selectedList = new ArrayList<>();
            for (int t : terminals) { selected[t] = true; selectedList.add(t); }
            for (int i = 0; i < optional.size(); i++) {
                if ((mask & (1 << i)) != 0) { selected[optional.get(i)] = true; selectedList.add(optional.get(i)); }
            }
            best = Math.min(best, mstCostInduced(n, sorted, selectedList, selected));
        }
        return best;
    }

    static DijkstraResult dijkstra(int n, ArrayList<AdjEdge>[] adj, int source) {
        DijkstraResult res = new DijkstraResult(n);
        Arrays.fill(res.dist, INF); Arrays.fill(res.parent, -1); Arrays.fill(res.parentEdge, -1);
        PriorityQueue<long[]> pq = new PriorityQueue<>(Comparator.comparingLong(a -> a[0]));
        res.dist[source] = 0; pq.add(new long[]{0, source});
        while (!pq.isEmpty()) {
            long[] cur = pq.poll(); long d = cur[0]; int u = (int)cur[1];
            if (d != res.dist[u]) continue;
            for (AdjEdge e : adj[u]) {
                long nd = d + e.w;
                if (nd < res.dist[e.to]) {
                    res.dist[e.to] = nd; res.parent[e.to] = u; res.parentEdge[e.to] = e.idx;
                    pq.add(new long[]{nd, e.to});
                }
            }
        }
        return res;
    }

    static long heuristicSteiner(int n, List<Edge> edges, int[] terminals) {
        @SuppressWarnings("unchecked")
        ArrayList<AdjEdge>[] adj = new ArrayList[n];
        for (int i = 0; i < n; i++) adj[i] = new ArrayList<>();
        for (Edge e : edges) { adj[e.u].add(new AdjEdge(e.v, e.w, e.idx)); adj[e.v].add(new AdjEdge(e.u, e.w, e.idx)); }
        int k = terminals.length;
        DijkstraResult[] dij = new DijkstraResult[k];
        for (int i = 0; i < k; i++) dij[i] = dijkstra(n, adj, terminals[i]);
        ArrayList<long[]> closure = new ArrayList<>();
        for (int i = 0; i < k; i++) for (int j = i + 1; j < k; j++) closure.add(new long[]{dij[i].dist[terminals[j]], i, j});
        closure.sort(Comparator.comparingLong(a -> a[0]));
        DSU termDsu = new DSU(k);
        ArrayList<int[]> chosen = new ArrayList<>();
        for (long[] ce : closure) {
            if (ce[0] >= INF) continue;
            int i = (int)ce[1], j = (int)ce[2];
            if (termDsu.union(i, j)) { chosen.add(new int[]{i, j}); if (chosen.size() == k - 1) break; }
        }
        if (chosen.size() != k - 1) return INF;
        boolean[] usedEdge = new boolean[edges.size()];
        for (int[] pair : chosen) {
            int i = pair[0], j = pair[1];
            int source = terminals[i], cur = terminals[j];
            while (cur != source) {
                int eid = dij[i].parentEdge[cur];
                if (eid < 0) return INF;
                usedEdge[eid] = true;
                cur = dij[i].parent[cur];
            }
        }
        ArrayList<Edge> sub = new ArrayList<>();
        for (Edge e : edges) if (usedEdge[e.idx]) sub.add(e);
        sub.sort(Comparator.comparingInt(e -> e.w));
        DSU dsu = new DSU(n); long cost = 0;
        for (Edge e : sub) if (dsu.union(e.u, e.v)) cost += e.w;
        int root = dsu.find(terminals[0]);
        for (int t : terminals) if (dsu.find(t) != root) return INF;
        return cost;
    }

    interface Solver { long run(); }
    static class Timed { long value; double ms; Timed(long value, double ms) { this.value = value; this.ms = ms; } }
    static Timed avgTime(Solver solver, int repetitions) {
        long result = solver.run(); // calentamiento
        long start = System.nanoTime();
        for (int i = 0; i < repetitions; i++) result = solver.run();
        long end = System.nanoTime();
        return new Timed(result, (end - start) / 1_000_000.0 / repetitions);
    }

    public static void main(String[] args) throws Exception {
        int repetitions = 1;
        if (args.length >= 1) repetitions = Math.max(1, Integer.parseInt(args[0]));
        FastScanner fs = new FastScanner(System.in);
        int n = fs.nextInt(), m = fs.nextInt();
        ArrayList<Edge> edges = new ArrayList<>();
        for (int i = 0; i < m; i++) edges.add(new Edge(fs.nextInt(), fs.nextInt(), fs.nextInt(), i));
        int k = fs.nextInt(); int[] terminals = new int[k];
        for (int i = 0; i < k; i++) terminals[i] = fs.nextInt();
        Timed exact = avgTime(() -> exactSteiner(n, edges, terminals), repetitions);
        Timed heur = avgTime(() -> heuristicSteiner(n, edges, terminals), repetitions);
        System.out.printf(Locale.US, "%d %.6f %d %.6f%n", exact.value, exact.ms, heur.value, heur.ms);
    }
}
