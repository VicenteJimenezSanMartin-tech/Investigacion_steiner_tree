#include <bits/stdc++.h>
using namespace std;
const long long INF = (long long)4e18;

struct Edge { int u, v, w, idx; };
struct DSU {
    vector<int> p, r;
    DSU(int n=0){ init(n); }
    void init(int n){ p.resize(n); r.assign(n,0); iota(p.begin(), p.end(), 0); }
    int find(int x){ while(p[x]!=x){ p[x]=p[p[x]]; x=p[x]; } return x; }
    bool unite(int a,int b){
        a=find(a); b=find(b); if(a==b) return false;
        if(r[a]<r[b]) swap(a,b);
        p[b]=a; if(r[a]==r[b]) r[a]++;
        return true;
    }
};

long long mstCostInduced(int n, const vector<Edge>& edgesSorted, const vector<int>& selectedList, const vector<char>& selected){
    DSU dsu(n);
    long long cost=0; int used=0; int target=(int)selectedList.size()-1;
    for(const auto& e: edgesSorted){
        if(selected[e.u] && selected[e.v] && dsu.unite(e.u,e.v)){
            cost += e.w; used++;
            if(used == target) break;
        }
    }
    if(used != target) return INF;
    int root = dsu.find(selectedList[0]);
    for(int v: selectedList) if(dsu.find(v) != root) return INF;
    return cost;
}

long long exactSteiner(int n, vector<Edge> edges, const vector<int>& terminals){
    vector<char> isTerm(n,false);
    for(int t: terminals) isTerm[t]=true;
    vector<int> optional;
    for(int v=0; v<n; ++v) if(!isTerm[v]) optional.push_back(v);
    sort(edges.begin(), edges.end(), [](const Edge& a,const Edge& b){return a.w < b.w;});
    long long best = INF;
    int totalMasks = 1 << (int)optional.size();
    for(int mask=0; mask<totalMasks; ++mask){
        vector<char> selected(n,false);
        vector<int> selectedList;
        for(int t: terminals){ selected[t]=true; selectedList.push_back(t); }
        for(int i=0; i<(int)optional.size(); ++i){
            if(mask & (1<<i)){ selected[optional[i]]=true; selectedList.push_back(optional[i]); }
        }
        best = min(best, mstCostInduced(n, edges, selectedList, selected));
    }
    return best;
}

struct DijkstraResult { vector<long long> dist; vector<int> parent, parentEdge; };
DijkstraResult dijkstra(int n, const vector<vector<tuple<int,int,int>>>& adj, int s){
    DijkstraResult res; res.dist.assign(n, INF); res.parent.assign(n,-1); res.parentEdge.assign(n,-1);
    priority_queue<pair<long long,int>, vector<pair<long long,int>>, greater<pair<long long,int>>> pq;
    res.dist[s]=0; pq.push({0,s});
    while(!pq.empty()){
        auto [d,u] = pq.top(); pq.pop();
        if(d != res.dist[u]) continue;
        for(auto [v,w,idx]: adj[u]){
            long long nd = d + w;
            if(nd < res.dist[v]){
                res.dist[v]=nd; res.parent[v]=u; res.parentEdge[v]=idx; pq.push({nd,v});
            }
        }
    }
    return res;
}

long long heuristicSteiner(int n, const vector<Edge>& edges, const vector<int>& terminals){
    vector<vector<tuple<int,int,int>>> adj(n);
    for(const auto& e: edges){
        adj[e.u].push_back({e.v,e.w,e.idx});
        adj[e.v].push_back({e.u,e.w,e.idx});
    }
    int k = terminals.size();
    vector<DijkstraResult> dij;
    for(int t: terminals) dij.push_back(dijkstra(n, adj, t));
    vector<tuple<long long,int,int>> closure;
    for(int i=0;i<k;i++) for(int j=i+1;j<k;j++) closure.push_back({dij[i].dist[terminals[j]], i, j});
    sort(closure.begin(), closure.end());
    DSU dsuTerms(k);
    vector<pair<int,int>> chosen;
    for(auto [d,i,j]: closure){
        if(d >= INF) continue;
        if(dsuTerms.unite(i,j)){ chosen.push_back({i,j}); if((int)chosen.size()==k-1) break; }
    }
    if((int)chosen.size()!=k-1) return INF;
    vector<char> usedEdge(edges.size(), false);
    for(auto [i,j]: chosen){
        int source = terminals[i]; int cur = terminals[j];
        while(cur != source){
            int eid = dij[i].parentEdge[cur];
            if(eid < 0) return INF;
            usedEdge[eid] = true;
            cur = dij[i].parent[cur];
        }
    }
    vector<Edge> sub;
    for(const auto& e: edges) if(usedEdge[e.idx]) sub.push_back(e);
    sort(sub.begin(), sub.end(), [](const Edge&a,const Edge&b){ return a.w < b.w; });
    DSU dsu(n); long long cost=0;
    for(const auto& e: sub) if(dsu.unite(e.u,e.v)) cost += e.w;
    int root = dsu.find(terminals[0]);
    for(int t: terminals) if(dsu.find(t)!=root) return INF;
    return cost;
}

template<class F>
pair<long long,double> avgTime(F func, int repetitions){
    long long res = func(); // calentamiento
    auto start = chrono::high_resolution_clock::now();
    for(int i=0;i<repetitions;i++) res = func();
    auto end = chrono::high_resolution_clock::now();
    double ms = chrono::duration<double, milli>(end-start).count() / repetitions;
    return {res, ms};
}

int main(int argc, char** argv){
    ios::sync_with_stdio(false); cin.tie(nullptr);
    int repetitions = 1;
    if(argc >= 2) repetitions = max(1, atoi(argv[1]));
    int n, m; if(!(cin >> n >> m)) return 1;
    vector<Edge> edges;
    for(int i=0;i<m;i++){ int u,v,w; cin >> u >> v >> w; edges.push_back({u,v,w,i}); }
    int k; cin >> k; vector<int> terminals(k);
    for(int i=0;i<k;i++) cin >> terminals[i];
    auto exact = avgTime([&](){ return exactSteiner(n, edges, terminals); }, repetitions);
    auto heur = avgTime([&](){ return heuristicSteiner(n, edges, terminals); }, repetitions);
    cout << exact.first << " " << fixed << setprecision(6) << exact.second << " "
         << heur.first << " " << fixed << setprecision(6) << heur.second << "\n";
    return 0;
}
