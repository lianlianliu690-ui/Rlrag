from pyvis.network import Network
import networkx as nx

gexf_path = "/Dataset4D/public/mas-liu.lianlian/output/RAGesture/rl_kg/semantic_spk2/semantic_layer.gexf"
G = nx.read_gexf(gexf_path)

# 创建 PyVis 网络
net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", select_menu=True)

# PyVis 目前对 NetworkX 的直接转换支持有限，最好手动转换或使用 from_nx
net.from_nx(G)

# 启用物理引擎控制面板
net.show_buttons(filter_=['physics'])

# 保存并打开
net.save_graph("rag_graph_semantic_graph.html")
print("Interactive graph saved to rag_graph_motion_graph.html")