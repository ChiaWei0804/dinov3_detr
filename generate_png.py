from graphviz import Digraph
import os

def create_architecture_diagram():
    # 建立有向圖 - 橫向布局
    dot = Digraph(comment='DINOv3 DETR Architecture', format='png')
    
    # === 全局樣式設定 ===
    dot.attr(rankdir='LR')  # Left to Right (橫向)
    dot.attr(compound='true')
    dot.attr(splines='polyline')
    dot.attr(nodesep='0.3', ranksep='0.5')
    dot.attr(fontname='Arial')
    
    # === 1. 輸入與 Backbone (Frozen) ===
    with dot.subgraph(name='cluster_0') as c:
        c.attr(label='🔒 Frozen Backbone', style='dashed', color='#546E7A', 
               fontcolor='#546E7A', fontsize='14', labelloc='t')
        c.attr('node', shape='box', style='filled,rounded', fillcolor='#ECEFF1', 
               color='#455A64', fontname='Arial', fontsize='11')
        
        c.node('input', '📷 Input Image\n[B=8, 3, 800, 800]')
        c.node('backbone', 'DINOv3 ViT-B/16\n(Frozen)\nPatch Size: 16×16')
        c.node('raw_tokens', 'Patch Tokens\n[8, 2500, 768]\n(50×50 grid)', 
               shape='ellipse', fillcolor='#CFD8DC')
        
        c.edge('input', 'backbone')
        c.edge('backbone', 'raw_tokens', label='forward_features()')

    # === 2. 特徵投影 ===
    with dot.subgraph(name='cluster_1') as c:
        c.attr(label='🔧 Feature Projection', style='solid', color='#00897B', 
               fontcolor='#00897B', fontsize='14', labelloc='t')
        c.attr('node', shape='box', style='filled,rounded', fillcolor='#B2DFDB', 
               color='#00695C', fontname='Arial', fontsize='11')
        
        c.node('proj', 'Linear Projection\n768 → 256')
        c.node('pos_embed', 'Sine Positional\nEncoding\n(Dynamic 50×50)')
        c.node('feat_256', 'Features [8, 2500, 256]', shape='ellipse', fillcolor='#E0F2F1')
        
        c.edge('proj', 'feat_256')

    # === 3. Transformer Encoder ===
    with dot.subgraph(name='cluster_2') as c:
        c.attr(label='🔄 Transformer Encoder (2 Layers)', style='solid', color='#F9A825', 
               fontcolor='#F57F17', fontsize='14', labelloc='t')
        c.attr('node', shape='box', style='filled,rounded', fillcolor='#FFF9C4', 
               color='#FBC02D', fontname='Arial', fontsize='11')
        
        c.node('enc_attn', '💡 Self-Attention\n(Multi-Head, 8 heads)\nQ=K=V from features', 
               fillcolor='#FFF59D', color='#F9A825')
        c.node('enc_ffn', 'Feed-Forward\nNetwork (FFN)\n256 → 1024 → 256')
        c.node('memory', 'Encoder Memory\n[8, 2500, 256]\n(Global Context)', 
               shape='ellipse', fillcolor='#E8F5E9', color='#43A047')
        
        c.edge('enc_attn', 'enc_ffn', label='× 2 layers')
        c.edge('enc_ffn', 'memory')

    # === 4. 混合查詢選擇 ===
    with dot.subgraph(name='cluster_3') as c:
        c.attr(label='🎯 Mixed Query Selection', style='solid', color='#EF6C00', 
               fontcolor='#E65100', fontsize='14', labelloc='t')
        c.attr('node', shape='box', style='filled,rounded', fillcolor='#FFE0B2', 
               color='#F57C00', fontname='Arial', fontsize='11')
        
        c.node('topk_head', 'Score Head\nLinear(256 → 1)')
        c.node('topk_select', 'Top-K Selection\nSelect top-100\nfrom 2500 patches', 
               fillcolor='#FFCC80')
        c.node('init_queries', 'Initial Queries\n[8, 100, 256]\n(Content + Pos)', 
               shape='ellipse', fillcolor='#E8F5E9', color='#43A047')
        
        c.edge('topk_head', 'topk_select', label='Scores [8, 2500]')
        c.edge('topk_select', 'init_queries', label='Gather top-100')

    # === 5. Transformer Decoder ===
    with dot.subgraph(name='cluster_4') as c:
        c.attr(label='🔄 Transformer Decoder (6 Layers)', style='solid', color='#6A1B9A', 
               fontcolor='#4A148C', fontsize='14', labelloc='t')
        c.attr('node', shape='box', style='filled,rounded', fillcolor='#E1BEE7', 
               color='#8E24AA', fontname='Arial', fontsize='11')
        
        c.node('dec_self_attn', '💡 Self-Attention\n(Query-to-Query)\nQ=K=V from queries', 
               fillcolor='#CE93D8', color='#8E24AA')
        c.node('dec_cross_attn', '🔗 Cross-Attention\n(Query-to-Memory)\nQ: queries\nK,V: encoder memory', 
               fillcolor='#BA68C8', color='#7B1FA2')
        c.node('dec_ffn', 'Feed-Forward\nNetwork (FFN)\n256 → 1024 → 256')
        c.node('dec_out', 'Query Features\n[8, 100, 256]', 
               shape='ellipse', fillcolor='#E8F5E9', color='#43A047')
        
        c.edge('dec_self_attn', 'dec_cross_attn', label='× 6 layers')
        c.edge('dec_cross_attn', 'dec_ffn')
        c.edge('dec_ffn', 'dec_out')

    # === 6. Prediction Heads ===
    with dot.subgraph(name='cluster_5') as c:
        c.attr(label='📊 Prediction Heads', style='solid', color='#C62828', 
               fontcolor='#B71C1C', fontsize='14', labelloc='t')
        c.attr('node', shape='box', style='filled,rounded', fillcolor='#FFCDD2', 
               color='#E53935', fontname='Arial', fontsize='11')
        
        c.node('cls_head', 'Classification Head\n256 → 256 (LN+ReLU)\n→ 92 classes')
        c.node('bbox_head', 'BBox Regression Head\n256 → 256 → 128\n→ 4 (cx,cy,w,h)')

    # === 7. Final Outputs ===
    dot.attr('node', shape='note', style='filled', fillcolor='#FFEBEE', 
             color='#C62828', fontname='Arial Bold', fontsize='12')
    dot.node('out_cls', '📋 Class Logits\n[8, 100, 92]\n(91 COCO + BG)')
    dot.node('out_bbox', '📦 BBox Predictions\n[8, 100, 4]\n(Normalized)')

    # === 主流程連線 ===
    dot.edge('raw_tokens', 'proj', label='768D')
    dot.edge('feat_256', 'enc_attn', label='+ Pos Embed')
    dot.edge('pos_embed', 'enc_attn', style='dashed', color='#00897B')
    
    # Query Selection
    dot.edge('memory', 'topk_head', label='2500 patches')
    dot.edge('memory', 'topk_select', style='dashed', color='#43A047', label='Gather features')
    
    # Decoder 輸入
    dot.edge('init_queries', 'dec_self_attn', label='Query Input')
    dot.edge('memory', 'dec_cross_attn', style='dashed', color='#43A047', 
             label='K, V\n(Encoder Memory)', fontsize='10')
    
    # Prediction Heads
    dot.edge('dec_out', 'cls_head')
    dot.edge('dec_out', 'bbox_head')
    dot.edge('cls_head', 'out_cls')
    dot.edge('bbox_head', 'out_bbox')

    # 輸出檔案
    output_path = 'dinov3_detr_architecture_horizontal'
    dot.render(output_path, view=False, cleanup=True)
    print(f"✅ 圖表已生成：{os.getcwd()}/{output_path}.png")
    print(f"   - 布局方式: 橫向 (Left to Right)")
    print(f"   - Self-Attention: 已標註於 Encoder 和 Decoder")
    print(f"   - Cross-Attention: 已標註於 Decoder")

if __name__ == '__main__':
    create_architecture_diagram()