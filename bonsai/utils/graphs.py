# import csv
# import igraph as ig
#
# ## this takes too long. Too big graph need a better way to do this
# def compute_hops_graph(file):
#     with open(file, 'r') as f:
#         reader = csv.DictReader(f)
#         g = ig.Graph()
#         vertices = {}
#         for row in reader:
#             src = row['source']
#             src_asn = row['source_asn']
#             dst = row['destination']
#             dst_asn = row['dest_asn']
#             count = row['count']
#             hop_avg = row['hop_avg']
#             hop_std = row['hop_std']
#             if count < 5 or src_asn < 0 or dst_asn < 0:
#                 continue
#             if (src, src_asn) not in vertices:
#                 vertices[(src, src_asn)] = g.add_vertex(name=src, asn=src_asn)
#             if (dst, dst_asn) not in vertices:
#                 vertices[(dst, dst_asn)] = g.add_vertex(name=dst, asn=dst_asn)
#
#             g.add_edge(vertices[(src, src_asn)], vertices[(dst, dst_asn)], count=count, hop_avg=hop_avg,
#                        hop_std=hop_std)
#     return g
