# HNSW Algorithm Deep Dive

## What is HNSW?

Hierarchical Navigable Small World (HNSW) is a graph-based approximate nearest neighbor (ANN) algorithm published by Malkov and Yashunin in 2016. It achieves sub-linear search complexity while maintaining very high recall, making it the algorithm of choice for production vector databases like Pinecone, Weaviate, and Qdrant.

## The Core Insight: Multiple Layers

Traditional "Navigable Small World" graphs suffer from logarithmic complexity degradation at large scales. HNSW solves this by adding a hierarchical structure. Vectors are inserted into multiple layers. The top layers have very few nodes and act as long-range "highways." The bottom layer (layer 0) contains every node and serves as the dense neighborhood graph.

Think of it like a highway system: if you want to drive from New York to Los Angeles, you start on an interstate (top layer), then transition to a state highway (middle layer), and finally take local streets to your exact destination (layer 0).

## Layer Assignment

When a new vector is inserted, it is randomly assigned a maximum layer using the formula:

```
level = floor(-ln(uniform_random) * (1 / ln(M)))
```

where M is the maximum connections per node. This exponential distribution ensures that approximately 1/M fraction of nodes appear in each successive higher layer, creating a natural hierarchy.

For M=16, roughly 93% of nodes are in layer 0 only, 6% appear in layers 0 and 1, and the remaining 1% form the higher-level highway structure.

## Insertion Algorithm

Insertion proceeds in two phases:

**Phase 1: Greedy Descent (No Connections)**
Start at the global entry point (the node with the highest layer). For each layer above the new node's assigned level, perform a greedy search with ef=1 (follow only the single closest neighbor). This efficiently narrows down the entry point for the connection phase.

**Phase 2: Connect**
For each layer from min(new_level, max_layer) down to 0:
1. Run a beam search with ef_construction candidates to find the best neighborhood.
2. Select the M closest candidates as neighbors.
3. Wire bidirectional edges.
4. Prune any neighbor's adjacency list that now exceeds M connections by removing the worst (most distant) neighbor.

## Search Algorithm

1. Start at the global entry point.
2. For layers from max_layer down to 1: greedy descent (ef=1), updating entry point.
3. At layer 0: full beam search with ef_search candidates.
4. Return the top-K results from the beam search.

The ef_search parameter controls the accuracy/speed tradeoff. Larger ef_search = higher recall but slower search.

## Selecting M and ef Parameters

- **M** (max connections): Values 8–64 work well. M=16 is a common default. Higher M improves recall but increases memory and insert time.
- **ef_construction**: Controls build quality. 64–200 are typical. VecDB uses 128.
- **ef_search**: Controls query quality. VecDB uses max(k, 200). Increase for higher recall.

## Why Cosine Similarity?

VecDB normalizes all vectors to unit length at insert time. For unit vectors, cosine similarity equals the dot product. This is important because dot product can be computed with SIMD FMA instructions in a tight loop, while cosine similarity requires an extra division. By normalizing once at insert time and then using dot product everywhere, VecDB achieves maximum SIMD throughput.

The distance stored is `1 - dot_product`, converting similarity to a proper distance metric where 0 means identical and 2 means maximally opposite.

## Recall vs Brute Force

HNSW trades a small recall loss for massive speed improvements. At 10K vectors, VecDB's HNSW achieves approximately 92% Recall@10 compared to exact brute-force search. At 1M vectors, HNSW is thousands of times faster than brute force while still achieving 95%+ recall with appropriate ef_search settings.
