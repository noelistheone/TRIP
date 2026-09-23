"""PACT — Pooled Allocation with Client-owned Twinned write-sets.

A defense against server-side inverse-probing attacks (TRIP and relatives) on
HE-protected federated recommendation, under two hard constraints:

  * the decrypted aggregate is **bit-identical** to honest FedAvg, so
    recommendation accuracy is unchanged by construction (Theorem 4);
  * per-round asymptotic cost is unchanged for client and server (Theorem 5).

Four rules:

  WS  Write-Set Sovereignty   the CLIENT decides which item rows its update
                              touches. Server-supplied probe assignments,
                              probes-only instructions and optimizer overrides
                              are advisory-for-display and discarded.
  BC  Block-Closed Aggregation clients are partitioned by a public beacon into
                              sticky blocks of size t; participant sets are
                              unions of whole blocks; pairwise masks cancel
                              within the block. => rank(B) <= ceil(N/t).
  TB  Transcript Binding      signed round header over (round, catalog Merkle
                              root, recipe digest, participant set), plus
                              norm-sanity bands on the broadcast model.
  SENTRY                      client-side one-class detector. Strictly outside
                              the security guarantee; fail-safe (can only
                              abort).

The security parameter is the block size `t`: the server can learn block sums,
never individual rows, so the best any solver can do is the pooled ceiling of
Theorem 3.
"""
from .blocks import StickyPartition, beacon_partition, pooling_matrix
from .catalog import RoundHeader, merkle_root, sign_header, verify_header
from .mask import block_masks, mask_for
from .policy import ClientPolicy, policy_from_cfg
from .prf import PactRng

__all__ = [
    "ClientPolicy", "policy_from_cfg", "PactRng",
    "StickyPartition", "beacon_partition", "pooling_matrix",
    "RoundHeader", "merkle_root", "sign_header", "verify_header",
    "block_masks", "mask_for",
]
