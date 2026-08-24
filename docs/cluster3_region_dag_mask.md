# Cluster-3 Region-DAG attention mask

Contract ID: `region_dag_conservative_gdn_v1`

The canonical mask has one row for every submitted absolute query position and
one column for every canonical sequence/page-table position. Parent visibility
always means the complete transitive ancestor closure.

- A stable query can see stable ancestors and positions no later than itself in
  its own stable region.
- A stable query can never see an active key.
- An active query can see all positions in its own active region and all
  positions in every stable or active ancestor region.
- Siblings cannot see each other unless an explicit ancestor path connects
  them.

The serving builder flattens each request's `query_count x sequence_length`
boolean mask in request order. It validates int64 absolute query positions,
exact dimensions, boolean dtype, contiguity, and total flattened length. The
contract always selects `custom_paged`; `native_structured` and `full_paged`
cannot silently replace an arbitrary Region-DAG mask.

The two helpers in the Torchtitan Qwen3.5 file are pure reference functions.
They are not connected to the training forward, `[x0; xt]` construction,
sampling, or loss path.
