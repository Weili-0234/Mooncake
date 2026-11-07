# Recovery-Unit Eviction Policy (`unit_gds`)

## Overview

The default memory eviction order of the master (`--eviction_policy=lease`) evicts objects whose lease ran out, oldest
lease first, a fixed fraction of the objects per round. It does not know which objects belong together, how large they
are, or what it costs a client to recreate them.

An inference engine that caches its KV state in Mooncake Store writes objects that are only useful together. A request
can resume from a cached prefix only when every object it needs is present: the attention pages of the prefix and,
for hybrid models with recurrent layers (Mamba, gated delta rule, Kimi delta attention and similar), the recurrent
state at the resume position. Evicting one of those objects wastes the others, a shared prefix page is needed by every
continuation that extends it, and a deep state saves far more prefill time than a shallow one of the same size.

With `--eviction_policy=unit_gds` the master evicts **recovery units** instead of single objects. Clients register
units on the admin port; the master ranks them by GreedyDual-Size over the prefill time each saves and evicts the
lowest-ranked unit whole. The policy is model-agnostic: the master only sees keys, byte sizes and numbers the client
computes.

## Units

A unit is a position a client can resume computation from. A client registers it with:

| Field | Meaning |
| --- | --- |
| `id` | The unit's id. The client also puts the unit's own objects with this id as their object group (`ReplicateConfig.group_ids`), so they share a metadata shard. |
| `members` | The objects the unit owns: evicted with it, all or none. A member belongs to one unit. |
| `refs` | The path pages the unit needs, in path order. Pages are shared: a page is evicted only once the last unit that references it is evicted. A page listed twice, or also among the unit's members, is ignored after its first listing. |
| `value` | The cost, in seconds, of computing the unit's position from position 0. Costs add along a path, so the master derives what a unit saves by subtraction. |
| `parent` | A shallower unit on the same path (a hint: it may be unknown or already evicted). |
| `aligned` | The unit ends on a page boundary, so its path pages alone identify it: any unit whose path holds its last page lies deeper on the same path. |

The master keeps, per unit, its rank inputs:

- **fallback**: the deepest live unit before it on its path, which a request falls back to once this unit is gone. It
  is the parent hint (resolved through units that were evicted, by the fallback each had when it went) or an aligned
  unit that ends on one of the unit's pages, whichever is deeper.
- **gain**: `value - value(fallback)`, or `value` without a fallback; 0 for a dead unit.
- **size**: the bytes evicting the unit frees: its present members and the path pages no other unit references.
  Reference counts never grow deeper along a path, so those pages are a suffix of the path.
- **stamp**: the clock `L` when the unit was last used (registered or touched).

The score is `stamp + gain / size`, lowest first. `L` rises to the score of each evicted unit (each evicted leaf with
leaf-last tiers), so a unit that is not used again ages relative to fresh ones.

### Leaf-last tiers

With `--unit_leaf_last=true` (the default of the policy) the rank is `(tier, score)`:

- tier 0, **dead**: a unit the client marked dead, a unit that lost a member or a path page, a unit registered after
  one of its objects was erased, and a unit whose members did not all arrive within `--unit_orphan_grace_s`;
- tier 1, **inner**: a unit with a deeper live unit on its path (through the parent hints or the aligned-unit rule),
  unless it is a branch point;
- tier 2, **leaf**: every other unit, and branch points: units touched by more than one request, or with two live
  continuations that part right after them.

With `--unit_idle_demote_s` > 0, a leaf that is not a branch point and was not used for that many seconds ranks as an
inner unit. Every inner unit goes before any leaf; GreedyDual-Size orders each tier.

With `--unit_anchor_tier=true`, an aligned unit that is the deepest live aligned unit under a live unaligned unit (a
tail that ends inside a page) ranks with the leaves instead of with the inner units. That aligned unit is the anchor
of the tail: the position a request resumes from when it rewrites the end of the tail, for example a conversation
turn whose last tokens the client changed. It is found through the parent hints (a live unaligned unit below it,
looking through units that went), or, without hints, when a live unaligned unit and no other live aligned unit holds
its last page.

### Dead-unit headroom

Regular rounds start only above the high watermark, so a store kept close to it can refuse puts for lack of space
before a round frees any. With `--unit_dead_headroom_ratio=r` (requires leaf-last tiers), the eviction thread also runs
dead-only rounds once usage
exceeds `high watermark - r`: they evict orphans and dead units, never a live unit, until usage is back below that
level. Clients that mark superseded units dead then find room before the store is full, while every live unit stays
until a regular round needs its space.

### Orphans

An object that no unit owns or references is an orphan. It becomes evictable `--unit_orphan_grace_s` after it was
stored (a unit may still register it), and orphans go before any unit, oldest first: they are ordered by the time they
were stored, and reads do not change that order. Pages a unit was the last to reference become orphans as soon as it
goes. A client that registers no units therefore sees its objects evicted in the order they were stored, however often
they are read.

A regular round that has taken everything else it may evict and still has space to free takes orphans that are still
within their grace period, again oldest first. This last resort keeps a store of objects that no client registers (a
client without unit support, or registrations lost when the master restarted) from refusing puts. Dead-only rounds
never take an orphan before its grace period ends.

## Eviction

The eviction thread keeps the existing triggers: usage above `--eviction_high_watermark_ratio` or a failed allocation.
A round has the lease order's target, `max(eviction_ratio, usage - high watermark + eviction_ratio)`, taken as a
fraction of the memory capacity's bytes instead of the number of objects. It takes orphans first, then units in rank
order, and last the orphans still within their grace period. For each unit the master checks every present member
under its shard lock: if any member is leased (being read), hard pinned, has an active soft pin, or has no evictable
memory replica, the whole unit is skipped until the next round.
(Under `unit_gds` an active soft pin always protects its unit; `allow_evict_soft_pinned_objects` does not apply.)
Otherwise every member is erased; pages whose last reference went become orphans. A member that stops being evictable
before it is erased (the unit was touched or registered again, or a read leased the member) skips the unit until the
next round as long as none of its members went yet; once one went, the unit is dead, ranked again at once.

Puts that were started but not completed are swept as in the lease order.

## Admin API

Both endpoints are served on the master's admin (metrics) port and return `409 Conflict` under the lease order.

`POST /api/v1/units` applies a batch of updates in order:

```json
{"updates": [
  {"op": "upsert", "id": "u7", "value": 2.31, "aligned": false, "parent": "u3",
   "members": ["k/state/0", "k/state/1"], "refs": ["k/page/0", "k/page/1"]},
  {"op": "touch", "id": "u3"},
  {"op": "dead", "id": "u2"}
]}
```

- `upsert` registers a unit. Registering a known id again counts as a use, adds members it did not list before and
  sets a missing parent hint.
- `touch` marks a use of a known unit: a request resumed from it. Only the touched unit is refreshed; shallower units
  on its path keep their stamp.
- `dead` marks a unit dead: it is evicted first, but stays until space is needed.

The response lists one message per rejected update: `{"success": false, "applied": 2, "errors": ["unit u9: a unit
owns at least one object"]}`. A body that is not a valid batch returns `400 Bad Request`: one that is not a JSON object
with a list of update objects, a field given with another type than above, or more than 4096 updates. The master
applies a batch in chunks of 256 updates and releases the policy's lock between them, so puts never wait for a whole
batch; updates of other requests may come between two chunks.

`GET /api/v1/units/stats` returns the policy's counters, kept up to date as they change, so a request costs the same
whatever the number of units: units, objects and their bytes, orphan candidates, updates applied (and path pages they
listed twice or also as members, ignored), evictions and their bytes (per tier with leaf-last tiers; `evicted_dead`,
`evicted_inner` and `evicted_leaf` stay 0 without them), orphans evicted (and those of them taken within their grace
period), units blocked by a lease, partial evictions, members and pages lost, what dead-only rounds evicted, and the
clock `L`. The units per tier (`units_dead`, `units_inner`, `units_leaf`, and `units_anchor` for the anchors among the
leaves) are 0 unless the request asks for `?check=1`, which counts them and runs the self-check: it compares every
unit's members and pages with the reverse indexes and returns the violations it finds (an empty list when consistent).
Both scan every unit while puts wait for the policy, so the master answers `?check=1` only with `--unit_debug` and
refuses it with `400 Bad Request` otherwise.

## Configuration

| Flag | Config key | Default | Description |
| --- | --- | --- | --- |
| `--eviction_policy` | `eviction_policy` | `lease` | `lease` or `unit_gds` |
| `--unit_leaf_last` | `unit_leaf_last` | `true` | Dead units first, then inner units, then leaves |
| `--unit_idle_demote_s` | `unit_idle_demote_s` | `60` | A leaf unused this many seconds ranks as an inner unit; `0` disables |
| `--unit_orphan_grace_s` | `unit_orphan_grace_s` | `120` | Seconds before an unregistered object may be evicted (earlier only as a full round's last resort), and before a unit whose members have not all arrived is dead |
| `--unit_debug` | `unit_debug` | `false` | Answer `GET /api/v1/units/stats?check=1` (the units per tier and the self-check) |
| `--unit_anchor_tier` | `unit_anchor_tier` | `false` | The anchor of a live tail ranks with the leaves (requires `unit_leaf_last`) |
| `--unit_dead_headroom_ratio` | `unit_dead_headroom_ratio` | `0` | Dead-only rounds from this fraction of the capacity below the high watermark on; `0` disables (requires `unit_leaf_last`, below the high watermark) |

`unit_gds` keeps its state in the master's memory only: it is refused together with high availability, offload,
snapshots, CXL and multiple tenants. It never evicts an object with an active soft pin, so
`allow_evict_soft_pinned_objects` has no effect under it; the master logs a warning at startup when that setting is on
(its default).

## Compatibility

- The lease order is unchanged: without `--eviction_policy=unit_gds` no policy object exists and every new code path is
  skipped.
- Clients need no change to read and write under `unit_gds`; a client that registers no units sees its objects as
  orphans, evicted oldest first after the grace period (within it only when a round finds nothing else to evict). Units
  use existing client features only (object groups and the admin HTTP port), so older clients and older masters
  interoperate as before.

## Implementation

`UnitPolicy` (`mooncake-store/include/unit_policy.h`) holds the unit table, the reverse indexes (object to owner,
page to holders, last page to aligned units, parent hints) and a lazily updated heap. Page and member keys are interned:
the units and the indexes hold 4-byte ids, so a key's string is kept once however many units reference it, and the
key goes once the last unit referencing it does. `MasterService` feeds it from
`PutEnd` (an object became readable, with its size) and from metadata erasure, and asks it for victims in
`UnitBatchEvict`. Every public method of the policy takes the policy's lock; the master calls into it under a metadata
shard lock and the policy never calls back into the master, so no lock order cycle exists. Time is passed in by the
caller, which makes the policy deterministic under test.

Tests: `unit_policy_test.cpp` (the order, sizes, fallbacks, tiers, demotion, branch points, orphans, blocked units,
the self-check and a randomized consistency run), `master_service_unit_policy_test.cpp` (whole-unit eviction through
the master, leases, erased members, the eviction thread) and the unit endpoints in `master_admin_server_test.cpp`.
