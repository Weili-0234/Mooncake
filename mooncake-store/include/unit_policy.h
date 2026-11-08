#pragma once

#include <chrono>
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <queue>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace mooncake {

// Settings of the recovery-unit eviction policy
// (``--eviction_policy=unit_gds``).
struct UnitPolicyConfig {
    bool enabled = false;
    // Dead units first, then inner units, then leaves; GreedyDual-Size within
    // each tier. False: one tier.
    bool leaf_last = true;
    // A leaf (not a branch point) unused for this many seconds ranks as an
    // inner unit; 0 disables.
    double idle_demote_s = 60.0;
    // An object that no unit owns or references becomes evictable this many
    // seconds after it was stored; a unit whose members have not all arrived
    // this long after its registration is dead.
    double orphan_grace_s = 120.0;
    // Read by the master: the admin API runs the self-check and counts the
    // units per tier (``?check=1``), each a scan of every unit under the
    // policy's lock.
    bool debug = false;
    // Leaf-last only: the deepest live aligned unit under a live tail (its
    // anchor, the state a rewrite of the tail's end resumes at) ranks with
    // the leaves instead of with the inner units.
    bool anchor_tier = false;
    // Leaf-last only, read by the master: from this fraction of the capacity
    // below the high watermark on, dead units and orphans (never live units)
    // are evicted, so a full store does not refuse puts first; 0 disables.
    double dead_headroom_ratio = 0.0;
};

enum class UnitOp { kUpsert, kTouch, kDead };

// One update from a client.
struct UnitUpdate {
    UnitOp op = UnitOp::kUpsert;
    std::string id;
    // Recompute seconds from position 0 to the unit's position. Costs add
    // along a path, so a unit's gain is its value minus its fallback's.
    double value = 0.0;
    // The unit sits on a page boundary: its path pages alone identify it, so
    // any unit whose path holds its last page is deeper on the same path.
    bool aligned = false;
    // A shallower unit on the same path (a hint: it may be unknown or gone).
    std::string parent;
    // Objects the unit owns: evicted with it, all or none.
    std::vector<std::string> members;
    // Path pages the unit needs, in path order, shared between units: a page
    // goes when the last unit referencing it goes.
    std::vector<std::string> refs;
};

struct UnitPolicyStats {
    uint64_t units = 0;
    uint64_t units_dead = 0;
    uint64_t units_inner = 0;
    uint64_t units_leaf = 0;
    // Of the leaf-tier units, the anchors (anchor_tier).
    uint64_t units_anchor = 0;
    uint64_t objects = 0;
    uint64_t object_bytes = 0;
    uint64_t orphan_candidates = 0;
    uint64_t upserts = 0;
    uint64_t touches = 0;
    uint64_t touch_unknown = 0;
    uint64_t dead_marks = 0;
    uint64_t dead_unknown = 0;
    uint64_t member_conflicts = 0;
    // Path pages an upsert listed twice or also as a member, ignored.
    uint64_t dropped_refs = 0;
    uint64_t evicted_units = 0;
    uint64_t evicted_unit_bytes = 0;
    // Dead, inner, leaf; with leaf-last tiers only.
    uint64_t evicted_by_tier[3] = {0, 0, 0};
    uint64_t orphans = 0;
    uint64_t orphan_bytes = 0;
    // Of the orphans, those taken within their grace period (NextGraceOrphan).
    uint64_t grace_orphans = 0;
    uint64_t grace_orphan_bytes = 0;
    uint64_t blocked = 0;
    uint64_t partial = 0;
    uint64_t lost_members = 0;
    uint64_t lost_pages = 0;
    uint64_t pending_expired = 0;
    // Evicted below the high watermark (dead_headroom_ratio), included in
    // the counts above.
    uint64_t headroom_units = 0;
    uint64_t headroom_unit_bytes = 0;
    uint64_t headroom_orphans = 0;
    uint64_t headroom_orphan_bytes = 0;
    uint64_t heap_entries = 0;
    double clock = 0.0;
    // Passes over the parent hints that drop the links of units that went.
    uint64_t gone_cleanups = 0;
    uint64_t demotion_entries = 0;

    bool operator==(const UnitPolicyStats&) const = default;
};

// Recovery units over the store's objects, ranked by GreedyDual-Size with
// leaf-last tiers, for one master. A unit is a resume point a client
// registers: the objects it owns (members) and the path pages it needs
// (refs). Its gain is its value minus its fallback's (the deepest live unit
// before it on its path: its parent hint, resolved through units that went,
// or an aligned unit ending on one of its pages) and its size the bytes that
// go with it (present members and the path pages no other unit references).
// Score: (tier, stamp + gain / size), lowest first; stamp is the clock L at
// the unit's last use (registration or touch), and L rises to the score of
// each evicted leaf (of each evicted unit without tiers). Objects no unit owns
// or references (orphans) go first once past their grace period; the pages
// an evicted unit was the last to reference are orphans at once.
//
// Thread safety: every public method takes the policy's lock. The master
// calls OnObjectStored, OnObjectErased and StillEvictable under a metadata
// shard lock, and the policy never calls back into the master.
class UnitPolicy {
   public:
    using Clock = std::chrono::steady_clock;

    explicit UnitPolicy(UnitPolicyConfig config);

    const UnitPolicyConfig& config() const { return config_; }

    // Client updates, applied in order. Returns one message per rejected
    // update (empty when every update applied).
    std::vector<std::string> Apply(const std::vector<UnitUpdate>& updates,
                                   Clock::time_point now);

    // Master hooks: ``key`` became readable with ``bytes``, or was erased.
    void OnObjectStored(const std::string& key, uint64_t bytes,
                        Clock::time_point now);
    void OnObjectErased(const std::string& key, Clock::time_point now);

    struct Victim {
        bool orphan = false;
        // An orphan taken by NextGraceOrphan.
        bool grace = false;
        std::string unit;
        // The present members of the unit, or the orphan.
        std::vector<std::string> keys;
        int tier = 0;
        double score = 0.0;
        // Taken by a dead-only round (dead_headroom_ratio).
        bool headroom = false;
    };

    // The lowest-ranked orphan or unit not yet tried this round, or nullopt;
    // with ``dead_only``, only orphans and dead units (leaf-last order puts
    // them first), nullopt once the next candidate is live.
    std::optional<Victim> NextVictim(Clock::time_point now,
                                     bool dead_only = false);
    // Whether the master may still erase ``key`` for ``victim``.
    bool StillEvictable(const Victim& victim, const std::string& key);
    // The last resort once NextVictim has nothing left: the orphan whose grace
    // period ends first, even if it has not ended yet, or nullopt.
    std::optional<Victim> NextGraceOrphan();
    // Every key of ``victim`` was erased (``complete``), freeing ``bytes``,
    // or only some: then the unit is dead and ranks again at once.
    void Evicted(const Victim& victim, bool complete, uint64_t bytes,
                 Clock::time_point now);
    // A key of ``victim`` is leased or pinned: skip it until EndRound.
    void Blocked(const Victim& victim);
    // End of an eviction round: blocked and waiting units rank again.
    void EndRound(Clock::time_point now);

    // Every counter, and the units per tier: a scan of every unit.
    UnitPolicyStats Stats(Clock::time_point now) const;
    // Stats without the tier counts (units_dead, units_inner, units_leaf and
    // units_anchor stay 0), in constant time.
    UnitPolicyStats Counters(Clock::time_point now) const;
    // A live unit's (tier, score), or nullopt for an unknown id.
    std::optional<std::pair<int, double>> RankOf(const std::string& id,
                                                 Clock::time_point now) const;
    // Violations of the policy's own invariants; empty when consistent.
    std::vector<std::string> Check() const;
    // Test-only: drops ``id``'s reference to its last page without telling
    // the unit, a corruption Check() must report.
    void DropPageRefForTesting(const std::string& id);

   private:
    static constexpr int kDead = 0;
    static constexpr int kInner = 1;
    static constexpr int kLeaf = 2;

    // Page and member keys are interned: units and the indexes over them
    // hold these ids, and each key's string is kept once.
    using KeyId = uint32_t;
    static constexpr KeyId kNoKey = ~KeyId{0};
    struct Key {
        // The key in key_ids_.
        const std::string* name = nullptr;
        // Occurrences in the units' members and refs.
        uint32_t refs = 0;
    };

    struct Object {
        uint64_t bytes = 0;
        Clock::time_point stored_at;
    };

    struct Unit {
        double value = 0.0;
        bool aligned = false;
        std::string parent;
        std::vector<KeyId> members;
        std::vector<KeyId> refs;
        double stamp = 0.0;
        Clock::time_point used_at;
        Clock::time_point registered_at;
        // Uses after the registration: more than one makes a branch point.
        uint64_t touches = 0;
        size_t present_members = 0;
        bool dead = false;
        bool evicting = false;
        // Skipped until EndRound (leased, or members still arriving).
        bool parked = false;
        bool used_while_evicting = false;
        uint64_t ver = 0;
        int pushed_tier = -1;
    };

    struct Entry {
        int tier;
        double score;
        uint64_t seq;
        std::string id;
        uint64_t ver;
        bool operator>(const Entry& other) const {
            if (tier != other.tier) return tier > other.tier;
            if (score != other.score) return score > other.score;
            return seq > other.seq;
        }
    };

    void UpsertLocked(const UnitUpdate& update, Clock::time_point now);
    void UseLocked(Unit& unit, const std::string& id, Clock::time_point now);
    void RemoveUnitLocked(const std::string& id, Clock::time_point now);
    void PushLocked(const std::string& id, Clock::time_point now);
    void AddOrphanLocked(const std::string& key, bool immediate);
    void RemoveOrphanLocked(const std::string& key);

    // The live, not dead unit ``id`` resolves to: itself, or through dead
    // units' parent hints and removed units' recorded fallbacks; empty if none.
    std::string ResolveLocked(const std::string& id) const;
    std::string FallbackLocked(const std::string& id, const Unit& unit) const;
    double GainLocked(const std::string& id, const Unit& unit) const;
    uint64_t SizeLocked(const std::string& id, const Unit& unit) const;
    // Up to ``limit`` live units below ``id`` in the hint graph, looking
    // through dead, removed and unknown units.
    std::vector<std::string> HintDescendantsLocked(const std::string& id,
                                                   size_t limit) const;
    bool HasDeeperLocked(const std::string& id, const Unit& unit) const;
    bool IsBranchPointLocked(const std::string& id, const Unit& unit) const;
    // anchor_tier: ``id`` is aligned and is the deepest live aligned unit
    // under a live tail (a live unaligned unit below it in the hint graph,
    // or holding its last page with no deeper live aligned unit holding it).
    bool IsAnchorLocked(const std::string& id, const Unit& unit) const;
    // A unit died: its resolved parent may stop being an anchor.
    void PushParentLocked(const Unit& unit, Clock::time_point now);
    int TierLocked(const std::string& id, const Unit& unit,
                   Clock::time_point now) const;
    std::pair<int, double> RankLocked(const std::string& id, const Unit& unit,
                                      Clock::time_point now) const;
    // Units whose rank may have dropped because ``id`` arrived: deeper units
    // that now fall back to it, and shallower units it makes inner.
    void PushAffectedByInsertLocked(const std::string& id, const Unit& unit,
                                    Clock::time_point now);
    void ProcessDemotionsLocked(Clock::time_point now);
    // Drops the demotions of units that went and those a later use of the
    // same unit superseded, keeping the order.
    void CompactDemotionsLocked();
    void TrimGoneLocked();
    void MaybeRebuildHeapLocked(Clock::time_point now);
    UnitPolicyStats CountersLocked() const;
    // ``key``'s id, interned with one more reference.
    KeyId AddKeyRef(const std::string& key);
    // One reference less; the id is reused once the last one went.
    void DropKeyRef(KeyId key);
    // The id of ``key``, or kNoKey if no unit owns or references it.
    KeyId FindKey(const std::string& key) const;
    const std::string& KeyOf(KeyId key) const { return *keys_[key].name; }

    const UnitPolicyConfig config_;
    const Clock::duration grace_;
    const Clock::duration demote_after_;
    mutable std::mutex mu_;
    double clock_ = 0.0;
    uint64_t seq_ = 0;
    std::unordered_map<std::string, Unit> units_;
    std::unordered_map<std::string, Object> objects_;
    // The bytes of objects_.
    uint64_t object_bytes_ = 0;
    std::unordered_map<std::string, KeyId> key_ids_;
    std::vector<Key> keys_;
    std::vector<KeyId> free_keys_;
    // Object -> the unit that owns it.
    std::unordered_map<KeyId, std::string> member_of_;
    // Page -> the units whose path holds it.
    std::unordered_map<KeyId, std::unordered_set<std::string>> holders_;
    // Last path page -> the aligned units ending on it.
    std::unordered_map<KeyId, std::unordered_set<std::string>> aligned_at_;
    // Unit id (live or not) -> the units whose parent hint names it.
    std::unordered_map<std::string, std::unordered_set<std::string>>
        hint_children_;
    // A unit that went -> its fallback then; bounded, oldest out first.
    std::unordered_map<std::string, std::string> gone_;
    std::deque<std::string> gone_order_;
    // Ids trimmed from gone_ since the last pass over hint_children_, and the
    // entries that pass kept.
    size_t gone_trimmed_ = 0;
    size_t hints_kept_ = 0;
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> heap_;
    // Orphan candidates by the time they become evictable.
    std::set<std::pair<Clock::time_point, std::string>> orphans_;
    std::unordered_map<std::string, Clock::time_point> orphan_at_;
    std::vector<std::string> blocked_units_;
    std::vector<std::string> blocked_objects_;
    // Blocked NextGraceOrphan victims: back with their grace at EndRound.
    std::vector<std::string> blocked_grace_objects_;
    // Recently erased objects, oldest out first: a unit registered after one
    // of its pages or members went can no longer be resumed at.
    std::unordered_set<std::string> erased_;
    std::deque<std::string> erased_order_;
    // (time a leaf demotes, unit id), oldest first.
    std::deque<std::pair<Clock::time_point, std::string>> demotions_;
    UnitPolicyStats counts_;
};

}  // namespace mooncake
