#include "unit_policy.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>

namespace mooncake {

namespace {
// Removed units whose fallback stays recorded, beyond the live units.
constexpr size_t kGoneSlack = 4096;
// The heap is rebuilt past this many entries per live unit, plus the slack.
constexpr size_t kHeapFactor = 4;
constexpr size_t kHeapSlack = 1024;
// The demotion queue is compacted past this many entries per live unit, plus
// the slack.
constexpr size_t kDemotionFactor = 4;
constexpr size_t kDemotionSlack = 4096;
// Hops followed through dead and removed units when resolving a hint.
constexpr int kMaxResolveHops = 64;
// Levels of the hint graph looked through for live descendants.
constexpr int kMaxHintDepth = 8;
// Live hint descendants looked at for an anchor's tail.
constexpr size_t kMaxHintFanout = 16;
// Erased object keys remembered for units that register after them.
constexpr size_t kErasedKept = 1 << 16;

bool RankGrew(int tier, double score, int old_tier, double old_score) {
    if (tier != old_tier) return tier > old_tier;
    return score > old_score + 1e-12 * std::abs(old_score) + 1e-300;
}
}  // namespace

UnitPolicy::UnitPolicy(UnitPolicyConfig config)
    : config_(std::move(config)),
      grace_(std::chrono::duration_cast<Clock::duration>(
          std::chrono::duration<double>(config_.orphan_grace_s))),
      demote_after_(std::chrono::duration_cast<Clock::duration>(
          std::chrono::duration<double>(config_.idle_demote_s))) {}

std::vector<std::string> UnitPolicy::Apply(
    const std::vector<UnitUpdate>& updates, Clock::time_point now) {
    std::vector<std::string> errors;
    std::lock_guard<std::mutex> lock(mu_);
    for (const auto& update : updates) {
        if (update.id.empty()) {
            errors.emplace_back("a unit update without an id");
            continue;
        }
        switch (update.op) {
            case UnitOp::kUpsert:
                if (!std::isfinite(update.value) || update.value < 0.0) {
                    errors.emplace_back("unit " + update.id +
                                        ": value must be a finite number >= 0");
                } else if (update.members.empty()) {
                    errors.emplace_back("unit " + update.id +
                                        ": a unit owns at least one object");
                } else {
                    UpsertLocked(update, now);
                }
                break;
            case UnitOp::kTouch: {
                auto it = units_.find(update.id);
                if (it == units_.end()) {
                    counts_.touch_unknown++;
                    break;
                }
                counts_.touches++;
                it->second.touches++;
                UseLocked(it->second, update.id, now);
                break;
            }
            case UnitOp::kDead: {
                auto it = units_.find(update.id);
                if (it == units_.end()) {
                    counts_.dead_unknown++;
                    break;
                }
                if (!it->second.dead) {
                    it->second.dead = true;
                    counts_.dead_marks++;
                    PushLocked(update.id, now);
                    PushParentLocked(it->second, now);
                }
                break;
            }
        }
    }
    return errors;
}

void UnitPolicy::UpsertLocked(const UnitUpdate& update, Clock::time_point now) {
    counts_.upserts++;
    const std::string& id = update.id;
    auto [it, inserted] = units_.try_emplace(id);
    Unit& unit = it->second;
    if (!inserted) {
        // The same content registered again (another request or engine
        // reached it): a use. Members it did not list before join it.
        for (const auto& member : update.members) {
            const KeyId key = AddKeyRef(member);
            auto [owner, fresh] = member_of_.try_emplace(key, id);
            if (!fresh) {
                DropKeyRef(key);
                if (owner->second != id) counts_.member_conflicts++;
                continue;
            }
            unit.members.push_back(key);
            RemoveOrphanLocked(member);
            if (objects_.contains(member)) unit.present_members++;
        }
        if (unit.parent.empty() && !update.parent.empty() &&
            update.parent != id) {
            unit.parent = update.parent;
            hint_children_[unit.parent].insert(id);
        }
        unit.touches++;
        UseLocked(unit, id, now);
        return;
    }
    gone_.erase(id);
    unit.value = update.value;
    unit.parent = update.parent == id ? std::string() : update.parent;
    unit.registered_at = now;
    for (const auto& member : update.members) {
        const KeyId key = AddKeyRef(member);
        auto [owner, fresh] = member_of_.try_emplace(key, id);
        if (!fresh) {
            DropKeyRef(key);
            if (owner->second != id) counts_.member_conflicts++;
            continue;
        }
        unit.members.push_back(key);
        RemoveOrphanLocked(member);
        if (objects_.contains(member)) unit.present_members++;
    }
    if (unit.members.empty()) {
        // Every member belongs to another unit already.
        units_.erase(it);
        return;
    }
    unit.refs.reserve(update.refs.size());
    for (const auto& page : update.refs) {
        // A page listed twice, or also as a member, would count twice in
        // the unit's size.
        const KeyId key = AddKeyRef(page);
        auto owner = member_of_.find(key);
        if ((owner != member_of_.end() && owner->second == id) ||
            !holders_[key].insert(id).second) {
            DropKeyRef(key);
            counts_.dropped_refs++;
            continue;
        }
        unit.refs.push_back(key);
        RemoveOrphanLocked(page);
        if (!objects_.contains(page) && erased_.contains(page)) {
            unit.dead = true;
            counts_.lost_pages++;
        }
    }
    unit.aligned = update.aligned && !unit.refs.empty();
    for (const KeyId key : unit.members) {
        const std::string& member = KeyOf(key);
        if (!objects_.contains(member) && erased_.contains(member)) {
            unit.dead = true;
            counts_.lost_members++;
        }
    }
    if (unit.aligned) aligned_at_[unit.refs.back()].insert(id);
    if (!unit.parent.empty()) hint_children_[unit.parent].insert(id);
    UseLocked(unit, id, now);
    PushAffectedByInsertLocked(id, unit, now);
}

void UnitPolicy::UseLocked(Unit& unit, const std::string& id,
                           Clock::time_point now) {
    unit.stamp = clock_;
    unit.used_at = now;
    if (unit.evicting) unit.used_while_evicting = true;
    if (config_.leaf_last && config_.idle_demote_s > 0) {
        demotions_.emplace_back(now + demote_after_, id);
        if (demotions_.size() >
            kDemotionFactor * units_.size() + kDemotionSlack) {
            CompactDemotionsLocked();
        }
    }
    PushLocked(id, now);
}

void UnitPolicy::PushLocked(const std::string& id, Clock::time_point now) {
    auto it = units_.find(id);
    if (it == units_.end()) return;
    Unit& unit = it->second;
    if (unit.evicting || unit.parked) return;
    auto [tier, score] = RankLocked(id, unit, now);
    unit.ver = ++seq_;
    unit.pushed_tier = tier;
    heap_.push(Entry{tier, score, seq_, id, unit.ver});
    MaybeRebuildHeapLocked(now);
}

void UnitPolicy::MaybeRebuildHeapLocked(Clock::time_point now) {
    if (heap_.size() <= kHeapFactor * units_.size() + kHeapSlack) return;
    std::vector<Entry> live;
    live.reserve(units_.size());
    while (!heap_.empty()) {
        const Entry& top = heap_.top();
        auto it = units_.find(top.id);
        if (it != units_.end() && it->second.ver == top.ver) {
            live.push_back(top);
        }
        heap_.pop();
    }
    for (auto& entry : live) heap_.push(std::move(entry));
    (void)now;
}

void UnitPolicy::AddOrphanLocked(const std::string& key, bool immediate) {
    auto object = objects_.find(key);
    if (object == objects_.end()) return;
    RemoveOrphanLocked(key);
    const Clock::time_point at =
        immediate ? Clock::time_point{} : object->second.stored_at + grace_;
    orphans_.emplace(at, key);
    orphan_at_[key] = at;
}

void UnitPolicy::RemoveOrphanLocked(const std::string& key) {
    auto it = orphan_at_.find(key);
    if (it == orphan_at_.end()) return;
    orphans_.erase({it->second, key});
    orphan_at_.erase(it);
}

UnitPolicy::KeyId UnitPolicy::AddKeyRef(const std::string& key) {
    auto [it, fresh] = key_ids_.try_emplace(key, kNoKey);
    if (fresh) {
        if (free_keys_.empty()) {
            it->second = static_cast<KeyId>(keys_.size());
            keys_.emplace_back();
        } else {
            it->second = free_keys_.back();
            free_keys_.pop_back();
        }
        keys_[it->second].name = &it->first;
    }
    keys_[it->second].refs++;
    return it->second;
}

void UnitPolicy::DropKeyRef(KeyId key) {
    Key& entry = keys_[key];
    if (--entry.refs > 0) return;
    key_ids_.erase(key_ids_.find(*entry.name));
    entry.name = nullptr;
    free_keys_.push_back(key);
}

UnitPolicy::KeyId UnitPolicy::FindKey(const std::string& key) const {
    auto it = key_ids_.find(key);
    return it == key_ids_.end() ? kNoKey : it->second;
}

std::string UnitPolicy::ResolveLocked(const std::string& start) const {
    std::string id = start;
    for (int hop = 0; hop < kMaxResolveHops && !id.empty(); ++hop) {
        auto it = units_.find(id);
        if (it != units_.end()) {
            if (!it->second.dead) return id;
            id = it->second.parent;
            continue;
        }
        auto gone = gone_.find(id);
        if (gone == gone_.end()) return {};
        id = gone->second;
    }
    return {};
}

std::string UnitPolicy::FallbackLocked(const std::string& id,
                                       const Unit& unit) const {
    std::string best;
    double best_value = -1.0;
    if (!unit.parent.empty()) {
        const std::string parent = ResolveLocked(unit.parent);
        if (!parent.empty() && parent != id) {
            const double value = units_.at(parent).value;
            if (value < unit.value) {
                best = parent;
                best_value = value;
            }
        }
    }
    for (size_t j = unit.refs.size(); j-- > 0;) {
        auto aligned = aligned_at_.find(unit.refs[j]);
        if (aligned == aligned_at_.end()) continue;
        bool found = false;
        for (const auto& other : aligned->second) {
            if (other == id) continue;
            const Unit& candidate = units_.at(other);
            if (candidate.dead || candidate.value >= unit.value) continue;
            found = true;
            if (candidate.value > best_value) {
                best = other;
                best_value = candidate.value;
            }
        }
        if (found) break;
    }
    return best;
}

double UnitPolicy::GainLocked(const std::string& id, const Unit& unit) const {
    if (unit.dead) return 0.0;
    const std::string fallback = FallbackLocked(id, unit);
    const double base = fallback.empty() ? 0.0 : units_.at(fallback).value;
    return std::max(0.0, unit.value - base);
}

uint64_t UnitPolicy::SizeLocked(const std::string& id, const Unit& unit) const {
    uint64_t bytes = 0;
    for (const KeyId member : unit.members) {
        auto it = objects_.find(KeyOf(member));
        if (it != objects_.end()) bytes += it->second.bytes;
    }
    // Reference counts never grow deeper along a path, so the pages only
    // this unit references are a suffix of its path.
    for (size_t j = unit.refs.size(); j-- > 0;) {
        auto holders = holders_.find(unit.refs[j]);
        if (holders == holders_.end() || holders->second.size() != 1 ||
            !holders->second.contains(id)) {
            break;
        }
        auto it = objects_.find(KeyOf(unit.refs[j]));
        if (it != objects_.end()) bytes += it->second.bytes;
    }
    return bytes;
}

std::vector<std::string> UnitPolicy::HintDescendantsLocked(
    const std::string& id, size_t limit) const {
    std::vector<std::string> out;
    std::vector<std::string> frontier{id};
    std::unordered_set<std::string> seen{id};
    for (int depth = 0; depth < kMaxHintDepth && !frontier.empty(); ++depth) {
        std::vector<std::string> next;
        for (const auto& node : frontier) {
            auto children = hint_children_.find(node);
            if (children == hint_children_.end()) continue;
            for (const auto& child : children->second) {
                if (!seen.insert(child).second) continue;
                auto it = units_.find(child);
                if (it != units_.end() && !it->second.dead) {
                    out.push_back(child);
                    if (out.size() >= limit) return out;
                } else {
                    next.push_back(child);
                }
            }
        }
        frontier.swap(next);
    }
    return out;
}

bool UnitPolicy::HasDeeperLocked(const std::string& id,
                                 const Unit& unit) const {
    if (unit.aligned) {
        auto holders = holders_.find(unit.refs.back());
        if (holders != holders_.end()) {
            for (const auto& other : holders->second) {
                if (other == id) continue;
                auto it = units_.find(other);
                if (it != units_.end() && !it->second.dead) return true;
            }
        }
    }
    return !HintDescendantsLocked(id, 1).empty();
}

bool UnitPolicy::IsBranchPointLocked(const std::string& id,
                                     const Unit& unit) const {
    // Resumed at by more than one request (a shared prefix), or two cached
    // continuations part from it.
    if (unit.touches > 1) return true;
    if (HintDescendantsLocked(id, 2).size() >= 2) return true;
    if (!unit.aligned) return false;
    auto holders = holders_.find(unit.refs.back());
    if (holders == holders_.end()) return false;
    const size_t depth = unit.refs.size();
    const KeyId* first = nullptr;
    for (const auto& other : holders->second) {
        if (other == id) continue;
        auto it = units_.find(other);
        if (it == units_.end() || it->second.dead ||
            it->second.refs.size() <= depth) {
            continue;
        }
        const KeyId& next = it->second.refs[depth];
        if (first == nullptr) {
            first = &next;
        } else if (*first != next) {
            return true;
        }
    }
    return false;
}

bool UnitPolicy::IsAnchorLocked(const std::string& id, const Unit& unit) const {
    if (!unit.aligned || unit.refs.empty()) return false;
    // A live tail whose parent chain reaches this unit through units that
    // went: the deepest aligned state under that tail, whatever other paths
    // go on from here.
    for (const auto& child : HintDescendantsLocked(id, kMaxHintFanout)) {
        if (!units_.at(child).aligned) return true;
    }
    // Without hints: a live tail holds this unit's last page and no deeper
    // live aligned unit does (a tail holds the full pages before its own
    // position, so every holder is at least as deep as this unit).
    auto holders = holders_.find(unit.refs.back());
    if (holders == holders_.end()) return false;
    bool tail = false;
    for (const auto& other : holders->second) {
        if (other == id) continue;
        auto it = units_.find(other);
        if (it == units_.end() || it->second.dead) continue;
        if (it->second.aligned) return false;
        tail = true;
    }
    return tail;
}

void UnitPolicy::PushParentLocked(const Unit& unit, Clock::time_point now) {
    if (!config_.anchor_tier || unit.parent.empty()) return;
    const std::string parent = ResolveLocked(unit.parent);
    if (!parent.empty()) PushLocked(parent, now);
}

int UnitPolicy::TierLocked(const std::string& id, const Unit& unit,
                           Clock::time_point now) const {
    if (!config_.leaf_last) return 0;
    if (unit.dead) return kDead;
    if (HasDeeperLocked(id, unit)) {
        if (IsBranchPointLocked(id, unit)) return kLeaf;
        if (config_.anchor_tier && IsAnchorLocked(id, unit)) return kLeaf;
        return kInner;
    }
    if (config_.idle_demote_s > 0 && now - unit.used_at >= demote_after_ &&
        !IsBranchPointLocked(id, unit)) {
        return kInner;
    }
    return kLeaf;
}

std::pair<int, double> UnitPolicy::RankLocked(const std::string& id,
                                              const Unit& unit,
                                              Clock::time_point now) const {
    const int tier = TierLocked(id, unit, now);
    const double gain = GainLocked(id, unit);
    const double size =
        static_cast<double>(std::max<uint64_t>(1, SizeLocked(id, unit)));
    return {tier, unit.stamp + gain / size};
}

void UnitPolicy::PushAffectedByInsertLocked(const std::string& id,
                                            const Unit& unit,
                                            Clock::time_point now) {
    std::vector<std::string> affected;
    if (unit.aligned) {
        auto holders = holders_.find(unit.refs.back());
        if (holders != holders_.end()) {
            for (const auto& other : holders->second) {
                if (other == id) continue;
                auto it = units_.find(other);
                if (it != units_.end() && it->second.value > unit.value) {
                    affected.push_back(other);
                }
            }
        }
    }
    // Units that named this one as their parent before it arrived.
    for (auto& child :
         HintDescendantsLocked(id, std::numeric_limits<size_t>::max())) {
        affected.push_back(std::move(child));
    }
    if (config_.leaf_last) {
        for (const auto& page : unit.refs) {
            auto aligned = aligned_at_.find(page);
            if (aligned == aligned_at_.end()) continue;
            for (const auto& other : aligned->second) {
                if (other == id) continue;
                auto it = units_.find(other);
                if (it != units_.end() && it->second.pushed_tier == kLeaf) {
                    affected.push_back(other);
                }
            }
        }
        const std::string parent = ResolveLocked(unit.parent);
        if (!parent.empty() && parent != id &&
            units_.at(parent).pushed_tier == kLeaf) {
            affected.push_back(parent);
        }
    }
    for (const auto& other : affected) PushLocked(other, now);
}

void UnitPolicy::ProcessDemotionsLocked(Clock::time_point now) {
    while (!demotions_.empty() && demotions_.front().first <= now) {
        const std::string id = std::move(demotions_.front().second);
        demotions_.pop_front();
        auto it = units_.find(id);
        if (it == units_.end() || it->second.pushed_tier != kLeaf) continue;
        if (TierLocked(id, it->second, now) != kLeaf) PushLocked(id, now);
    }
}

void UnitPolicy::CompactDemotionsLocked() {
    // Only the entry of a unit's last use can demote it: keep, in order, the
    // last entry of each unit whose time is still that use's.
    std::unordered_set<const Unit*> seen;
    std::vector<bool> keep(demotions_.size());
    for (size_t i = demotions_.size(); i-- > 0;) {
        const auto& [at, id] = demotions_[i];
        auto it = units_.find(id);
        keep[i] = it != units_.end() &&
                  at == it->second.used_at + demote_after_ &&
                  seen.insert(&it->second).second;
    }
    size_t kept = 0;
    for (size_t i = 0; i < demotions_.size(); ++i) {
        if (!keep[i]) continue;
        if (kept != i) demotions_[kept] = std::move(demotions_[i]);
        kept++;
    }
    demotions_.resize(kept);
}

void UnitPolicy::TrimGoneLocked() {
    while (gone_order_.size() > units_.size() + kGoneSlack) {
        const std::string id = std::move(gone_order_.front());
        gone_order_.pop_front();
        if (!units_.contains(id)) {
            gone_.erase(id);
            gone_trimmed_++;
        }
    }
    // A pass visits every hint entry: one per as many trimmed ids as the last
    // pass kept entries amortizes it. The current count would include the
    // stale entries only a pass drops and could outgrow the trims for good.
    if (gone_trimmed_ < std::max(kGoneSlack, hints_kept_)) return;
    gone_trimmed_ = 0;
    counts_.gone_cleanups++;
    // Hint links through ids that are neither live nor recorded lead nowhere.
    for (auto it = hint_children_.begin(); it != hint_children_.end();) {
        if (!units_.contains(it->first) && !gone_.contains(it->first)) {
            it = hint_children_.erase(it);
            continue;
        }
        for (auto child = it->second.begin(); child != it->second.end();) {
            if (!units_.contains(*child) && !gone_.contains(*child)) {
                child = it->second.erase(child);
            } else {
                ++child;
            }
        }
        if (it->second.empty()) {
            it = hint_children_.erase(it);
        } else {
            ++it;
        }
    }
    hints_kept_ = hint_children_.size();
}

void UnitPolicy::RemoveUnitLocked(const std::string& id,
                                  Clock::time_point now) {
    auto it = units_.find(id);
    if (it == units_.end()) return;
    const std::string fallback = FallbackLocked(id, it->second);
    Unit unit = std::move(it->second);
    units_.erase(it);
    gone_[id] = fallback;
    gone_order_.push_back(id);
    for (const auto& member : unit.members) {
        auto owner = member_of_.find(member);
        if (owner == member_of_.end() || owner->second != id) continue;
        member_of_.erase(owner);
        if (!holders_.contains(member)) AddOrphanLocked(KeyOf(member), true);
    }
    std::vector<std::string> grown;
    for (const auto& page : unit.refs) {
        auto holders = holders_.find(page);
        if (holders == holders_.end()) continue;
        holders->second.erase(id);
        if (holders->second.empty()) {
            holders_.erase(holders);
            if (!member_of_.contains(page)) {
                AddOrphanLocked(KeyOf(page), true);
            }
        } else if (holders->second.size() == 1) {
            // The one unit left on this page frees it when it goes.
            grown.push_back(*holders->second.begin());
        }
    }
    if (unit.aligned) {
        auto aligned = aligned_at_.find(unit.refs.back());
        if (aligned != aligned_at_.end()) {
            aligned->second.erase(id);
            if (aligned->second.empty()) aligned_at_.erase(aligned);
        }
    }
    for (const KeyId key : unit.members) DropKeyRef(key);
    for (const KeyId key : unit.refs) DropKeyRef(key);
    for (const auto& other : grown) PushLocked(other, now);
    TrimGoneLocked();
}

void UnitPolicy::OnObjectStored(const std::string& key, uint64_t bytes,
                                Clock::time_point now) {
    std::lock_guard<std::mutex> lock(mu_);
    auto [it, fresh] = objects_.try_emplace(key);
    object_bytes_ = object_bytes_ - it->second.bytes + bytes;
    it->second.bytes = bytes;
    if (fresh) {
        it->second.stored_at = now;
        erased_.erase(key);
    }
    const KeyId interned = FindKey(key);
    auto owner = member_of_.find(interned);
    if (owner != member_of_.end()) {
        if (fresh) {
            units_.at(owner->second).present_members++;
            PushLocked(owner->second, now);
        }
        return;
    }
    auto holders = holders_.find(interned);
    if (holders != holders_.end()) {
        if (fresh && holders->second.size() == 1) {
            PushLocked(*holders->second.begin(), now);
        }
        return;
    }
    if (fresh) AddOrphanLocked(key, false);
}

void UnitPolicy::OnObjectErased(const std::string& key, Clock::time_point now) {
    std::lock_guard<std::mutex> lock(mu_);
    auto object = objects_.find(key);
    if (object == objects_.end()) return;
    object_bytes_ -= object->second.bytes;
    objects_.erase(object);
    RemoveOrphanLocked(key);
    if (erased_.insert(key).second) {
        erased_order_.push_back(key);
        if (erased_order_.size() > kErasedKept) {
            erased_.erase(erased_order_.front());
            erased_order_.pop_front();
        }
    }
    std::vector<std::string> dead;
    const KeyId interned = FindKey(key);
    auto owner = member_of_.find(interned);
    if (owner != member_of_.end()) {
        Unit& unit = units_.at(owner->second);
        if (unit.present_members > 0) unit.present_members--;
        if (!unit.evicting && !unit.dead) {
            counts_.lost_members++;
            dead.push_back(owner->second);
        }
    }
    auto holders = holders_.find(interned);
    if (holders != holders_.end()) {
        for (const auto& id : holders->second) {
            const Unit& unit = units_.at(id);
            if (!unit.evicting && !unit.dead) {
                counts_.lost_pages++;
                dead.push_back(id);
            }
        }
    }
    // A unit that lost a member or a path page can no longer be resumed at.
    for (const auto& id : dead) {
        units_.at(id).dead = true;
        PushLocked(id, now);
        PushParentLocked(units_.at(id), now);
    }
}

std::optional<UnitPolicy::Victim> UnitPolicy::NextVictim(Clock::time_point now,
                                                         bool dead_only) {
    std::lock_guard<std::mutex> lock(mu_);
    ProcessDemotionsLocked(now);
    while (!orphans_.empty()) {
        auto first = orphans_.begin();
        if (first->first > now) break;
        const std::string key = first->second;
        orphans_.erase(first);
        orphan_at_.erase(key);
        const KeyId interned = FindKey(key);
        if (!objects_.contains(key) || member_of_.contains(interned) ||
            holders_.contains(interned)) {
            continue;
        }
        Victim victim;
        victim.orphan = true;
        victim.headroom = dead_only;
        victim.keys.push_back(key);
        return victim;
    }
    while (!heap_.empty()) {
        Entry top = heap_.top();
        heap_.pop();
        auto it = units_.find(top.id);
        if (it == units_.end() || it->second.ver != top.ver ||
            it->second.evicting || it->second.parked) {
            continue;
        }
        Unit& unit = it->second;
        if (!unit.dead && unit.present_members < unit.members.size()) {
            if (now - unit.registered_at < grace_) {
                // Another rank's members are still arriving.
                unit.parked = true;
                blocked_units_.push_back(top.id);
                continue;
            }
            unit.dead = true;
            counts_.pending_expired++;
            PushParentLocked(unit, now);
        }
        auto [tier, score] = RankLocked(top.id, unit, now);
        if (RankGrew(tier, score, top.tier, top.score)) {
            unit.ver = ++seq_;
            unit.pushed_tier = tier;
            heap_.push(Entry{tier, score, seq_, top.id, unit.ver});
            continue;
        }
        if (dead_only && !unit.dead) {
            // Leaf-last order: every unit below this one is live too.
            heap_.push(top);
            return std::nullopt;
        }
        unit.evicting = true;
        unit.used_while_evicting = false;
        Victim victim;
        victim.unit = top.id;
        victim.tier = tier;
        victim.score = score;
        victim.headroom = dead_only;
        for (const auto& member : unit.members) {
            const std::string& key = KeyOf(member);
            if (objects_.contains(key)) victim.keys.push_back(key);
        }
        return victim;
    }
    return std::nullopt;
}

std::optional<UnitPolicy::Victim> UnitPolicy::NextGraceOrphan() {
    std::lock_guard<std::mutex> lock(mu_);
    while (!orphans_.empty()) {
        const std::string key = orphans_.begin()->second;
        orphans_.erase(orphans_.begin());
        orphan_at_.erase(key);
        const KeyId interned = FindKey(key);
        if (!objects_.contains(key) || member_of_.contains(interned) ||
            holders_.contains(interned)) {
            continue;
        }
        Victim victim;
        victim.orphan = true;
        victim.grace = true;
        victim.keys.push_back(key);
        return victim;
    }
    return std::nullopt;
}

bool UnitPolicy::StillEvictable(const Victim& victim, const std::string& key) {
    std::lock_guard<std::mutex> lock(mu_);
    const KeyId interned = FindKey(key);
    if (victim.orphan) {
        return !member_of_.contains(interned) && !holders_.contains(interned);
    }
    auto owner = member_of_.find(interned);
    if (owner == member_of_.end() || owner->second != victim.unit) {
        return false;
    }
    auto it = units_.find(victim.unit);
    return it != units_.end() && it->second.evicting &&
           !it->second.used_while_evicting;
}

void UnitPolicy::Evicted(const Victim& victim, bool complete, uint64_t bytes,
                         Clock::time_point now) {
    std::lock_guard<std::mutex> lock(mu_);
    if (victim.orphan) {
        if (victim.grace) {
            counts_.grace_orphans++;
            counts_.grace_orphan_bytes += bytes;
        }
        counts_.orphans++;
        counts_.orphan_bytes += bytes;
        if (victim.headroom) {
            counts_.headroom_orphans++;
            counts_.headroom_orphan_bytes += bytes;
        }
        return;
    }
    auto it = units_.find(victim.unit);
    if (it == units_.end()) return;
    it->second.evicting = false;
    if (!complete) {
        counts_.partial++;
        it->second.dead = true;
        PushLocked(victim.unit, now);
        PushParentLocked(it->second, now);
        return;
    }
    if (!config_.leaf_last || victim.tier == kLeaf) {
        clock_ = std::max(clock_, victim.score);
    }
    counts_.evicted_units++;
    counts_.evicted_unit_bytes += bytes;
    if (config_.leaf_last) counts_.evicted_by_tier[victim.tier]++;
    if (victim.headroom) {
        counts_.headroom_units++;
        counts_.headroom_unit_bytes += bytes;
    }
    RemoveUnitLocked(victim.unit, now);
}

void UnitPolicy::Blocked(const Victim& victim) {
    std::lock_guard<std::mutex> lock(mu_);
    counts_.blocked++;
    if (victim.orphan) {
        (victim.grace ? blocked_grace_objects_ : blocked_objects_)
            .push_back(victim.keys.front());
        return;
    }
    auto it = units_.find(victim.unit);
    if (it == units_.end()) return;
    it->second.evicting = false;
    it->second.parked = true;
    blocked_units_.push_back(victim.unit);
}

void UnitPolicy::EndRound(Clock::time_point now) {
    std::lock_guard<std::mutex> lock(mu_);
    std::vector<std::string> units;
    units.swap(blocked_units_);
    for (const auto& id : units) {
        auto it = units_.find(id);
        if (it == units_.end()) continue;
        it->second.parked = false;
        PushLocked(id, now);
    }
    std::vector<std::string> objects;
    objects.swap(blocked_objects_);
    for (const auto& key : objects) {
        const KeyId interned = FindKey(key);
        if (!member_of_.contains(interned) && !holders_.contains(interned)) {
            AddOrphanLocked(key, true);
        }
    }
    objects.clear();
    objects.swap(blocked_grace_objects_);
    for (const auto& key : objects) {
        const KeyId interned = FindKey(key);
        if (!member_of_.contains(interned) && !holders_.contains(interned)) {
            AddOrphanLocked(key, false);
        }
    }
}

UnitPolicyStats UnitPolicy::Stats(Clock::time_point now) const {
    std::lock_guard<std::mutex> lock(mu_);
    UnitPolicyStats stats = CountersLocked();
    for (const auto& [id, unit] : units_) {
        switch (TierLocked(id, unit, now)) {
            case kDead:
                stats.units_dead++;
                break;
            case kInner:
                stats.units_inner++;
                break;
            default:
                stats.units_leaf++;
                if (config_.anchor_tier && HasDeeperLocked(id, unit) &&
                    IsAnchorLocked(id, unit)) {
                    stats.units_anchor++;
                }
                break;
        }
    }
    return stats;
}

UnitPolicyStats UnitPolicy::Counters(Clock::time_point now) const {
    std::lock_guard<std::mutex> lock(mu_);
    return CountersLocked();
}

UnitPolicyStats UnitPolicy::CountersLocked() const {
    UnitPolicyStats stats = counts_;
    stats.units = units_.size();
    stats.objects = objects_.size();
    stats.object_bytes = object_bytes_;
    stats.orphan_candidates = orphans_.size();
    stats.heap_entries = heap_.size();
    stats.clock = clock_;
    stats.demotion_entries = demotions_.size();
    return stats;
}

std::optional<std::pair<int, double>> UnitPolicy::RankOf(
    const std::string& id, Clock::time_point now) const {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = units_.find(id);
    if (it == units_.end()) return std::nullopt;
    return RankLocked(id, it->second, now);
}

std::vector<std::string> UnitPolicy::Check() const {
    std::lock_guard<std::mutex> lock(mu_);
    std::vector<std::string> out;
    auto note = [&out](const std::string& message) {
        if (out.size() < 64) out.push_back(message);
    };
    // A key id's string, also for an id a broken count freed.
    auto name = [this](KeyId key) {
        return key < keys_.size() && keys_[key].name != nullptr
                   ? *keys_[key].name
                   : "<freed key " + std::to_string(key) + ">";
    };
    std::unordered_map<std::string, std::unordered_set<KeyId>> members;
    std::unordered_map<std::string, std::unordered_set<KeyId>> refs;
    for (const auto& [id, unit] : units_) {
        members[id].insert(unit.members.begin(), unit.members.end());
        refs[id].insert(unit.refs.begin(), unit.refs.end());
    }
    for (const auto& [id, unit] : units_) {
        size_t present = 0;
        for (const auto& member : unit.members) {
            auto owner = member_of_.find(member);
            if (owner == member_of_.end() || owner->second != id) {
                note("unit " + id + ": member " + name(member) +
                     " not owned by it");
            }
            if (objects_.contains(name(member))) present++;
        }
        if (present != unit.present_members) {
            note("unit " + id + ": present member count " +
                 std::to_string(unit.present_members) + ", actual " +
                 std::to_string(present));
        }
        for (const auto& page : unit.refs) {
            auto holders = holders_.find(page);
            if (holders == holders_.end() || !holders->second.contains(id)) {
                note("unit " + id + ": page " + name(page) +
                     " does not count it as a holder");
            }
        }
        if (unit.aligned) {
            auto aligned = aligned_at_.find(unit.refs.back());
            if (aligned == aligned_at_.end() || !aligned->second.contains(id)) {
                note("aligned unit " + id + " missing from its last page");
            }
        }
    }
    for (const auto& [member, id] : member_of_) {
        auto it = members.find(id);
        if (it == members.end() || !it->second.contains(member)) {
            note("object " + name(member) + " owned by " + id +
                 ", which does not list it");
        }
    }
    for (const auto& [page, holders] : holders_) {
        if (holders.empty()) {
            note("page " + name(page) + " has an empty holder set");
        }
        for (const auto& id : holders) {
            auto it = refs.find(id);
            if (it == refs.end() || !it->second.contains(page)) {
                note("page " + name(page) + " held by " + id +
                     ", which does not reference it");
            }
        }
    }
    for (const auto& [page, ids] : aligned_at_) {
        for (const auto& id : ids) {
            auto it = units_.find(id);
            if (it == units_.end() || !it->second.aligned ||
                it->second.refs.back() != page) {
                note("aligned index " + name(page) + " lists " + id);
            }
        }
    }
    for (const auto& [key, at] : orphan_at_) {
        if (!objects_.contains(key)) note("orphan " + key + " is not stored");
        if (!orphans_.contains({at, key})) note("orphan index lost " + key);
    }
    if (orphans_.size() != orphan_at_.size()) {
        note("orphan set and index sizes differ");
    }
    std::unordered_map<KeyId, uint32_t> held;
    for (const auto& [id, unit] : units_) {
        for (const KeyId key : unit.members) held[key]++;
        for (const KeyId key : unit.refs) held[key]++;
    }
    for (const auto& [key, interned] : key_ids_) {
        auto it = held.find(interned);
        const uint32_t count = it == held.end() ? 0 : it->second;
        if (keys_[interned].refs != count) {
            note("key " + key + " counts " +
                 std::to_string(keys_[interned].refs) +
                 " references, units hold " + std::to_string(count));
        }
    }
    if (held.size() != key_ids_.size()) {
        note("units hold keys that are not interned");
    }
    return out;
}

void UnitPolicy::DropPageRefForTesting(const std::string& id) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = units_.find(id);
    if (it == units_.end() || it->second.refs.empty()) return;
    holders_[it->second.refs.back()].erase(id);
}

}  // namespace mooncake
