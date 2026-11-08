#include "unit_policy.h"

#include <gtest/gtest.h>

#include <map>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace mooncake::test {
namespace {

using Clock = UnitPolicy::Clock;
using namespace std::chrono_literals;

constexpr uint64_t kObject = 1000;

UnitUpdate Upsert(const std::string& id, double value,
                  std::vector<std::string> members,
                  std::vector<std::string> refs, bool aligned = false,
                  const std::string& parent = "") {
    UnitUpdate update;
    update.op = UnitOp::kUpsert;
    update.id = id;
    update.value = value;
    update.members = std::move(members);
    update.refs = std::move(refs);
    update.aligned = aligned;
    update.parent = parent;
    return update;
}

UnitUpdate Op(UnitOp op, const std::string& id) {
    UnitUpdate update;
    update.op = op;
    update.id = id;
    return update;
}

class UnitPolicyTest : public ::testing::Test {
   protected:
    UnitPolicyConfig Config(bool leaf_last = true, double demote_s = 0.0,
                            double grace_s = 30.0) {
        UnitPolicyConfig config;
        config.enabled = true;
        config.leaf_last = leaf_last;
        config.idle_demote_s = demote_s;
        config.orphan_grace_s = grace_s;
        return config;
    }

    void Store(UnitPolicy& policy, const std::vector<std::string>& keys,
               uint64_t bytes = kObject) {
        for (const auto& key : keys) policy.OnObjectStored(key, bytes, now_);
    }

    void Apply(UnitPolicy& policy, std::vector<UnitUpdate> updates) {
        EXPECT_TRUE(policy.Apply(updates, now_).empty());
    }

    // Evicts the next ``n`` victims as the master does (every key erased,
    // none leased), then ends the round; returns the evicted units' ids and
    // "orphan:<key>" for orphans, in order.
    std::vector<std::string> Evict(UnitPolicy& policy, size_t n = 1000) {
        std::vector<std::string> order;
        while (order.size() < n) {
            auto victim = policy.NextVictim(now_);
            if (!victim) break;
            uint64_t bytes = 0;
            for (const auto& key : victim->keys) {
                EXPECT_TRUE(policy.StillEvictable(*victim, key)) << key;
                policy.OnObjectErased(key, now_);
                bytes += kObject;
            }
            policy.Evicted(*victim, true, bytes, now_);
            order.push_back(victim->orphan ? "orphan:" + victim->keys.front()
                                           : victim->unit);
        }
        policy.EndRound(now_);
        return order;
    }

    Clock::time_point now_ = Clock::now();
};

TEST_F(UnitPolicyTest, GreedyDualSizeEvictsTheLowestGainPerByteFirst) {
    UnitPolicy policy(Config(false));
    // Same size; b saves three times the recompute of a.
    Store(policy, {"a.m", "b.m"});
    Apply(policy,
          {Upsert("a", 1.0, {"a.m"}, {}), Upsert("b", 3.0, {"b.m"}, {})});
    EXPECT_EQ(Evict(policy, 1), std::vector<std::string>{"a"});
    // The clock rose to a's score: b keeps its own.
    EXPECT_NEAR(policy.Stats(now_).clock, 1.0 / kObject, 1e-12);
}

TEST_F(UnitPolicyTest, SizeCountsOnlyExclusivePagesAndPagesFollowTheLastUnit) {
    UnitPolicy policy(Config(false));
    // a and b share p0; each owns one member, b also owns p1 on its path.
    Store(policy, {"p0", "p1", "a.m", "b.m"});
    Apply(policy, {Upsert("a", 2.0, {"a.m"}, {"p0"}),
                   Upsert("b", 2.0, {"b.m"}, {"p0", "p1"})});
    // b frees two objects (its member and p1), a only its member.
    EXPECT_LT(policy.RankOf("b", now_)->second,
              policy.RankOf("a", now_)->second);
    auto order = Evict(policy, 1);
    ASSERT_EQ(order, std::vector<std::string>{"b"});
    // p1 followed b at once; p0 stays for a, whose size now includes p0 (a
    // keeps the stamp of its registration).
    order = Evict(policy, 1);
    EXPECT_EQ(order, std::vector<std::string>{"orphan:p1"});
    EXPECT_NEAR(policy.RankOf("a", now_)->second, 2.0 / (2 * kObject), 1e-12);
    order = Evict(policy);
    EXPECT_EQ(order, (std::vector<std::string>{"a", "orphan:p0"}));
    EXPECT_TRUE(policy.Check().empty());
}

TEST_F(UnitPolicyTest, GainIsMeasuredFromTheDeepestLiveFallback) {
    UnitPolicy policy(Config(false));
    Store(policy, {"p0", "p1", "s1.m", "s2.m", "t.m"});
    // s1 at block 1, s2 at block 2 (aligned), t a tail inside block 2.
    Apply(policy, {Upsert("s1", 1.0, {"s1.m"}, {"p0"}, true),
                   Upsert("s2", 3.0, {"s2.m"}, {"p0", "p1"}, true, "s1"),
                   Upsert("t", 3.5, {"t.m"}, {"p0", "p1"}, false, "s2")});
    const double clock = policy.Stats(now_).clock;
    // t falls back to s2: it saves 0.5 s over one member.
    EXPECT_NEAR(policy.RankOf("t", now_)->second, clock + 0.5 / kObject, 1e-12);
    // Without s2 it falls back to s1 (its hint resolves through s2) and also
    // frees p1, which it alone references then.
    Apply(policy, {Op(UnitOp::kDead, "s2")});
    EXPECT_EQ(Evict(policy, 1), std::vector<std::string>{"s2"});
    EXPECT_NEAR(policy.RankOf("t", now_)->second, clock + 2.5 / (2 * kObject),
                1e-12);
}

TEST_F(UnitPolicyTest, LeafLastEvictsInnerUnitsBeforeAnyLeaf) {
    UnitPolicy policy(Config(true));
    Store(policy, {"p0", "s.m", "t.m", "x.m"});
    // s (aligned at block 1) is inner: the tail t is deeper on its path. s has
    // the highest gain per byte, x (another session's leaf) the lowest.
    Apply(policy, {Upsert("s", 9.0, {"s.m"}, {"p0"}, true),
                   Upsert("t", 9.2, {"t.m"}, {"p0"}, false, "s"),
                   Upsert("x", 0.1, {"x.m"}, {})});
    EXPECT_EQ(policy.RankOf("s", now_)->first, 1);
    EXPECT_EQ(policy.RankOf("t", now_)->first, 2);
    EXPECT_EQ(policy.RankOf("x", now_)->first, 2);
    auto order = Evict(policy, 2);
    EXPECT_EQ(order, (std::vector<std::string>{"s", "x"}));
}

// anchor_tier: the deepest live aligned unit under a live tail ranks with the
// leaves. Each case also runs without the tier, where the same unit is inner:
// the positive control that the case can tell the two orders apart.
TEST_F(UnitPolicyTest, AnAnchorRanksWithTheLeaves) {
    for (bool anchors : {true, false}) {
        auto config = Config(true);
        config.anchor_tier = anchors;
        UnitPolicy policy(config);
        Store(policy, {"p0", "p1", "b1.m", "b2.m", "t.m", "x.m"});
        // b1 and b2 end pages 0 and 1; the tail t lies after b2 (its parent);
        // x is another session's leaf with the lowest gain per byte.
        Apply(policy, {Upsert("b1", 3.0, {"b1.m"}, {"p0"}, true),
                       Upsert("b2", 8.0, {"b2.m"}, {"p0", "p1"}, true, "b1"),
                       Upsert("t", 9.0, {"t.m"}, {"p0", "p1"}, false, "b2"),
                       Upsert("x", 0.1, {"x.m"}, {})});
        EXPECT_EQ(policy.RankOf("b1", now_)->first, 1);
        EXPECT_EQ(policy.RankOf("b2", now_)->first, anchors ? 2 : 1);
        EXPECT_EQ(policy.Stats(now_).units_anchor, anchors ? 1u : 0u);
        EXPECT_EQ(Evict(policy, 2),
                  (anchors ? std::vector<std::string>{"b1", "x"}
                           : std::vector<std::string>{"b1", "b2"}))
            << "anchors " << anchors;
        EXPECT_TRUE(policy.Check().empty());
    }
}

TEST_F(UnitPolicyTest, AnAnchorIsFoundThroughDeadTailsAndAcrossBranches) {
    // b ends page 0; e, registered without a parent hint, continues b on one
    // path (page pa); the tail t2 lies on another path after b, its parent t1
    // a superseded tail marked dead. Only the parent chain shows b is t2's
    // anchor: e, a deeper aligned unit, holds b's last page too, and b is no
    // branch point (one hinted descendant, one next page). Without the chain
    // (t2 names no parent) b is inner, the control.
    for (bool chain : {true, false}) {
        auto config = Config(true);
        config.anchor_tier = true;
        UnitPolicy policy(config);
        Store(policy, {"p0", "pa", "b.m", "e.m", "t1.m", "t2.m"});
        Apply(policy,
              {Upsert("b", 4.0, {"b.m"}, {"p0"}, true),
               Upsert("e", 8.0, {"e.m"}, {"p0", "pa"}, true),
               Upsert("t1", 5.0, {"t1.m"}, {"p0"}, false, "b"),
               Upsert("t2", 6.0, {"t2.m"}, {"p0"}, false, chain ? "t1" : "")});
        Apply(policy, {Op(UnitOp::kDead, "t1")});
        EXPECT_EQ(policy.RankOf("b", now_)->first, chain ? 2 : 1)
            << "chain " << chain;
    }
}

TEST_F(UnitPolicyTest, TheAnchorMovesDownWithTheSession) {
    auto config = Config(true);
    config.anchor_tier = true;
    UnitPolicy policy(config);
    Store(policy, {"p0", "p1", "p2", "b1.m", "b2.m", "t.m", "b3.m", "u.m"});
    Apply(policy, {Upsert("b1", 2.0, {"b1.m"}, {"p0"}, true),
                   Upsert("b2", 6.0, {"b2.m"}, {"p0", "p1"}, true, "b1"),
                   Upsert("t", 7.0, {"t.m"}, {"p0", "p1"}, false, "b2")});
    EXPECT_EQ(policy.RankOf("b2", now_)->first, 2);
    // The next turn resumes at t, stores b3 at the end of page 2 and the new
    // tail u after it, and marks t dead: b3 is the anchor, b2 inner again.
    Apply(policy, {Upsert("b3", 9.0, {"b3.m"}, {"p0", "p1", "p2"}, true, "t"),
                   Upsert("u", 10.0, {"u.m"}, {"p0", "p1", "p2"}, false, "b3"),
                   Op(UnitOp::kDead, "t")});
    EXPECT_EQ(policy.RankOf("b2", now_)->first, 1);
    EXPECT_EQ(policy.RankOf("b3", now_)->first, 2);
    // The heap agrees: the dead tail, then the two inner units, before b3.
    EXPECT_EQ(Evict(policy, 3), (std::vector<std::string>{"t", "b1", "b2"}));
    EXPECT_TRUE(policy.Check().empty());
}

// A dead-only round (dead_headroom_ratio) takes orphans and dead units and
// stops at the first live unit; a full round, the control, takes live units.
TEST_F(UnitPolicyTest, ADeadOnlyRoundTakesNoLiveUnit) {
    auto config = Config(true, 0.0, 0.0);
    UnitPolicy policy(config);
    Store(policy, {"a.m", "b.m", "c.m", "stray"});
    Apply(policy, {Upsert("a", 1.0, {"a.m"}, {}), Upsert("b", 5.0, {"b.m"}, {}),
                   Upsert("c", 2.0, {"c.m"}, {})});
    Apply(policy, {Op(UnitOp::kDead, "b")});
    std::vector<std::string> order;
    while (auto victim = policy.NextVictim(now_, /*dead_only=*/true)) {
        EXPECT_TRUE(victim->headroom);
        for (const auto& key : victim->keys) policy.OnObjectErased(key, now_);
        policy.Evicted(*victim, true, kObject, now_);
        order.push_back(victim->orphan ? "orphan:" + victim->keys.front()
                                       : victim->unit);
    }
    policy.EndRound(now_);
    EXPECT_EQ(order, (std::vector<std::string>{"orphan:stray", "b"}));
    const auto stats = policy.Stats(now_);
    EXPECT_EQ(stats.headroom_units, 1u);
    EXPECT_EQ(stats.headroom_unit_bytes, kObject);
    EXPECT_EQ(stats.headroom_orphans, 1u);
    EXPECT_EQ(stats.evicted_units, 1u);
    EXPECT_EQ(Evict(policy), (std::vector<std::string>{"a", "c"}));
    EXPECT_EQ(policy.Stats(now_).headroom_units, 1u);
    EXPECT_TRUE(policy.Check().empty());
}

TEST_F(UnitPolicyTest, DeadUnitsGoFirstAndOrphansBeforeThem) {
    UnitPolicy policy(Config(true));
    Store(policy, {"a.m", "b.m", "stray"});
    Apply(policy,
          {Upsert("a", 1.0, {"a.m"}, {}), Upsert("b", 5.0, {"b.m"}, {})});
    Apply(policy, {Op(UnitOp::kDead, "b")});
    // The stray object is an orphan only after its grace period.
    EXPECT_EQ(Evict(policy, 1), std::vector<std::string>{"b"});
    now_ += 31s;
    EXPECT_EQ(Evict(policy, 2),
              (std::vector<std::string>{"orphan:stray", "a"}));
}

TEST_F(UnitPolicyTest, ATouchRefreshesOnlyTheTouchedUnit) {
    UnitPolicy policy(Config(false));
    Store(policy, {"a.m", "b.m", "c.m", "d.m"});
    Apply(policy,
          {Upsert("a", 1.0, {"a.m"}, {}), Upsert("b", 1.0, {"b.m"}, {}),
           Upsert("c", 2.0, {"c.m"}, {}), Upsert("d", 1.5, {"d.m"}, {})});
    ASSERT_EQ(Evict(policy, 1), std::vector<std::string>{"a"});
    // The clock is now a's score; touching b stamps it there, above d.
    Apply(policy, {Op(UnitOp::kTouch, "b")});
    EXPECT_EQ(Evict(policy, 2), (std::vector<std::string>{"d", "c"}));
    EXPECT_EQ(policy.Stats(now_).touches, 1u);
}

TEST_F(UnitPolicyTest, ABlockedUnitWaitsForTheNextRound) {
    UnitPolicy policy(Config(false));
    Store(policy, {"a.m", "b.m"});
    Apply(policy,
          {Upsert("a", 1.0, {"a.m"}, {}), Upsert("b", 2.0, {"b.m"}, {})});
    auto victim = policy.NextVictim(now_);
    ASSERT_TRUE(victim && victim->unit == "a");
    policy.Blocked(*victim);
    victim = policy.NextVictim(now_);
    ASSERT_TRUE(victim && victim->unit == "b");
    policy.Blocked(*victim);
    EXPECT_FALSE(policy.NextVictim(now_).has_value());
    policy.EndRound(now_);
    victim = policy.NextVictim(now_);
    ASSERT_TRUE(victim && victim->unit == "a");
}

TEST_F(UnitPolicyTest, AUnitWaitsForItsMembersThenDiesAfterTheGrace) {
    UnitPolicy policy(Config(true, 0.0, 30.0));
    Store(policy, {"u.m0"});
    Apply(policy, {Upsert("u", 4.0, {"u.m0", "u.m1"}, {})});
    EXPECT_FALSE(policy.NextVictim(now_).has_value());
    policy.EndRound(now_);
    now_ += 31s;
    auto victim = policy.NextVictim(now_);
    ASSERT_TRUE(victim && victim->unit == "u");
    EXPECT_EQ(victim->tier, 0);
    EXPECT_EQ(victim->keys, std::vector<std::string>{"u.m0"});
    EXPECT_EQ(policy.Stats(now_).pending_expired, 1u);
}

TEST_F(UnitPolicyTest, LosingAMemberOrAPathPageKillsTheUnit) {
    UnitPolicy policy(Config(true));
    Store(policy, {"p0", "a.m", "b.m"});
    Apply(policy,
          {Upsert("a", 4.0, {"a.m"}, {"p0"}), Upsert("b", 4.0, {"b.m"}, {})});
    policy.OnObjectErased("p0", now_);
    policy.OnObjectErased("b.m", now_);
    EXPECT_EQ(policy.RankOf("a", now_)->first, 0);
    EXPECT_EQ(policy.RankOf("b", now_)->first, 0);
    const auto stats = policy.Stats(now_);
    EXPECT_EQ(stats.lost_pages, 1u);
    EXPECT_EQ(stats.lost_members, 1u);
}

TEST_F(UnitPolicyTest, AUnitRegisteredAfterItsPageWentIsDead) {
    UnitPolicy policy(Config(true));
    Store(policy, {"p0", "p1", "a.m", "b.m"});
    Apply(policy, {Upsert("a", 1.0, {"a.m"}, {"p0"})});
    policy.OnObjectErased("p0", now_);  // a goes dead; p0 was erased
    // b registers after p0 went (its parent's eviction raced its puts): dead at
    // once. A page not yet stored (another rank still writing it) is not a
    // loss.
    Apply(policy, {Upsert("b", 5.0, {"b.m"}, {"p0", "p1"}),
                   Upsert("c", 5.0, {"c.m"}, {"p2"})});
    EXPECT_EQ(policy.RankOf("b", now_)->first, 0);
    EXPECT_EQ(policy.RankOf("c", now_)->first, 2);
    EXPECT_EQ(policy.Stats(now_).lost_pages, 2u);
}

TEST_F(UnitPolicyTest, AnIdleLeafRanksAsInner) {
    UnitPolicy policy(Config(true, 60.0));
    Store(policy, {"a.m", "b.m"});
    Apply(policy, {Upsert("a", 1.0, {"a.m"}, {})});
    now_ += 50s;
    Apply(policy, {Upsert("b", 9.0, {"b.m"}, {})});
    now_ += 20s;
    EXPECT_EQ(policy.RankOf("a", now_)->first, 1);
    EXPECT_EQ(policy.RankOf("b", now_)->first, 2);
    EXPECT_EQ(Evict(policy, 1), std::vector<std::string>{"a"});
}

TEST_F(UnitPolicyTest, ABranchPointRanksAsALeaf) {
    UnitPolicy policy(Config(true));
    Store(policy,
          {"p0", "p1a", "p1b", "p9", "s.m", "a.m", "b.m", "c.m", "q.m"});
    // s (block 1) is where two sessions' paths part: p1a against p1b.
    Apply(policy, {Upsert("s", 1.0, {"s.m"}, {"p0"}, true),
                   Upsert("a", 2.0, {"a.m"}, {"p0", "p1a"}, true, "s"),
                   Upsert("b", 2.0, {"b.m"}, {"p0", "p1b"}, true, "s")});
    EXPECT_EQ(policy.RankOf("s", now_)->first, 2);
    // A unit resumed at by more than one request is a branch point too.
    Apply(policy, {Upsert("c", 1.0, {"c.m"}, {"p9"}, true),
                   Upsert("q", 1.5, {"q.m"}, {"p9"}, false, "c")});
    EXPECT_EQ(policy.RankOf("c", now_)->first, 1);
    Apply(policy, {Op(UnitOp::kTouch, "c"), Op(UnitOp::kTouch, "c")});
    EXPECT_EQ(policy.RankOf("c", now_)->first, 2);
}

TEST_F(UnitPolicyTest, RegisteringTheSameUnitAgainIsAUse) {
    UnitPolicy policy(Config(false));
    Store(policy, {"z.m", "a.m", "b.m"});
    Apply(policy, {Upsert("z", 0.5, {"z.m"}, {}), Upsert("a", 1.0, {"a.m"}, {}),
                   Upsert("b", 1.2, {"b.m"}, {})});
    ASSERT_EQ(Evict(policy, 1), std::vector<std::string>{"z"});
    // Registered again after the clock rose: a is stamped above b.
    Apply(policy, {Upsert("a", 1.0, {"a.m"}, {})});
    EXPECT_EQ(policy.Stats(now_).units, 2u);
    EXPECT_EQ(Evict(policy, 1), std::vector<std::string>{"b"});
}

TEST_F(UnitPolicyTest, ATouchDuringAnEvictionStopsIt) {
    UnitPolicy policy(Config(false));
    Store(policy, {"a.m"});
    Apply(policy, {Upsert("a", 1.0, {"a.m"}, {})});
    auto victim = policy.NextVictim(now_);
    ASSERT_TRUE(victim.has_value());
    Apply(policy, {Op(UnitOp::kTouch, "a")});
    EXPECT_FALSE(policy.StillEvictable(*victim, "a.m"));
}

TEST_F(UnitPolicyTest, InvalidUpdatesAreRejected) {
    UnitPolicy policy(Config(false));
    auto errors =
        policy.Apply({Upsert("", 1.0, {"m"}, {}), Upsert("a", -1.0, {"m"}, {}),
                      Upsert("b", 1.0, {}, {})},
                     now_);
    EXPECT_EQ(errors.size(), 3u);
    EXPECT_EQ(policy.Stats(now_).units, 0u);
}

TEST_F(UnitPolicyTest, CheckReportsACorruptedReferenceCount) {
    UnitPolicy policy(Config(true, 0.0, 30.0));
    Store(policy, {"p0", "a.m"});
    Apply(policy, {Upsert("a", 1.0, {"a.m"}, {"p0"})});
    EXPECT_TRUE(policy.Check().empty());
    policy.DropPageRefForTesting("a");
    const auto violations = policy.Check();
    ASSERT_FALSE(violations.empty());
    EXPECT_NE(violations.front().find("does not count it as a holder"),
              std::string::npos);
}

// Long after more units went than stay live, the parent hints of the units
// that went are cleaned up once per batch of removals, not on every removal.
TEST_F(UnitPolicyTest, HintsOfGoneUnitsAreCleanedUpInBatches) {
    UnitPolicy policy(Config(false));
    constexpr int kLive = 20000;
    constexpr int kRound = 4000;
    int next = 0;
    // Each unit names the one registered before it as its parent.
    auto add = [&](int n) {
        std::vector<UnitUpdate> updates;
        for (int i = 0; i < n; ++i, ++next) {
            const std::string id = "u" + std::to_string(next);
            policy.OnObjectStored(id + ".m", kObject, now_);
            updates.push_back(
                Upsert(id, 1.0, {id + ".m"}, {}, false,
                       next > 0 ? "u" + std::to_string(next - 1) : ""));
        }
        Apply(policy, std::move(updates));
    };
    add(kLive);
    uint64_t removed = 0;
    for (int round = 0; round < 20; ++round) {
        add(kRound);
        ASSERT_EQ(Evict(policy, kRound).size(), static_cast<size_t>(kRound));
        removed += kRound;
        ASSERT_LE(policy.Stats(now_).gone_cleanups, removed / 4096 + 1)
            << "after " << removed << " removals";
    }
    EXPECT_GE(policy.Stats(now_).gone_cleanups, 1u);
    EXPECT_EQ(policy.Stats(now_).units, static_cast<uint64_t>(kLive));
    EXPECT_TRUE(policy.Check().empty());
}

// Below the watermark no eviction round drains the idle demotions: uses keep
// the queue bounded all the same, and an idle leaf still demotes.
TEST_F(UnitPolicyTest, TheDemotionQueueStaysBoundedWithoutEvictions) {
    UnitPolicy policy(Config(true, 60.0));
    Store(policy, {"idle.m", "low.m"});
    // idle saves more per byte than low, so only its demotion puts it first.
    Apply(policy, {Upsert("idle", 9.0, {"idle.m"}, {}),
                   Upsert("low", 1.0, {"low.m"}, {})});
    for (int i = 0; i < 200000; ++i) {
        now_ += 1ms;
        Apply(policy, {Op(UnitOp::kTouch, "low")});
    }
    EXPECT_LE(policy.Stats(now_).demotion_entries, 4u * 2 + 4096);
    EXPECT_EQ(policy.RankOf("idle", now_)->first, 1);
    EXPECT_EQ(policy.RankOf("low", now_)->first, 2);
    EXPECT_EQ(Evict(policy, 1), std::vector<std::string>{"idle"});
}

// The last resort of a round: orphans still in their grace period, oldest
// first. A blocked one is back after the round with its grace.
TEST_F(UnitPolicyTest, OrphansInTheirGraceAreTheLastResort) {
    UnitPolicy policy(Config(true, 0.0, 30.0));
    Store(policy, {"o1"});
    now_ += 1s;
    Store(policy, {"o2"});
    now_ += 1s;
    Store(policy, {"o3"});
    EXPECT_FALSE(policy.NextVictim(now_).has_value());
    auto victim = policy.NextGraceOrphan();
    ASSERT_TRUE(victim && victim->orphan && victim->grace);
    EXPECT_EQ(victim->keys, std::vector<std::string>{"o1"});
    EXPECT_TRUE(policy.StillEvictable(*victim, "o1"));
    policy.OnObjectErased("o1", now_);
    policy.Evicted(*victim, true, kObject, now_);
    victim = policy.NextGraceOrphan();
    ASSERT_TRUE(victim.has_value());
    EXPECT_EQ(victim->keys, std::vector<std::string>{"o2"});
    policy.Blocked(*victim);
    policy.EndRound(now_);
    EXPECT_FALSE(policy.NextVictim(now_).has_value());
    EXPECT_TRUE(policy.Check().empty());
    EXPECT_EQ(policy.NextGraceOrphan()->keys, std::vector<std::string>{"o2"});
    const auto stats = policy.Stats(now_);
    EXPECT_EQ(stats.grace_orphans, 1u);
    EXPECT_EQ(stats.grace_orphan_bytes, kObject);
    EXPECT_EQ(stats.orphans, 1u);
}

// Counters are Stats without the tier counts (a scan of every unit); the
// stored bytes are kept as objects come, change size and go.
TEST_F(UnitPolicyTest, CountersAreTheStatsWithoutTheTierCounts) {
    UnitPolicy policy(Config(true, 20.0, 5.0));
    Store(policy, {"p0", "p1", "a.m", "b.m", "c.m", "stray"});
    policy.OnObjectStored("b.m", 3 * kObject, now_);
    policy.OnObjectStored("a.m", kObject / 2, now_);
    Apply(policy, {Upsert("a", 2.0, {"a.m"}, {"p0"}, true),
                   Upsert("b", 4.0, {"b.m"}, {"p0", "p1"}, false, "a"),
                   Upsert("c", 1.0, {"c.m"}, {})});
    Apply(policy, {Op(UnitOp::kTouch, "a"), Op(UnitOp::kDead, "c")});
    policy.OnObjectErased("p1", now_);
    policy.OnObjectErased("p1", now_);
    now_ += 10s;
    EXPECT_EQ(Evict(policy, 1), std::vector<std::string>{"orphan:stray"});
    const auto counters = policy.Counters(now_);
    EXPECT_EQ(counters.objects, 4u);
    EXPECT_EQ(counters.object_bytes, 5 * kObject + kObject / 2);
    auto stats = policy.Stats(now_);
    EXPECT_EQ(stats.units_dead, 2u);
    stats.units_dead = stats.units_inner = stats.units_leaf = 0;
    stats.units_anchor = 0;
    EXPECT_TRUE(counters == stats);
}

// A page listed twice, or also as a member, counts once in the unit's size.
TEST_F(UnitPolicyTest, ARepeatedPageCountsOnce) {
    UnitPolicy policy(Config(false));
    Store(policy, {"p0", "u.m"});
    Apply(policy, {Upsert("u", 2.0, {"u.m"}, {"p0", "p0", "u.m"}, true)});
    // Its member and one page: two objects.
    EXPECT_NEAR(policy.RankOf("u", now_)->second, 2.0 / (2 * kObject), 1e-12);
    EXPECT_EQ(policy.Stats(now_).dropped_refs, 2u);
    EXPECT_TRUE(policy.Check().empty());
    EXPECT_EQ(Evict(policy), (std::vector<std::string>{"u", "orphan:p0"}));
}

// Evictions count per tier only with leaf-last tiers: without them every unit
// would count as dead.
TEST_F(UnitPolicyTest, EvictionsCountPerTierOnlyWithLeafLastTiers) {
    for (const bool leaf_last : {false, true}) {
        UnitPolicy policy(Config(leaf_last));
        Store(policy, {"a.m"});
        Apply(policy, {Upsert("a", 1.0, {"a.m"}, {})});
        ASSERT_EQ(Evict(policy), std::vector<std::string>{"a"});
        const auto stats = policy.Stats(now_);
        EXPECT_EQ(stats.evicted_units, 1u);
        EXPECT_EQ(stats.evicted_by_tier[0], 0u) << "leaf_last " << leaf_last;
        EXPECT_EQ(stats.evicted_by_tier[2], leaf_last ? 1u : 0u);
    }
}

// Random operations over a tree of paths, as engines sharing one store would
// issue them: every round the policy stays consistent, never gives up a unit
// whose member another unit owns, and pages go only once no unit needs them.
TEST_F(UnitPolicyTest, RandomOperationsKeepThePolicyConsistent) {
    std::mt19937 rng(7);
    for (bool leaf_last : {false, true}) {
        UnitPolicy policy(Config(leaf_last, 20.0, 5.0));
        std::map<std::string, std::vector<std::string>> refs_of;
        std::set<std::string> stored;
        int next = 0;
        for (int step = 0; step < 3000; ++step) {
            now_ += std::chrono::milliseconds(rng() % 500);
            const int action = rng() % 10;
            if (action < 5) {
                // A unit on a random path: a prefix of an existing unit's
                // path, extended by fresh pages.
                std::vector<std::string> path;
                if (!refs_of.empty() && rng() % 3) {
                    auto it = refs_of.begin();
                    std::advance(it, rng() % refs_of.size());
                    path.assign(
                        it->second.begin(),
                        it->second.begin() + rng() % (it->second.size() + 1));
                }
                const int fresh = rng() % 3;
                for (int i = 0; i < fresh; ++i) {
                    path.push_back("p" + std::to_string(next++));
                }
                const std::string id = "u" + std::to_string(next++);
                std::vector<std::string> members{id + ".m0", id + ".m1"};
                for (const auto& key : path) {
                    if (stored.insert(key).second) {
                        policy.OnObjectStored(key, kObject, now_);
                    }
                }
                if (rng() % 4) policy.OnObjectStored(members[0], kObject, now_);
                if (rng() % 4) policy.OnObjectStored(members[1], kObject, now_);
                std::string parent;
                if (!refs_of.empty() && rng() % 2) {
                    auto it = refs_of.begin();
                    std::advance(it, rng() % refs_of.size());
                    parent = it->first;
                }
                EXPECT_TRUE(
                    policy
                        .Apply(
                            {Upsert(id,
                                    1.0 + path.size() + (rng() % 100) / 100.0,
                                    members, path, !path.empty() && rng() % 2,
                                    parent)},
                            now_)
                        .empty());
                refs_of[id] = path;
            } else if (action < 7 && !refs_of.empty()) {
                auto it = refs_of.begin();
                std::advance(it, rng() % refs_of.size());
                policy.Apply(
                    {Op(rng() % 4 ? UnitOp::kTouch : UnitOp::kDead, it->first)},
                    now_);
            } else {
                // An eviction round; some victims are leased.
                for (int i = 0; i < 6; ++i) {
                    auto victim = policy.NextVictim(now_);
                    if (!victim) break;
                    if (rng() % 5 == 0) {
                        policy.Blocked(*victim);
                        continue;
                    }
                    for (const auto& key : victim->keys) {
                        ASSERT_TRUE(policy.StillEvictable(*victim, key));
                        policy.OnObjectErased(key, now_);
                        stored.erase(key);
                    }
                    policy.Evicted(*victim, true, 0, now_);
                    if (!victim->orphan) refs_of.erase(victim->unit);
                }
                policy.EndRound(now_);
            }
            const auto violations = policy.Check();
            ASSERT_TRUE(violations.empty()) << violations.front();
        }
        // Drain: every unit and object goes, pages after their last unit.
        now_ += 60s;
        for (int round = 0; round < 100; ++round) {
            bool any = false;
            while (auto victim = policy.NextVictim(now_)) {
                any = true;
                for (const auto& key : victim->keys) {
                    policy.OnObjectErased(key, now_);
                }
                policy.Evicted(*victim, true, 0, now_);
            }
            policy.EndRound(now_);
            if (!any) break;
        }
        const auto stats = policy.Stats(now_);
        EXPECT_EQ(stats.units, 0u);
        EXPECT_EQ(stats.objects, 0u);
        EXPECT_TRUE(policy.Check().empty());
    }
}

}  // namespace
}  // namespace mooncake::test
