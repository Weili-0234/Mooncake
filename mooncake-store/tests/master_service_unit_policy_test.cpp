// --eviction_policy=unit_gds inside MasterService: objects put through the
// master's own Put path, units registered through ApplyUnitUpdates, eviction
// cycles driven synchronously or by the eviction thread.

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <set>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include "master_service.h"
#include "types.h"

namespace mooncake::test {
namespace {

constexpr size_t kSegmentBase = 0x300000000;
constexpr size_t kMiB = 1024 * 1024;

UnitUpdate Upsert(const std::string& id, double value,
                  std::vector<std::string> members,
                  std::vector<std::string> refs) {
    UnitUpdate update;
    update.op = UnitOp::kUpsert;
    update.id = id;
    update.value = value;
    update.members = std::move(members);
    update.refs = std::move(refs);
    return update;
}

UnitUpdate Dead(const std::string& id) {
    UnitUpdate update;
    update.op = UnitOp::kDead;
    update.id = id;
    return update;
}

class MasterServiceUnitPolicyTest : public ::testing::Test {
   protected:
    void SetUp() override {
        google::InitGoogleLogging("MasterServiceUnitPolicyTest");
        FLAGS_logtostderr = true;
    }
    void TearDown() override { google::ShutdownGoogleLogging(); }

    std::unique_ptr<MasterService> MakeService(
        size_t segment_bytes, double high_watermark = 0.95, double ratio = 0.05,
        uint64_t lease_ms = 50, double grace_s = 0.0, double headroom = 0.0) {
        UnitPolicyConfig unit;
        unit.enabled = true;
        unit.leaf_last = true;
        unit.idle_demote_s = 0.0;
        unit.orphan_grace_s = grace_s;
        unit.dead_headroom_ratio = headroom;
        auto config = MasterServiceConfig::builder()
                          .set_default_kv_lease_ttl(lease_ms)
                          .set_eviction_high_watermark_ratio(high_watermark)
                          .set_eviction_ratio(ratio)
                          .set_unit_policy(unit)
                          .build();
        auto service = std::make_unique<MasterService>(config);
        Segment segment;
        segment.id = generate_uuid();
        segment.name = "unit_policy_segment";
        segment.base = kSegmentBase;
        segment.size = segment_bytes;
        segment.te_endpoint = segment.name;
        EXPECT_TRUE(
            service->MountSegment(segment, generate_uuid()).has_value());
        return service;
    }

    bool Put(MasterService& service, const std::string& key,
             uint64_t bytes = kMiB, const std::string& group = "") {
        ReplicateConfig config;
        config.replica_num = 1;
        if (!group.empty()) config.group_ids = std::vector<std::string>{group};
        auto start =
            service.PutStart(client_, key, TenantId::Default(), bytes, config);
        if (!start.has_value()) return false;
        EXPECT_TRUE(
            service
                .PutEnd(client_, key, TenantId::Default(), ReplicaType::MEMORY)
                .has_value());
        return true;
    }

    std::set<std::string> Keys(MasterService& service) {
        auto keys = service.GetAllKeys(TenantId::Default());
        EXPECT_TRUE(keys.has_value());
        return {keys->begin(), keys->end()};
    }

    void Register(MasterService& service, std::vector<UnitUpdate> updates) {
        auto errors = service.ApplyUnitUpdates(updates);
        ASSERT_TRUE(errors.has_value());
        EXPECT_TRUE(errors->empty());
    }

    void ExpectConsistent(MasterService& service) {
        auto violations = service.CheckUnitPolicy();
        ASSERT_TRUE(violations.has_value());
        EXPECT_TRUE(violations->empty()) << violations->front();
    }

    const UUID client_ = generate_uuid();
};

TEST_F(MasterServiceUnitPolicyTest, TheLeaseOrderRefusesUnitUpdates) {
    MasterService service;
    auto result = service.ApplyUnitUpdates({Upsert("u", 1.0, {"m"}, {})});
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), ErrorCode::UNAVAILABLE_IN_CURRENT_MODE);
    EXPECT_FALSE(service.GetUnitPolicyStats().has_value());
}

TEST_F(MasterServiceUnitPolicyTest, EvictsWholeUnitsInThePolicysOrder) {
    auto service = MakeService(16 * kMiB);
    for (const std::string key : {"p0", "p1"}) ASSERT_TRUE(Put(*service, key));
    ASSERT_TRUE(Put(*service, "a0", kMiB, "unit-a"));
    ASSERT_TRUE(Put(*service, "a1", kMiB, "unit-a"));
    ASSERT_TRUE(Put(*service, "b0", kMiB, "unit-b"));
    ASSERT_TRUE(Put(*service, "b1", kMiB, "unit-b"));
    // a saves little per byte, b much: one MiB to free takes all of a.
    Register(*service, {Upsert("unit-a", 1.0, {"a0", "a1"}, {"p0"}),
                        Upsert("unit-b", 10.0, {"b0", "b1"}, {"p0", "p1"})});
    service->RunUnitBatchEvictForTesting(1.0 / 16);
    EXPECT_EQ(Keys(*service), (std::set<std::string>{"p0", "p1", "b0", "b1"}));
    const auto stats = service->GetUnitPolicyStats();
    ASSERT_TRUE(stats.has_value());
    EXPECT_EQ(stats->evicted_units, 1u);
    EXPECT_EQ(stats->evicted_unit_bytes, 2 * kMiB);
    ExpectConsistent(*service);
    // Everything else: b, then the pages no unit needs any more.
    service->RunUnitBatchEvictForTesting(1.0);
    EXPECT_TRUE(Keys(*service).empty());
    ExpectConsistent(*service);
}

TEST_F(MasterServiceUnitPolicyTest, ALeasedMemberKeepsItsWholeUnit) {
    auto service = MakeService(16 * kMiB);
    ASSERT_TRUE(Put(*service, "a0", kMiB, "unit-a"));
    ASSERT_TRUE(Put(*service, "a1", kMiB, "unit-a"));
    ASSERT_TRUE(Put(*service, "b0", kMiB, "unit-b"));
    ASSERT_TRUE(Put(*service, "b1", kMiB, "unit-b"));
    Register(*service, {Upsert("unit-a", 1.0, {"a0", "a1"}, {}),
                        Upsert("unit-b", 10.0, {"b0", "b1"}, {})});
    // A reader holds a lease on one of a's members: a stays whole, b goes.
    ASSERT_TRUE(service->GetReplicaList("a1", TenantId::Default()).has_value());
    service->RunUnitBatchEvictForTesting(1.0 / 16);
    EXPECT_EQ(Keys(*service), (std::set<std::string>{"a0", "a1"}));
    EXPECT_EQ(service->GetUnitPolicyStats()->blocked, 1u);
    // Once the lease expires a goes too.
    std::this_thread::sleep_for(std::chrono::milliseconds(80));
    service->RunUnitBatchEvictForTesting(1.0 / 16);
    EXPECT_TRUE(Keys(*service).empty());
    ExpectConsistent(*service);
}

// A touch, or a read's lease, after the master checked a victim's members and
// before it erases them: the unit stays whole and live, skipped for the round
// instead of marked dead (and partly evicted).
TEST_F(MasterServiceUnitPolicyTest, AUseBetweenTheTwoPassesKeepsTheUnitWhole) {
    for (const bool touch : {true, false}) {
        auto service = MakeService(16 * kMiB, 0.95, 0.05, /*lease_ms=*/10000);
        ASSERT_TRUE(Put(*service, "a0", kMiB, "unit-a"));
        ASSERT_TRUE(Put(*service, "a1", kMiB, "unit-a"));
        ASSERT_TRUE(Put(*service, "b0", kMiB, "unit-b"));
        ASSERT_TRUE(Put(*service, "b1", kMiB, "unit-b"));
        Register(*service, {Upsert("unit-a", 1.0, {"a0", "a1"}, {}),
                            Upsert("unit-b", 10.0, {"b0", "b1"}, {})});
        bool used = false;
        service->SetUnitEvictBetweenPassesForTesting([&] {
            if (used) return;
            used = true;
            if (touch) {
                UnitUpdate update;
                update.op = UnitOp::kTouch;
                update.id = "unit-a";
                Register(*service, {update});
            } else {
                ASSERT_TRUE(service->GetReplicaList("a0", TenantId::Default())
                                .has_value());
            }
        });
        service->RunUnitBatchEvictForTesting(1.0 / 16);
        EXPECT_TRUE(used);
        EXPECT_EQ(Keys(*service), (std::set<std::string>{"a0", "a1"}))
            << "touch " << touch;
        const auto stats = service->GetUnitPolicyStats(/*exact=*/true);
        ASSERT_TRUE(stats.has_value());
        EXPECT_EQ(stats->blocked, 1u);
        EXPECT_EQ(stats->partial, 0u);
        EXPECT_EQ(stats->units_dead, 0u);
        ExpectConsistent(*service);
    }
}

// A store of objects no unit claims, all in their grace period: a full round
// takes them as its last resort, oldest first, so it still frees space; a
// dead-only round takes none of them.
TEST_F(MasterServiceUnitPolicyTest, AFullRoundTakesOrphansInTheirGraceLast) {
    auto service = MakeService(16 * kMiB, 0.95, 0.05, /*lease_ms=*/50,
                               /*grace_s=*/120.0);
    std::set<std::string> keys;
    for (int i = 10; i < 22; ++i) {
        const std::string key = "o" + std::to_string(i);
        ASSERT_TRUE(Put(*service, key));
        keys.insert(key);
    }
    service->RunUnitBatchEvictForTesting(1.0, /*dead_only=*/true);
    EXPECT_EQ(Keys(*service), keys);
    service->RunUnitBatchEvictForTesting(2.0 / 16);
    keys.erase("o10");
    keys.erase("o11");
    EXPECT_EQ(Keys(*service), keys);
    const auto stats = service->GetUnitPolicyStats();
    ASSERT_TRUE(stats.has_value());
    EXPECT_EQ(stats->orphans, 2u);
    EXPECT_EQ(stats->orphan_bytes, 2 * kMiB);
    EXPECT_EQ(stats->grace_orphans, 2u);
    ExpectConsistent(*service);
}

TEST_F(MasterServiceUnitPolicyTest,
       AnErasedMemberKillsItsUnitAndOrphansGoFirst) {
    auto service = MakeService(16 * kMiB);
    ASSERT_TRUE(Put(*service, "stray"));
    ASSERT_TRUE(Put(*service, "a0", kMiB, "unit-a"));
    ASSERT_TRUE(Put(*service, "a1", kMiB, "unit-a"));
    ASSERT_TRUE(Put(*service, "b0", kMiB, "unit-b"));
    Register(*service, {Upsert("unit-a", 9.0, {"a0", "a1"}, {}),
                        Upsert("unit-b", 1.0, {"b0"}, {})});
    // A client removes one member of a: a can no longer be resumed at.
    ASSERT_TRUE(service->Remove("a1", TenantId::Default()).has_value());
    EXPECT_EQ(service->GetUnitPolicyStats()->lost_members, 1u);
    // The orphan goes first, then the dead a, before b.
    service->RunUnitBatchEvictForTesting(2.0 / 16);
    EXPECT_EQ(Keys(*service), (std::set<std::string>{"b0"}));
    ExpectConsistent(*service);
}

TEST_F(MasterServiceUnitPolicyTest, TheEvictionThreadKeepsEveryUnitWhole) {
    // 32 MiB evicted from 70% use: units of two 1 MiB members arrive until
    // the store is full many times over. The grace keeps a member that
    // arrived before its unit's registration.
    auto service = MakeService(32 * kMiB, 0.7, 0.2, /*lease_ms=*/1,
                               /*grace_s=*/5.0);
    std::vector<std::pair<std::string, std::string>> units;
    int failures = 0;
    for (int i = 0; i < 120 && failures < 2000; ++i) {
        const std::string id = "unit-" + std::to_string(i);
        const std::string m0 = id + ".m0", m1 = id + ".m1";
        while (!Put(*service, m0, kMiB, id)) {
            failures++;
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        while (!Put(*service, m1, kMiB, id)) {
            failures++;
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        Register(*service, {Upsert(id, 1.0 + i % 7, {m0, m1}, {})});
        units.emplace_back(m0, m1);
    }
    ASSERT_LT(failures, 2000);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    const auto keys = Keys(*service);
    size_t whole = 0;
    for (const auto& [m0, m1] : units) {
        EXPECT_EQ(keys.contains(m0), keys.contains(m1)) << m0;
        whole += keys.contains(m0);
    }
    EXPECT_GT(whole, 0u);
    EXPECT_LT(whole, units.size());
    const auto stats = service->GetUnitPolicyStats();
    EXPECT_GT(stats->evicted_units, 0u);
    EXPECT_EQ(stats->partial, 0u);
    ExpectConsistent(*service);
}

TEST_F(MasterServiceUnitPolicyTest,
       BelowTheWatermarkTheHeadroomTakesOnlyDeadUnits) {
    // 16 MiB at watermark 0.9, six units of two 1 MiB members (75% used), one
    // dead: with a headroom of 0.3 the eviction thread takes dead units from
    // 60% on, so it takes the dead unit and then nothing more although 62.5%
    // is still above 60% (only live units are left below the watermark);
    // without the headroom (the control) nothing goes.
    for (double headroom : {0.3, 0.0}) {
        auto service = MakeService(16 * kMiB, 0.9, 0.05, /*lease_ms=*/10,
                                   /*grace_s=*/5.0, headroom);
        std::set<std::string> live;
        for (int u = 0; u < 6; ++u) {
            const std::string id = "unit-" + std::to_string(u);
            ASSERT_TRUE(Put(*service, id + ".0", kMiB, id));
            ASSERT_TRUE(Put(*service, id + ".1", kMiB, id));
            Register(*service,
                     {Upsert(id, 1.0 + u, {id + ".0", id + ".1"}, {})});
            if (u >= 1) live.insert({id + ".0", id + ".1"});
        }
        Register(*service, {Dead("unit-0")});
        if (headroom > 0.0) {
            for (int i = 0; i < 200 && Keys(*service) != live; ++i) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            // Give the thread time to take a live unit if it ever would.
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
            EXPECT_EQ(Keys(*service), live);
            const auto stats = service->GetUnitPolicyStats();
            ASSERT_TRUE(stats.has_value());
            EXPECT_EQ(stats->headroom_units, 1u);
            EXPECT_EQ(stats->headroom_unit_bytes, 2 * kMiB);
            EXPECT_EQ(stats->evicted_units, 1u);
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
            EXPECT_EQ(Keys(*service).size(), 12u);
            EXPECT_EQ(service->GetUnitPolicyStats()->evicted_units, 0u);
        }
        ExpectConsistent(*service);
    }
}

TEST_F(MasterServiceUnitPolicyTest,
       TheNewSettingsNeedLeafLastAndAHeadroomBelowTheWatermark) {
    for (auto [leaf_last, anchors, headroom] :
         {std::tuple{false, true, 0.0}, std::tuple{false, false, 0.1},
          std::tuple{true, false, 0.95}, std::tuple{true, false, -0.1}}) {
        UnitPolicyConfig unit;
        unit.enabled = true;
        unit.leaf_last = leaf_last;
        unit.anchor_tier = anchors;
        unit.dead_headroom_ratio = headroom;
        auto config = MasterServiceConfig::builder()
                          .set_eviction_high_watermark_ratio(0.9)
                          .set_unit_policy(unit)
                          .build();
        EXPECT_THROW(MasterService service(config), std::invalid_argument)
            << leaf_last << anchors << headroom;
    }
}

}  // namespace
}  // namespace mooncake::test
