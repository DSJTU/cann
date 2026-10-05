// CPU control-flow model: one SDK Cube producer and two paired AIV consumers.
// Flags protect shared GM slots; this does not model NPU pipe completion timing.
#pragma once
namespace sim {
class Pair {
    std::mutex mutex;
    std::condition_variable changed;
    bool aborted=false, ready[2][16]{}, free[2][16]{};
public:
    void Abort() { std::lock_guard<std::mutex> lock(mutex);aborted=true;changed.notify_all(); }
    void Send(bool cube, uint32_t sub, uint16_t id) {
        std::lock_guard<std::mutex> lock(mutex);
        if(id>=16) throw std::runtime_error("invalid cross-core flag");
        if(cube) {
            if(ready[0][id] || ready[1][id]) throw std::runtime_error("ready flag overwritten");
            ready[0][id]=ready[1][id]=true;
        } else {
            if(free[sub][id]) throw std::runtime_error("free credit overwritten");
            free[sub][id]=true;
        }
        changed.notify_all();
    }
    void Wait(bool cube, uint32_t sub, uint16_t id) {
        std::unique_lock<std::mutex> lock(mutex);
        if(id>=16) throw std::runtime_error("invalid cross-core flag");
        if(!changed.wait_for(lock,std::chrono::seconds(20),[&] {
            return aborted || (cube ? free[0][id] && free[1][id] : ready[sub][id]);
        })) throw std::runtime_error("cross-core wait timeout");
        if(aborted) throw std::runtime_error("cross-core launch aborted");
        if(cube) free[0][id]=free[1][id]=false;
        else ready[sub][id]=false;
    }
};
inline Pair*& activePair() { static thread_local Pair* p=nullptr;return p; }
template<typename F> void LaunchMixed(uint32_t n,F f) {
    ++launchCount();blocks()=n;
    Collective collective(2*n);activeCollective()=&collective;
    std::vector<std::unique_ptr<Pair>> pairs;
    for(uint32_t i=0;i<n;++i) pairs.emplace_back(new Pair);
    std::exception_ptr failure;std::mutex mutex;
    std::vector<std::thread> workers;
    for(uint32_t i=0;i<3*n;++i) workers.emplace_back([&,i] {
        const uint32_t core=i/3, member=i%3;
        isCube()=member==0;block()=isCube() ? core : 2*core+member-1;
        logicalBlocks()=n;activePair()=pairs[core].get();
        try { f(); } catch(...) {
            {std::lock_guard<std::mutex> lock(mutex);if(!failure) failure=std::current_exception();}
            collective.Abort();for(auto& p:pairs) p->Abort();
        }
    });
    for(auto& worker:workers) worker.join();
    activeCollective()=nullptr;
    if(failure) std::rethrow_exception(failure);
}
}
namespace AscendC {
template<uint8_t Mode,Pipe P> void CrossCoreSetFlag(uint16_t id) {
    static_assert(Mode==2,"only Cube/paired AIV flags modeled");
    sim::activePair()->Send(sim::isCube(),GetSubBlockIdx(),id);
}
inline void CrossCoreWaitFlag(uint16_t id) {
    sim::activePair()->Wait(sim::isCube(),GetSubBlockIdx(),id);
}
}
