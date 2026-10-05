// Synchronous Cube instructions plus paired CPU workers for address/control checks.
// FP64 accumulation deliberately does not model hardware rounding or pipe timing.
#pragma once
namespace sim {
class Pair {
    std::mutex mutex;
    std::condition_variable changed;
    bool aborted=false;
    bool pending[2][16]{};
public:
    void Abort(){std::lock_guard<std::mutex> lock(mutex);aborted=true;changed.notify_all();}
    void Send(bool sender,uint16_t id) {
        std::lock_guard<std::mutex> lock(mutex);
        if(id>=16 || pending[!sender][id])throw std::runtime_error("cross-core flag overwritten");
        pending[!sender][id]=true;changed.notify_all();
    }
    void Wait(bool receiver,uint16_t id) {
        std::unique_lock<std::mutex> lock(mutex);
        if(id>=16)throw std::runtime_error("invalid cross-core flag");
        if(!changed.wait_for(lock,std::chrono::seconds(20),[&]{return pending[receiver][id]||aborted;}))
            throw std::runtime_error("cross-core wait timeout");
        if(aborted)throw std::runtime_error("cross-core aborted");
        pending[receiver][id]=false;
    }
};
inline std::vector<std::unique_ptr<Pair>>*& activePairs(){static std::vector<std::unique_ptr<Pair>>* p=nullptr;return p;}
template<typename F>void LaunchDirect(uint32_t n,F f) {
    ++launchCount();blocks()=n;
    Collective collective(n);activeCollective()=&collective;
    std::vector<std::unique_ptr<Pair>> pairs;
    for(uint32_t i=0;i<n;++i)pairs.emplace_back(new Pair);
    activePairs()=&pairs;
    std::exception_ptr failure;std::mutex mutex;
    std::vector<std::thread> workers;
    for(uint32_t i=0;i<2*n;++i)workers.emplace_back([&,i] {
        block()=i/2;logicalBlocks()=0;isCube()=i%2==0;
        try{f();}catch(...) {
            {std::lock_guard<std::mutex> lock(mutex);if(!failure)failure=std::current_exception();}
            collective.Abort();for(auto& p:pairs)p->Abort();
        }
    });
    for(auto& worker:workers)worker.join();
    activePairs()=nullptr;activeCollective()=nullptr;
    if(failure)std::rethrow_exception(failure);
}
}
namespace AscendC {
template<uint8_t Mode,Pipe P>void CrossCoreSetFlag(uint16_t id) {
    static_assert(Mode==2,"only paired flags modeled");(*sim::activePairs())[GetBlockIdx()]->Send(sim::isCube(),id);
}
template<uint8_t Mode,Pipe P>void CrossCoreWaitFlag(uint16_t id) {
    static_assert(Mode==2,"only paired flags modeled");(*sim::activePairs())[GetBlockIdx()]->Wait(sim::isCube(),id);
}
template<typename T>struct InitConstValueParams {uint16_t repeatTimes,blockNum,dstGap;T value;};
template<typename T>void InitConstValue(LocalTensor<T> dst,InitConstValueParams<T> p) {
    for(uint32_t r=0;r<p.repeatTimes;++r)
        for(uint32_t i=0;i<p.blockNum*32/sizeof(T);++i)dst.SetValue(r*(p.blockNum+p.dstGap)*32/sizeof(T)+i,p.value);
}
struct Nd2NzParams {uint16_t ndNum,nValue,dValue,srcNdMatrixStride,srcDValue,dstNzC0Stride,dstNzNStride,dstNzMatrixStride;};
template<typename T>void DataCopy(LocalTensor<T> dst,GlobalTensor<T> src,Nd2NzParams p) {
    if(p.ndNum!=1)throw std::runtime_error("only single ND-to-NZ matrix modeled");
    for(uint32_t r=0;r<p.nValue;++r)for(uint32_t c=0;c<(uint32_t(p.dValue)+15)/16*16;++c)
        dst.SetValue(c/16*p.dstNzC0Stride*16+r*p.dstNzNStride*16+c%16,
                     c<p.dValue ? src.GetValue(r*p.srcDValue+c) : T(0));
}
struct LoadData2dParams {uint16_t startIndex;uint8_t repeatTimes;uint16_t srcStride;uint8_t sid;uint16_t dstGap;bool transpose;uint8_t addrmode;};
template<typename T>void LoadData(LocalTensor<T> dst,LocalTensor<T> src,LoadData2dParams p) {
    for(uint32_t r=0;r<p.repeatTimes;++r)for(uint32_t i=0;i<16;++i)for(uint32_t j=0;j<16;++j)
        dst.SetValue(r*(p.dstGap+1)*256+i*16+j,
                     src.GetValue((p.startIndex+r*p.srcStride)*256+(p.transpose?j*16+i:i*16+j)));
}
struct MmadParams {uint16_t m=0,n=0,k=0;bool cmatrixInitVal=true;};
template<typename T>void Mmad(LocalTensor<float> dst,LocalTensor<T> a,LocalTensor<T> b,MmadParams p) {
    uint32_t ma=(p.m+15)/16*16,na=(p.n+15)/16*16;
    std::vector<float> av(ma*p.k),bv(na*p.k);
    for(uint32_t i=0;i<av.size();++i)av[i]=float(a.GetValue(i));
    for(uint32_t i=0;i<bv.size();++i)bv[i]=float(b.GetValue(i));
    auto& accum=dst.Accumulator();if(p.cmatrixInitVal)accum.assign(ma*na,0.0);
    if(accum.size()!=ma*na)throw std::runtime_error("Mmad accumulator shape changed mid-dot");
    for(uint32_t m=0;m<p.m;++m)for(uint32_t n=0;n<p.n;++n) {
        uint32_t ci=n/16*ma*16+m*16+n%16;double sum=accum[ci];
        for(uint32_t k=0;k<p.k;++k)
            sum+=double(av[m/16*p.k*16+k/16*256+m%16*16+k%16])*
                       bv[k/16*na*16+n/16*256+n%16*16+k%16];
        accum[ci]=sum;dst.SetValue(ci,float(sum));
    }
}
struct FixpipeConfig {};
constexpr FixpipeConfig CFG_ROW_MAJOR{};
struct FixpipeParamsV220 {uint16_t nSize,mSize,srcStride;uint32_t dstStride;bool reluEn;};
template<typename T,typename U,const FixpipeConfig& Config>void Fixpipe(GlobalTensor<T> dst,LocalTensor<U> src,FixpipeParamsV220 p) {
    for(uint32_t r=0;r<p.mSize;++r)for(uint32_t c=0;c<p.nSize;++c)
        dst.SetValue(r*p.dstStride+c,src.GetValue(c/16*p.srcStride*16+r*16+c%16));
}
}
