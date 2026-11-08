// state_space_e6_l49_npc_event_v3 rules on packed public states. One thread advances a game.
// The host prepends public enum, layout, event and feature constants.
typedef unsigned int u32;
// Addresses are 64-bit even though bounded counts/compact prefixes use int32.
// In particular an action-row prefix * FEATURE_DIM can exceed 2^31 elements
// on a 96 GiB device while both the prefix and the allocation remain valid.
typedef long long index_t;
static_assert(sizeof(index_t) == 8, "CUDA linear addresses require 64-bit integers");

struct RNG {
    u32* mt;
    int stride;
    __device__ u32& at(int i) {return mt[index_t(i)*stride];}
    __device__ u32 next() {
        int pos = (int)at(624);
        if (pos >= 624) {
            for (int i=0;i<624;i++) {
                u32 y=(at(i)&0x80000000u)|(at((i+1)%624)&0x7fffffffu);
                at(i)=at((i+397)%624)^(y>>1)^((y&1u)?0x9908b0dfu:0u);
            }
            pos=0;
        }
        u32 y=at(pos++); at(624)=pos;
        y^=y>>11; y^=(y<<7)&0x9d2c5680u; y^=(y<<15)&0xefc60000u; y^=y>>18;
        return y;
    }
    __device__ int below(int n) {
        int bits=32-__clz((u32)n);
        u32 v;
        do {v=next()>>(32-bits);} while (v>=(u32)n);
        return (int)v;
    }
    __device__ bool bern(int num,int den) {
        return num>=den || (num>0 && below(den)<num);
    }
};

__device__ int imin(int a,int b) {return a<b?a:b;}
__device__ int imax(int a,int b) {return a>b?a:b;}
__device__ int ret_phase(int p) {return p==0?PH_ADVANCE:p==1?PH_SHOP:PH_END;}
__device__ int hindex(int r,int q,int t,int level) {return ((r-1)*2+q-1)*42+t*6+level-1;}
__device__ int hscore(const int* s) {
    int total=0;
    for(int i=0;i<252;i++) total+=s[S_H+i]*imin(i/6%7,i%6+1);
    return total;
}
__device__ void clear_offer(int* s) {s[S_offer_n]=0;}
__device__ void set_source(int* s,int rarity,int mode,int bound,int back,int reroll) {
    s[S_source]=1;s[S_rarity]=rarity;s[S_mode]=mode;s[S_bound]=bound;
    s[S_source_return]=back;s[S_reroll]=reroll;
}

// random.Random.sample for k <= 4; includes CPython's pool/set branch.
__device__ void sample(const int* population,int n,int k,int* result,RNG& rng) {
    if(n<=21) {
        int pool[36];for(int i=0;i<n;i++)pool[i]=population[i];
        for(int i=0;i<k;i++){int j=rng.below(n-i);result[i]=pool[j];pool[j]=pool[n-i-1];}
    } else {
        int selected[4];
        for(int i=0;i<k;i++) {
            int j;bool used;
            do {j=rng.below(n);used=false;for(int z=0;z<i;z++)used|=selected[z]==j;}while(used);
            selected[i]=j;result[i]=population[j];
        }
    }
}
__device__ void sort_ints(int* a,int n) {
    for(int i=1;i<n;i++){int x=a[i],j=i;while(j>0 && a[j-1]>x){a[j]=a[j-1];j--;}a[j]=x;}
}
__device__ int eligible(const int* s,int role,int quality,int* expanded) {
    int n=0;
    for(int i=0;i<252;i++) {
        if(i%6==5 || (role && i/84+1!=role) || (quality && i/42%2+1!=quality))continue;
        for(int j=0;j<s[S_H+i];j++)expanded[n++]=i;
    }
    return n;
}
__device__ void old_card(int* out,int category,int bonus) {
    out[0]=0;out[1]=category/84+1;out[2]=category/42%2+1;
    out[3]=category/6%7;out[4]=category%6+1;out[5]=imin(6,out[4]+1+bonus);out[6]=0;
}
__device__ int quality_count(int* s,int k,int nr,int nc,RNG& rng) {
    if(k>nr+nc){s[S_error]=1;return 0;}
    int raw=s[S_rarity]==2?k:0;
    if(s[S_rarity]==0)for(int i=0;i<k;i++)raw+=rng.bern(1,5);
    return imin(imin(k,nr),imax(imax(0,k-nc),raw));
}
__device__ void new_draw(int* pool,int n,int k,RNG& rng) {
    if(k>0 && k<n)for(int j=0;j<k;j++){int v=j+rng.below(n-j),t=pool[j];pool[j]=pool[v];pool[v]=t;}
}
__device__ void regular_page(int* s,RNG& rng) {
    int role=s[S_mode]==1?1:s[S_bound];
    if(s[S_mode]==0) {
        int weights[3],total=0;
        for(int r=0;r<3;r++) {
            int c=0;for(int i=r*84;i<(r+1)*84;i++)c+=s[S_H+i];
            int tau=r==0?6:5,w=imax(1,imin(100,30*imin(4,imax(0,tau-c))-10));
            weights[r]=(r==0?4:3)*w;total+=weights[r];
        }
        int v=rng.below(total);role=1;
        while(role<3 && v>=weights[role-1]){v-=weights[role-1];role++;}
    }
    int own=0;for(int i=(role-1)*84;i<role*84;i++)own+=s[S_H+i];
    int oldr[36],oldc[36];int ur=eligible(s,role,1,oldr),uc=eligible(s,role,2,oldc);
    int qn=own<=1?0:own<=3?51:own<(role==1?6:5)?3:1;
    int qd=own<=1?1:own<=3?100:own<(role==1?6:5)?4:1;
    int oldn=0;for(int i=0;i<3;i++)oldn+=rng.bern(qn,qd);
    oldn=imin(ur+uc,oldn);int newn=3-oldn;
    int pi=(role-1)*2,nr=s[S_pool_n+pi],nc=s[S_pool_n+pi+1];
    int kr=quality_count(s,newn,nr,nc,rng),ko=quality_count(s,oldn,ur,uc,rng);
    if(s[S_error])return;
    int* pr=s+S_pool+pi*9;int* pc=pr+9;
    new_draw(pr,nr,kr,rng);new_draw(pc,nc,newn-kr,rng);
    int count=0;
    for(int q=1;q<=2;q++) {
        int n=q==1?kr:newn-kr;int* pool=q==1?pr:pc;
        for(int j=0;j<n;j++) {
            int grade;
            if(!s[S_P] && rng.bern(49,100)){grade=1;s[S_P]=1;}
            else if(s[S_h]>=10){grade=2;s[S_P]=0;s[S_Q]=0;}
            else if(s[S_Q]){grade=3;s[S_P]=0;s[S_Q]=0;}
            else if(rng.bern(1,3)){grade=2;s[S_P]=0;s[S_Q]=1;}
            else{grade=3;s[S_P]=0;s[S_Q]=0;}
            int* c=s+S_offer+7*count++;
            c[0]=1;c[1]=role;c[2]=q;c[3]=pool[j];c[4]=0;c[5]=grade;c[6]=j+1;
        }
    }
    if(s[S_h]>=10)s[S_Q]=0;
    for(int q=1;q<=2;q++) {
        int picked[3],n=q==1?ko:oldn-ko;
        sample(q==1?oldr:oldc,q==1?ur:uc,n,picked,rng);sort_ints(picked,n);
        for(int j=0;j<n;j++)old_card(s+S_offer+7*count++,picked[j],0);
    }
    s[S_offer_n]=count;s[S_phase]=PH_PAGE;
}
__device__ void select_card(int* s,int index) {
    if(index<1 || index>s[S_offer_n]){s[S_error]=7;return;}
    int* c=s+S_offer+(index-1)*7;
    if(c[0]) {
        int pi=2*(c[1]-1)+c[2]-1;int* p=s+S_pool+pi*9;
        int n=--s[S_pool_n+pi];p[c[6]-1]=p[n];p[n]=0;
        s[S_H+hindex(c[1],c[2],c[3],c[5])]++;
        if(c[5]==3)s[S_h]++;if(s[S_h]>=10)s[S_Q]=0;
    } else {
        s[S_H+hindex(c[1],c[2],c[3],c[4])]--;
        s[S_H+hindex(c[1],c[2],c[3],c[5])]++;
    }
    s[S_phase]=ret_phase(s[S_source_return]);s[S_source]=0;clear_offer(s);
}
__device__ int enhance_num(int g,int f) {return g>=5?0:imin(3*(f+1),10);}
__device__ int enhance_next(int g,int f,int bonus) {
    int ng=g+bonus,nf=bonus?0:imin(f+1,3);return ng>=5?20:ng*4+nf;
}
__device__ void enhance(int* s,double* beliefs,RNG& rng) {
    int pool[36],picked[3],n=eligible(s,0,0,pool),k=imin(3,n);
    if(!n){s[S_error]=2;return;}
    sample(pool,n,k,picked,rng);sort_ints(picked,k);
    int g=s[S_gE],f=s[S_fE],num=enhance_num(g,f);
    // Fraction reduces the denominator before calling randrange.
    int den=10;if(num==0)den=1;else if(num==6){num=3;den=5;}else if(num==10){num=1;den=1;}
    int bonus=rng.bern(num,den),next=enhance_next(g,f,bonus);
    s[S_gE]=next/4;s[S_fE]=next%4;
    bool visible=false;for(int i=0;i<k;i++)visible|=picked[i]%6+1!=5;
    double posterior[24]={0};double mass=0;
    for(int i=0;i<24;i++)if(beliefs[i]>0) {
        double p=enhance_num(i/4,i%4)/10.0;
        for(int b=0;b<2;b++)if(!visible || b==bonus) {
            double w=beliefs[i]*(b?p:1-p);
            posterior[enhance_next(i/4,i%4,b)]+=w;mass+=w;
        }
    }
    for(int i=0;i<24;i++)beliefs[i]=posterior[i]/mass;
    for(int i=0;i<k;i++)old_card(s+S_offer+i*7,picked[i],bonus);
    s[S_offer_n]=k;s[S_phase]=PH_ENHANCE_PAGE;
}
__device__ void shelf(int* s,RNG& rng,int first,bool initial) {
    int tickets[10]={0,1,2,3,4,5,6,7,8,9},picked[2];sample(tickets,10,2,picked,rng);
    int y=picked[0]<6,bonus=picked[1]<8;
    int x=0;for(int i=0;i<4;i++)x+=rng.bern(65-5*y,100);x=imin(2,x);
    int b=s[S_shop_ordinal]==1,ng=2-b+x,nm=b+y;
    bool middle_initial=initial && (s[S_shop_ordinal]==2 || s[S_shop_ordinal]==3);
    int halfn=0;for(int i=0;i<3;i++)halfn+=rng.bern(middle_initial?3:7,middle_initial?4:10);halfn=imin(2,halfn);
    int ids[8]={0,1,2,3,4,5,6,7},discounted[4],half[2];
    sample(ids,8,4,discounted,rng);sample(discounted,4,halfn,half,rng);
    for(int i=0;i<6;i++)s[S_stock+i]=0;
    for(int i=0;i<ng+nm;i++) {
        bool d=false,h=false;for(int j=0;j<4;j++)d|=discounted[j]==i;
        for(int j=0;j<halfn;j++)h|=half[j]==i;
        s[S_stock+(i<ng?0:3)+(h?0:d?1:2)]++;
    }
    s[S_phase]=PH_SHOP;s[S_first]=0;s[S_source]=0;clear_offer(s);
}
__device__ void advance(int* s,double* belief,RNG& rng,const int* ci,const int* cp,const double* probs,const int* descs) {
    int key=(((s[S_node]*5+s[S_k])*2+s[S_A])*2+s[S_npc3_event])*48;
    int latent=(s[S_gC]*4+s[S_fC])*2+s[S_G1];
    const int* idx=ci+(key+latent)*3;
    int v=rng.below(idx[2]),chosen=idx[0];
    for(int j=0;j<idx[1];j++){chosen=idx[0]+j;int w=cp[chosen*3];if(v<w)break;v-=w;}
    int observed=cp[chosen*3+1],next=cp[chosen*3+2];
    double post[48]={0},mass=0;
    for(int i=0;i<48;i++)if(belief[24+i]>0) {
        const int* group=ci+(key+i)*3;
        for(int j=0;j<group[1];j++) {
            int p=group[0]+j;
            if(cp[p*3+1]==observed) {double w=belief[24+i]*probs[p];post[cp[p*3+2]]+=w;mass+=w;}
        }
    }
    for(int i=0;i<48;i++)belief[24+i]=post[i]/mass;
    s[S_gC]=next/8;s[S_fC]=next/2%4;s[S_G1]=next%2;
    const int* d=descs+observed*7;
    if(d[0]>7)s[S_npc3_event]=0;
    s[S_node]=d[0];s[S_floor]=d[1];s[S_k]=d[2];s[S_A]=d[3];s[S_phase]=d[4];s[S_coins]+=d[5];
    if(d[4]==PH_DRAW)set_source(s,0,0,0,0,1);
    if(d[4]==PH_SHOP_UNSEEN) {
        s[S_shop_ordinal]=d[6];s[S_first]=d[6]==1;s[S_paid]=0;s[S_shop_return]=0;
        for(int i=0;i<6;i++)s[S_stock+i]=0;
    }
}
__device__ void event_draw(int* s,RNG& rng) {
    int eid=0,seen=s[S_seen];
    if(s[S_floor]<5 && !(seen&(1<<4)) && rng.bern(7,10))eid=105;
    else {
        int total=0;for(int i=0;i<28;i++)if(!(seen&(1<<i)))total+=EVENT_WEIGHTS[i];
        if(!total){s[S_error]=3;return;}
        int v=rng.below(total);
        for(int i=0;i<28;i++)if(!(seen&(1<<i)) && EVENT_WEIGHTS[i]>0) {
            if(v<EVENT_WEIGHTS[i]){eid=i+101;break;}v-=EVENT_WEIGHTS[i];
        }
    }
    s[S_phase]=PH_EVENT;s[S_event]=eid;s[S_event_options]=EVENT_MASKS[eid-101];s[S_seen]|=1<<(eid-101);
}
__device__ void event_option(int* s,const int* a,RNG& rng) {
    int eid=s[S_event]-100;s[S_phase]=PH_ADVANCE;s[S_event]=0;s[S_event_options]=0;
    if(a[0]==AC_SKIP_EVENT)return;
    const int* opt=EVENT_OPTIONS[eid][a[1]%100];
    int v=rng.below(opt[3]),chosen=opt[1];
    for(int i=0;i<opt[2];i++){chosen=opt[1]+i;int w=EVENT_OUTCOMES[chosen][0];if(v<w)break;v-=w;}
    const int* outcome=EVENT_OUTCOMES[chosen];s[S_coins]+=outcome[1];
    if(outcome[2]) {
        if(s[S_node]>=3 && s[S_node]<=6)s[S_npc3_event]=1;
        int mode=outcome[3]==3?2:outcome[3]==2?1:0;
        int bound=outcome[3]==3?2+rng.below(2):0;
        set_source(s,outcome[2]-1,mode,bound,0,1);s[S_phase]=PH_DRAW;
    }
}
__device__ void apply_action(int* s,double* belief,RNG& rng,const int* a,
        const int* ci,const int* cp,const double* probs,const int* descs) {
    int kind=a[0];
    if(kind==AC_SELECT)select_card(s,a[1]);
    else if(kind==AC_REROLL){s[S_coins]-=40;s[S_phase]=PH_DRAW;clear_offer(s);}
    else if(kind==AC_BUY) {
        s[S_stock+(a[3]-1)*3+a[1]-1]--;s[S_coins]-=a[2];s[S_phase]=PH_DRAW;clear_offer(s);
        set_source(s,0,a[3]==2?1:0,0,1,1);
    } else if(kind==AC_ENHANCE) {
        s[S_coins]-=a[2];if(!s[S_free])s[S_paid]=imin(6,s[S_paid]+1);s[S_free]=0;
        set_source(s,0,0,0,1,0);enhance(s,belief,rng);
    } else if(kind==AC_RESTOCK){s[S_coins]-=100;s[S_restocks]--;shelf(s,rng,0,false);}
    else if(kind==AC_LEAVE) {
        s[S_phase]=ret_phase(s[S_shop_return]);s[S_source]=0;s[S_paid]=0;s[S_first]=0;clear_offer(s);
        for(int i=0;i<6;i++)s[S_stock+i]=0;
    } else if(kind==AC_EVENT_OPTION || kind==AC_SKIP_EVENT)event_option(s,a,rng);
    else if(kind==AC_ADVANCE)advance(s,belief,rng,ci,cp,probs,descs);
    else if(kind==AC_CONFIRM_CORE)s[S_phase]=PH_ADVANCE;
    else if(kind==AC_GENERATE_PAGE)regular_page(s,rng);
    else if(kind==AC_GENERATE_EVENT)event_draw(s,rng);
    else if(kind==AC_OPEN_SHOP)shelf(s,rng,s[S_first],true);
    else s[S_error]=4;
}
__device__ void forced(int* s,double* belief,RNG& rng,const int* ci,const int* cp,const double* probs,const int* descs) {
    while(s[S_phase]<PH_PAGE && !s[S_error]) {
        switch(s[S_phase]) {
            case PH_ADVANCE:advance(s,belief,rng,ci,cp,probs,descs);break;
            case PH_CORE:s[S_phase]=PH_ADVANCE;break;
            case PH_DRAW:regular_page(s,rng);break;
            case PH_EVENT_DRAW:event_draw(s,rng);break;
            case PH_SHOP_UNSEEN:shelf(s,rng,s[S_first],true);break;
        }
    }
}

__device__ bool same_card(const int* s,const int* a,const int* b) {
    if(a[0]!=b[0])return false;
    if(!a[0]){for(int i=1;i<=5;i++)if(a[i]!=b[i])return false;return true;}
    if(a[1]!=b[1] || a[2]!=b[2] || a[3]!=b[3] || a[5]!=b[5])return false;
    int pi=2*(a[1]-1)+a[2]-1,n=s[S_pool_n+pi];const int* p=s+S_pool+pi*9;
    for(int i=0;i<n-1;i++)if(p[i==a[6]-1?n-1:i]!=p[i==b[6]-1?n-1:i])return false;
    return true;
}
__device__ bool card_less(const int* a,const int* b) {
    if(a[0]!=b[0])return a[0]<b[0];
    const int order[5]={1,2,6,3,5};
    for(int i=0;i<5;i++){int j=a[0]?order[i]:i+1;if(a[j]!=b[j])return a[j]<b[j];}
    return false;
}
__device__ void push_action(int* a,int& n,int kind,int index=0,int price=0,int type=0) {
    // Preserve an invalid count for the status check without first writing past
    // the packed action row if a future rule table exceeds this ABI's capacity.
    if(n>=MAX_ACTIONS){n=MAX_ACTIONS+1;return;}
    int* out=a+4*n++;out[0]=kind;out[1]=index;out[2]=price;out[3]=type;
}
__device__ int legal(const int* s,int* a) {
    int n=0;
    if(s[S_phase]==PH_PAGE || s[S_phase]==PH_ENHANCE_PAGE) {
        int ids[3],count=0;
        for(int i=0;i<s[S_offer_n];i++) {
            bool duplicate=false;for(int j=0;j<count;j++)duplicate|=same_card(s,s+S_offer+7*i,s+S_offer+7*ids[j]);
            if(!duplicate)ids[count++]=i;
        }
        for(int i=1;i<count;i++){int x=ids[i],j=i;while(j>0 && card_less(s+S_offer+x*7,s+S_offer+ids[j-1]*7)){ids[j]=ids[j-1];j--;}ids[j]=x;}
        for(int i=0;i<count;i++)push_action(a,n,AC_SELECT,ids[i]+1);
        if(s[S_phase]==PH_PAGE && s[S_coins]>=40 && (!s[S_source] || s[S_reroll]))push_action(a,n,AC_REROLL,0,40);
    } else if(s[S_phase]==PH_SHOP) {
        bool can=false;for(int i=0;i<252;i++)can|=(i%6<5 && s[S_H+i]>0);
        int cost=s[S_free]?0:PRICES[imin(s[S_paid],6)];
        if(can && s[S_coins]>=cost)push_action(a,n,AC_ENHANCE,0,cost);
        for(int t=0;t<2;t++)for(int j=0;j<3;j++)if(s[S_stock+t*3+j]>0){if(s[S_coins]>=DRINK_PRICES[j])push_action(a,n,AC_BUY,j+1,DRINK_PRICES[j],t+1);break;}
        if(s[S_restocks]>0 && s[S_coins]>=100)push_action(a,n,AC_RESTOCK,0,100);
        push_action(a,n,AC_LEAVE);
    } else if(s[S_phase]==PH_EVENT) {
        int eid=s[S_event]-100,mask=s[S_event_options]&EVENT_MASKS[eid-1],valid=0;
        for(int j=0;j<10;j++)if((mask&(1<<j)) && s[S_coins]>=EVENT_OPTIONS[eid][j][0])valid|=1<<j;
        int keep=valid;for(int j=0;j<10;j++)if(EVENT_DOMINANCE[eid][j]&valid)keep&=~(1<<j);
        for(int j=1;j<10;j++)if((keep&(1<<j)) && s[S_coins]+EVENT_OPTIONS[eid][j][4]>=0)
            push_action(a,n,AC_EVENT_OPTION,EVENT_OPTIONS[eid][j][6],EVENT_OPTIONS[eid][j][0],s[S_event]);
        if(keep&1)push_action(a,n,AC_SKIP_EVENT,0,0,s[S_event]);
    }
    return n;
}

extern "C" __global__ void game_step(int* states,double* beliefs,u32* randoms,const int* actions,const int* choices,
        const int* ci,const int* cp,const double* probs,const int* descs,int n,int mode) {
    index_t i=index_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n)return;
    int* s=states+i*STATE_SIZE;if(s[S_phase]==PH_END || s[S_error])return;
    RNG rng{randoms+i,n};double* belief=beliefs+i*72;
    if((mode!=0 && mode!=1) || (mode==0 && (choices[i]<0 || choices[i]>=MAX_ACTIONS))){s[S_error]=7;return;}
    const int* a=mode==1?actions+i*4:actions+(i*MAX_ACTIONS+choices[i])*4;
    apply_action(s,belief,rng,a,ci,cp,probs,descs);forced(s,belief,rng,ci,cp,probs,descs);
    if(++s[S_steps]>1600)s[S_error]=5;
}
extern "C" __global__ void game_actions(const int* states,int* actions,int* counts,int n) {
    index_t i=index_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<n)counts[i]=legal(states+i*STATE_SIZE,actions+i*MAX_ACTIONS*4);
}
// Compact sizes and status reset share the existing enumeration launch.
// flags are consumed only by later kernels on the same CUDA stream.
extern "C" __global__ void game_actions_compact(const int* states,int* actions,int* counts,
        int* sizes,int* flags,int n) {
    index_t i=index_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<2)flags[i]=0;
    if(i<n) {
        int count=legal(states+i*STATE_SIZE,actions+i*MAX_ACTIONS*4);
        counts[i]=count;sizes[i]=count>1?count:0;
    }
}
extern "C" __global__ void game_start(int* states,double* beliefs,u32* randoms,
        const int* ci,const int* cp,const double* probs,const int* descs,int n) {
    index_t i=index_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n)return;
    RNG rng{randoms+i,n};forced(states+i*STATE_SIZE,beliefs+i*72,rng,ci,cp,probs,descs);
}

__device__ void encode_card(float* x,const int* c) {
    x[0]=1;x[1]=c[0];x[2+c[1]-1]=1;x[5+c[2]-1]=1;x[7+c[3]]=1;x[14+c[4]]=1;x[21+c[5]]=1;
    x[28]=(float)(c[6]/9.0);x[29]=(float)((imin(c[3],c[5])-imin(c[3],c[4]))/6.0);
    x[30]=(float)(imax(c[3]-c[5],0)/6.0);
}
__device__ int preferred(const int* s,const int* a,int n) {
    if(s[S_phase]==PH_PAGE || s[S_phase]==PH_ENHANCE_PAGE) {
        int best=-1,bv[4]={0};
        for(int i=0;i<n;i++)if(a[4*i]==AC_SELECT) {
            const int* c=s+S_offer+(a[4*i+1]-1)*7;
            int v[4]={imin(c[3],c[5])-imin(c[3],c[4]),c[3]>0,imax(c[3]-c[5],0)*c[0],-a[4*i+1]};
            bool better=best<0;for(int j=0;best>=0 && j<4;j++)if(v[j]!=bv[j]){better=v[j]>bv[j];break;}
            if(better){best=i;for(int j=0;j<4;j++)bv[j]=v[j];}
        }return best;
    }
    if(s[S_phase]==PH_SHOP) {
        int enhance=-1,buy=-1,leave=-1;
        for(int i=0;i<n;i++) {
            if(a[4*i]==AC_ENHANCE){enhance=i;if(a[4*i+2]==0)return i;}
            if(a[4*i]==AC_BUY && (buy<0 || a[4*i+2]<a[4*buy+2] || (a[4*i+2]==a[4*buy+2] && a[4*i+3]<a[4*buy+3])))buy=i;
            if(a[4*i]==AC_LEAVE)leave=i;
        }return buy>=0?buy:enhance>=0?enhance:leave;
    }
    int best=0;double value=-1e100;
    for(int i=0;i<n;i++){double v=a[4*i]==AC_SKIP_EVENT?0:EVENT_UTILITIES[s[S_event]-100][a[4*i+1]%100];if(v>value){best=i;value=v;}}
    return best;
}
__device__ void state_features(float* x,const int* s,const double* belief) {
    int tid=threadIdx.x;
    __shared__ int pool_scores[6];
    for(int i=tid;i<252;i+=blockDim.x)x[F_inventory+i]=(float)(s[S_H+i]/9.0);
    if(tid<42) {
        int p=tid/7,t=tid%7,h=0;
        for(int l=0;l<6;l++)h+=s[S_H+p*42+t*6+l];
        for(int j=0;j<s[S_pool_n+p];j++)h+=s[S_pool+p*9+j]==t;
        x[F_original_histogram+tid]=(float)(h/9.0);
    }
    if(tid<6) {
        int p=tid,off=(p/2)*12+(p%2)*3,sums[5]={0};
        for(int i=0;i<42;i++) {
            int n=s[S_H+p*42+i],t=i/6,l=i%6+1;
            sums[0]+=n;sums[1]+=n*l;sums[2]+=n*imin(t,l);sums[3]+=n*imax(t-l,0);sums[4]+=n*(t>l);
        }
        pool_scores[p]=sums[2];
        int n=s[S_pool_n+p],sum=0;
        for(int j=0;j<n;j++){int t=s[S_pool+p*9+j];sum+=t;x[F_ordered_pools+(off+j)*2]=(float)(t/6.0);x[F_ordered_pools+(off+j)*2+1]=1;}
        for(int j=0;j<5;j++)x[F_inventory_summary+p*8+j]=(float)(sums[j]/(j==0 || j==4?9.0:54.0));
        x[F_inventory_summary+p*8+5]=(float)(sum/54.0);
        x[F_inventory_summary+p*8+6]=(float)(n/9.0);
        x[F_inventory_summary+p*8+7]=(float)(((double)(float)(sum/(double)imax(1,n)))/6.0);
    }
    if(tid<24)x[F_event_posterior+tid]=(float)belief[tid];
    if(tid<48)x[F_coin_posterior+tid]=(float)(s[S_npc3_event] ? (tid%2?0:belief[24+tid]+belief[25+tid]) : belief[24+tid]);
    if(tid<s[S_offer_n])encode_card(x+F_offers+31*tid,s+S_offer+7*tid);
    if(tid<28)x[F_seen_events+tid]=(s[S_seen]>>tid)&1;
    if(tid<10 && s[S_event] && (s[S_event_options]&(1<<tid)))x[F_event_options+EVENT_OPTIONS[s[S_event]-100][tid][5]]=1;
    if(tid<16) {
        x[F_affordability+2*tid]=s[S_coins]>=THRESHOLDS[tid];
        x[F_affordability+2*tid+1]=fminf(3.f,fmaxf(-3.f,((float)s[S_coins]-(float)THRESHOLDS[tid])/500.f));
    }
    __syncthreads();
    if(tid!=0)return;
    int total_score=0;for(int i=0;i<6;i++)total_score+=pool_scores[i];
    x[F_phase+s[S_phase]]=1;
    x[F_position]=(float)(s[S_node]/46.0);x[F_position+1]=(float)(s[S_floor]/20.0);
    x[F_position+2]=(float)(s[S_shop_ordinal]/4.0);x[F_position+3]=(float)((46-s[S_node])/46.0);x[F_position+4]=(float)((20-s[S_floor])/20.0);
    x[F_return_phase+s[S_shop_return]]=1;
    x[F_wallet]=(float)(s[S_coins]/2000.0);x[F_wallet+1]=(float)(log1p((double)s[S_coins])/10.0);
    x[F_wallet+2]=s[S_free];x[F_wallet+3]=(float)(s[S_paid]/6.0);
    for(int i=0;i<6;i++)x[F_wallet+4+i]=(float)(s[S_stock+i]/8.0);
    x[F_wallet+10]=(float)(s[S_restocks]/2.0);x[F_wallet+11]=s[S_first];x[F_wallet+12]=(float)(PRICES[imin(s[S_paid],6)]/740.0);
    x[F_new_history]=s[S_P];x[F_new_history+1]=s[S_Q];x[F_new_history+2]=(float)(s[S_h]/10.0);
    if(s[S_source]) {
        x[F_source]=1;x[F_source+1+s[S_rarity]]=1;x[F_source+4+s[S_mode]]=1;
        x[F_source+7+s[S_bound]]=1;x[F_source+11+s[S_source_return]]=1;x[F_source+14]=s[S_reroll];
    }
    x[F_event_id+(s[S_event]?s[S_event]-100:0)]=1;
    x[F_event_history]=(float)(s[S_k]/4.0);x[F_event_history+1]=s[S_A];
    x[F_objective]=(float)(s[S_cap]/144.0);x[F_objective+1]=(float)(total_score/144.0);x[F_objective+2]=(float)((s[S_cap]-total_score)/144.0);
}

__device__ bool valid_compact_range(const int* counts,const int* offsets,int n,index_t i) {
    int count=counts[i];index_t previous=i?offsets[i-1]:0,current=offsets[i];
    return count>=0 && count<=MAX_ACTIONS && previous>=0 &&
           current==previous+(count>1?count:0) && current<=index_t(n)*MAX_ACTIONS &&
           current<=offsets[n-1];
}

__device__ void encode_game_features(const int* states,const double* beliefs,const int* actions,const int* counts,
        const int* offsets,float* features,int n) {
    index_t i=blockIdx.x;if(i>=n || counts[i]<2 || !valid_compact_range(counts,offsets,n,i))return;
    int count=counts[i];index_t start=index_t(offsets[i])-count;
    float* base=features+start*FEATURE_DIM;
    for(int j=threadIdx.x;j<FEATURE_DIM;j+=blockDim.x)base[j]=0;
    __syncthreads();
    const int* s=states+i*STATE_SIZE;const int* a=actions+i*MAX_ACTIONS*4;
    state_features(base,s,beliefs+i*72);
    __syncthreads();
    for(int j=threadIdx.x;j<(count-1)*FEATURE_DIM;j+=blockDim.x) {
        int feature=j%FEATURE_DIM;features[(start+1)*FEATURE_DIM+j]=feature<STATE_DIM?base[feature]:0;
    }
    __syncthreads();
    __shared__ int pref;
    __shared__ double prior;
    if(threadIdx.x==0) {
        pref=preferred(s,a,count);double remaining=imax(0,s[S_cap]-hscore(s));
        prior=fmin(fmax(0.0,remaining-.5),3.1*(20-s[S_floor])+s[S_coins]/180.0);
    }
    __syncthreads();
    if(threadIdx.x<count) {
            int j=threadIdx.x;
            float* x=features+(start+j)*FEATURE_DIM;const int* ac=a+j*4;int kind=ac[0],gain=0;
            x[F_action_kind+kind]=1;
            if(kind==AC_SELECT){const int* card=s+S_offer+(ac[1]-1)*7;encode_card(x+F_selected_offer,card);gain=imin(card[3],card[5])-imin(card[3],card[4]);}
            x[F_action_numbers]=(float)(ac[2]/740.0);x[F_action_numbers+1]=(float)((s[S_coins]-ac[2])/2000.0);
            x[F_action_numbers+2]=(kind==AC_SELECT || kind==AC_BUY)?(float)(ac[1]/9.0):0;
            x[F_action_numbers+3]=kind==AC_BUY?(float)(ac[3]/2.0):0;x[F_action_numbers+4]=ac[2]==0;x[F_action_numbers+5]=(float)(gain/6.0);
            if(kind==AC_EVENT_OPTION || kind==AC_SKIP_EVENT) {
                int eid=s[S_event]-100,slot=kind==AC_EVENT_OPTION?ac[1]%100:0;
                x[F_action_option+EVENT_OPTIONS[eid][slot][5]]=1;
                for(int z=0;z<10;z++)x[F_event_expectation+z]=EVENT_EXPECTATIONS[eid][slot][z];
            }
            x[F_heuristic_q_prior]=(float)((prior+.25*(j==pref))/20.0);
    }
}
__device__ void collect_game_status(const int* states,const int* counts,const int* offsets,
        int* flags,int n,index_t i) {
    if(i<n) {
        const int* s=states+i*STATE_SIZE;int count=counts[i];
        if(s[S_error])atomicMax(flags,s[S_error]);
        if(!valid_compact_range(counts,offsets,n,i))atomicMax(flags,7);
        if(count==0 && s[S_phase]!=PH_END)atomicMax(flags,6);
        if(count>0)atomicAdd(flags+1,1);
        if(i==n-1)flags[2]=offsets[i];
    }
 }
// Status shares the feature launch. Enumeration reset flags on the same stream;
// no block in this launch resets a value being atomically updated by another.
extern "C" __global__ void game_features_status(const int* states,const double* beliefs,
        const int* actions,const int* counts,const int* offsets,float* features,int* flags,int n) {
    index_t i=blockIdx.x;
    if(threadIdx.x==0)collect_game_status(states,counts,offsets,flags,n,i);
    encode_game_features(states,beliefs,actions,counts,offsets,features,n);
}
extern "C" __global__ void game_features(const int* states,const double* beliefs,
        const int* actions,const int* counts,const int* offsets,float* features,int n) {
    encode_game_features(states,beliefs,actions,counts,offsets,features,n);
}
// Status was checked before inference; only the original strict FP32 argmax remains.
extern "C" __global__ void game_choose_only(const int* counts,const int* offsets,
        const float* values,int* choices,int n) {
    index_t i=index_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n)return;
    choices[i]=0;int count=counts[i];if(count<2)return;
    index_t start=index_t(offsets[i])-count;int best=0;float maximum=values[start];
    for(int j=1;j<count;j++)if(values[start+j]>maximum){best=j;maximum=values[start+j];}
    choices[i]=best;
}
extern "C" __global__ void game_choose(const int* states,const int* counts,const int* offsets,const float* values,
        int* choices,int* flags,int n) {
    index_t i=index_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n)return;
    const int* s=states+i*STATE_SIZE;
    if(s[S_error])atomicMax(flags,s[S_error]);
    int count=counts[i];
    if(!valid_compact_range(counts,offsets,n,i)){atomicMax(flags,7);return;}
    choices[i]=0;
    if(count==0){if(s[S_phase]!=PH_END)atomicMax(flags,6);return;}
    atomicAdd(flags+1,1);
    if(count<2)return;
    index_t start=index_t(offsets[i])-count;int best=0;float maximum=values[start];
    for(int j=1;j<count;j++)if(values[start+j]>maximum){best=j;maximum=values[start+j];}
    choices[i]=best;
}
extern "C" __global__ void game_finish(const int* states,int* scores,int* steps,int n) {
    index_t i=index_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<n){scores[i]=hscore(states+i*STATE_SIZE);steps[i]=states[i*STATE_SIZE+S_steps];}
}

// Public-only hard-root descriptors. One launch replaces per-decision copies of
// full action/Q tables and dozens of small elementwise Torch launches.
// Category order is hard_examples.SPECIFIC_KINDS, then shop/page general.
extern "C" __global__ void game_hard_category(int* states,const int* actions,
        const int* counts,const int* choices,int* output,int n) {
    index_t i=index_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n)return;
    output[i]=0;int count=counts[i];int* s=states+i*STATE_SIZE;
    if(count<0 || count>MAX_ACTIONS){s[S_error]=7;return;}
    if(count<=1)return;
    int selected=choices[i];
    if(selected<0 || selected>=count){s[S_error]=7;return;}
    const int* a=actions+i*MAX_ACTIONS*4;const int* chosen=a+selected*4;
    if(s[S_phase]==PH_PAGE) {
        bool reroll=false,select_good=false,chosen_good=false;
        for(int j=0;j<count;j++) {
            const int* local=a+j*4;
            reroll|=local[0]==AC_REROLL;
            if(local[0]==AC_SELECT) {
                int slot=local[1]-1;
                if(slot<0 || slot>=s[S_offer_n]){s[S_error]=7;return;}
                const int* card=s+S_offer+slot*7;
                bool good=card[0]==1 && card[3]==3 && card[5]==3;
                select_good|=good;chosen_good|=j==selected && good;
            }
        }
        if(reroll && select_good) {
            if(chosen[0]==AC_REROLL){output[i]=2;return;}
            if(chosen[0]==AC_SELECT && chosen_good){output[i]=1;return;}
        }
    }
    if(s[S_phase]==PH_SHOP) {
        bool can_skip=false,buy=false,enhance=false;
        for(int j=0;j<count;j++) {
            const int* local=a+j*4;
            can_skip|=local[0]==AC_RESTOCK || local[0]==AC_LEAVE;
            buy|=local[0]==AC_BUY && local[2]==160;
            enhance|=local[0]==AC_ENHANCE && local[2]==180;
        }
        bool skipped=chosen[0]==AC_RESTOCK || chosen[0]==AC_LEAVE;
        if(can_skip && buy) {
            if(chosen[0]==AC_BUY && chosen[2]==160){output[i]=3;return;}
            if(skipped){output[i]=4;return;}
        }
        if(can_skip && enhance) {
            if(chosen[0]==AC_ENHANCE && chosen[2]==180){output[i]=5;return;}
            if(skipped){output[i]=6;return;}
        }
        output[i]=7;return;
    }
    if(s[S_phase]==PH_PAGE || s[S_phase]==PH_ENHANCE_PAGE)output[i]=8;
}
