const SHA = '46c5eca82b840745538e072bd20181e0b63036ca';
const BASE = `https://github.com/kvcache-ai/Mooncake/blob/${SHA}/`;
const sources = [
['overview','整体分层与可组合关系','docs/source/design/architecture.md',1],
['build','顶层源码构建默认值','CMakeLists.txt',6],
['te-build','USE_TENT 默认关闭','mooncake-common/common.cmake',231],
['te-switch','经典 API wrapper 如何切换 TENT','mooncake-transfer-engine/src/transfer_engine.cpp',534],
['te-api','Transfer Engine API / Batch 生命周期','mooncake-transfer-engine/tent/include/tent/transfer_engine.h',291],
['classic','经典 TE 的 MultiTransport 分派','mooncake-transfer-engine/src/multi_transport.cpp',362],
['register','内存注册与可用 transport 描述','mooncake-transfer-engine/tent/src/runtime/transfer_engine_impl.cpp',966],
['open','openSegment：打开远端地址空间','mooncake-transfer-engine/tent/src/runtime/transfer_engine_impl.cpp',754],
['selector','TENT 策略匹配、能力检查与 QP pool','mooncake-transfer-engine/tent/src/runtime/transport_selector.cpp',539],
['queue','runtime queue / progress worker 默认值','mooncake-transfer-engine/tent/src/runtime/transfer_engine_impl.cpp',413],
['submit','提交与 admission queue 的分支','mooncake-transfer-engine/tent/src/runtime/transfer_engine_impl.cpp',3009],
['failover','有预算和限制条件的 failover','mooncake-transfer-engine/tent/src/runtime/transfer_engine_impl.cpp',3349],
['transports','硬件后端加载与条件开关','mooncake-transfer-engine/tent/src/runtime/transport_loader.cpp',80],
['store-design','经典 Store 的 Master / Client 与部署模式','docs/source/design/store/mooncake-store.md',30],
['put','C++ Client::Put：申请、搬运、完成 / 撤销','mooncake-store/src/client_service.cpp',2070],
['get','C++ GET：副本选择、传输与 lease 检查','mooncake-store/src/client_service.cpp',1585],
['query','Master 返回可读副本与 read lease','mooncake-store/src/master_service.cpp',4770],
['upsert','独立的 UpsertStart 覆写生命周期','mooncake-store/src/master_service.cpp',5955],
['evict','淘汰前检查 hard pin 和 read / group lease','mooncake-store/src/eviction/evictor.cpp',154],
['store-flags','HA / offload 默认关闭','mooncake-store/src/master.cpp',238],
['tenant','租户配额按内存副本占用计量','mooncake-store/include/tenant/quota_policy.h',3],
['ha','HA 选主后端与编译条件','mooncake-store/src/ha/leadership/leader_coordinator_factory.cpp',12],
['promotion','Standby 晋升前导出完整上下文','mooncake-store/src/ha/leadership/master_service_supervisor.cpp',637],
['rs-readme','Rust 原生 Store 与 Python 后端选择','mooncake-store-rs/README.md',3],
['rs-default','Store-RS 默认 EmbeddedWrh / LocalOnly','mooncake-store-rs/crates/mooncake-store-client/src/client/builder.rs',147],
['rs-route','EmbeddedWrh 与 MetadataOnly 分支','mooncake-store-rs/crates/mooncake-store-route/src/shim/factory.rs',8],
['rs-arch','Rust authority mesh、路由与成员快照','docs/source/design/store/store-rs/architecture.md',49],
['rs-put','Rust 写入：reserve → transfer → publish route','mooncake-store-rs/crates/mooncake-store-client/src/client/runtime_io.rs',450],
['rs-quota','严格配额仍可访问 metadata backend','mooncake-store-rs/crates/mooncake-store-client/src/client/runtime_io.rs',120],
['rs-get','Rust get / get_into 复用 batch 路径','mooncake-store-rs/crates/mooncake-store-client/src/client/facade.rs',1600],
['rs-cold','Rust cold tier 显式开关，默认 false','mooncake-store-rs/crates/mooncake-store-client/src/client/cold_tier/mod.rs',20],
['rs-cold-design','Rust cold restore 的 DRAM staging 路径','docs/source/design/store/store-rs/cold-tier-design.md',197],
['conductor','Conductor::Query 返回索引查询结果','mooncake-conductor/src/kvevent/conductor_service.cpp',68],
['events','按引擎来源处理 KV 事件','mooncake-conductor/src/kvevent/event_handler.cpp',307],
['prefix','分实例、DP rank、存储层的前缀命中结构','mooncake-conductor/include/conductor/prefixindex/prefix_indexer.h',73],
['pg','PyTorch ProcessGroup 与集合通信 API','mooncake-pg/torch/include/mooncake_backend.h',108],
['pg-membership','PG 活跃 rank、epoch 与恢复接口','mooncake-pg/torch/include/mooncake_backend.h',150],
['ep','EP dispatch / combine 与 active ranks','mooncake-ep/include/mooncake_ep_buffer.h',132],
['p2p','P2P Store：Register / shard / Gold','mooncake-p2p-store/src/p2pstore/core.go',141],
['p2p-get','P2P Store：副本传输与位置登记','mooncake-p2p-store/src/p2pstore/core.go',426],
['reshard','权重布局规划：address-free placements','mooncake-reshard/python/mooncake/reshard/weight/_planner/api.py',16],
['binding','逻辑 plan 绑定运行时缓冲区','mooncake-reshard/python/mooncake/reshard/weight/_planner/binding.py',427],
['executor','Reshard TE 执行器与 completion fence','mooncake-reshard/python/mooncake/reshard/transfer_engine/executor.py',153],
['reshard-scope','Reshard copy-only 能力边界','mooncake-reshard/README.md',7],
['integration','Python engine / Store 绑定','mooncake-integration/CMakeLists.txt',170],
['ecosystem','本仓库记录的上游集成目录','README.md',211],
['metadata','Transfer Engine 元数据与 P2P handshake','docs/source/design/architecture.md',21],
['ssd','C++ Store SSD offload 设计','docs/source/design/store/ssd-offload.md',1]
].map(([id,title,path,line],i)=>({id,title,path,line,n:i+1,url:`${BASE}${path}#L${line}`}));
const components = [
{id:'framework',name:'推理框架 / 连接器',en:'APPLICATION SEMANTICS',summary:'决定哪些状态值得复用，以及怎样把模型状态映射成 Mooncake 的对象或地址区间。Mooncake 不负责执行 attention，也不替框架管理完整推理请求。',input:'请求 token、KV blocks、模型版本、并行布局与设备缓冲区。',output:'Store 的 key / value 操作，或 TE 的 READ / WRITE 请求。',points:['prefix hash、模型与租户隔离、KV block 布局由集成层衔接。Store 看到的是对象 key 和字节，不会凭内容自动理解 token。','PD 分离可直接用 TE 搬运 prefill 产生的 KV；跨请求共享缓存可通过 Store 复用；二者可组合但不是同一功能。','mooncake-integration/ 主要提供语言绑定。实际 serving connector 很多在 vLLM、SGLang、LMCache 等上游仓库。'],boundary:'本页审计 Mooncake 仓库及其集成说明，不把所有外部 connector 的行为视为已逐行验证。',refs:['integration','ecosystem','overview']},
{id:'master',name:'Store Master',en:'C++ STORE · CONTROL PLANE',summary:'维护对象到副本位置的目录，决定空间分配、可读状态与回收。它返回数据位置，客户端据此传输。',input:'PutStart / PutEnd、GetReplicaList、MountSegment、心跳与管理请求。',output:'副本描述符、分配位置、读租约与对象状态。',points:['PutStart 先分配副本，客户端写完后 PutEnd 标记完成；失败走 PutRevoke。已存在对象的普通 Put 可以直接成功，覆写由 Upsert 处理。','Master 同时负责 segment 成员关系、对象淘汰、可选租户配额；配额按内存副本占用计量，不是逻辑对象大小只算一次。','默认单 Master；HA 可通过选主和 standby 恢复提升可用性。选主成功后还需完成元数据恢复与领导权复核。'],boundary:'Master 不转发对象 payload。Master HA 保护控制面，与数据多副本解决的问题不同。',refs:['put','query','tenant','ha','promotion']},
{id:'client',name:'Store Client / Storage',en:'C++ STORE · OBJECT DATA PATH',summary:'Client 既可以发起对象请求，也可以贡献存储 segment。同一个类承担两种角色，部署时可以拆开。',input:'应用的对象请求；本地 DRAM / accelerator buffers；Master 返回的副本位置。',output:'直接落到本地或远端副本的 payload；读取完成的数据。',points:['嵌入式 Client 与推理进程同驻；独立 Store service 包装 Client，专门贡献内存或存储容量。','global_segment_size=0 可只消费共享池；dummy / real 模式让多个推理 rank 共享一个持有资源的 real client。','GET 查询可读副本与 lease，选择副本后经 TE 直读；可选 local hot cache、DFS / SSD 等路径改变实际数据流。'],boundary:'客户端直传不等于每个 API 在每种硬件上零拷贝。是否 staging、memcpy 或设备直达取决于缓冲区与后端。',refs:['store-design','get','put','ssd']},
{id:'te',name:'Transfer Engine',en:'REGISTER · OPEN · SUBMIT · POLL',summary:'将注册的内存区域暴露为可访问的 segment，统一提交跨进程、跨节点的批量字节传输。它是可独立使用的库。',input:'本地地址、远端 segment、目标地址 / offset、长度，以及 READ / WRITE 操作。',output:'异步 task / batch 状态，以及完成后的目标缓冲区。',points:['registerLocalMemory 注册地址范围和可用传输能力，不是把 KV 上传到 Store。openSegment 打开远端地址空间，也不是分配对象。','经典 MultiTransport 根据远端描述符和地址找到后端，将相同 transport 的任务聚合提交。','注册 → open → allocateBatch → submit → poll/wait → free。submit 返回成功只是接收请求，不能提前复用或释放仍被 DMA 使用的内存。'],boundary:'TE 不管理 prefix hash、对象淘汰或副本可见性。传输元数据服务也不等于 Store Master。',refs:['classic','register','open','te-api','metadata']},
{id:'tent',name:'TENT',en:'TRANSFER ENGINE NEXT · OPTIONAL',summary:'TE 的可选运行时实现，把 transport 选择、能力检查、分片调度与有限重试收进运行时。',input:'传输请求、内存类型与拓扑、policy、priority / intent、可选 deadline。',output:'匹配的 transport / QP pool、可选排队任务，以及完成或错误。',points:['USE_TENT 在源码构建中默认 OFF。编译后经典 wrapper 还需 MC_USE_TENT 等显式启用；也可使用 tent 原生 API。','selector 使用第一条匹配规则，并检查后端可用性与 buffer 注册能力。显式 hint 受到候选集合约束，不保证自动选到“最快路径”。','runtime queue、progress worker、deadline-aware 调度默认关闭。RDMA QoS 依赖已配置的 QP pool，policy 的 SL / TC 字段不是逐请求修改 QP。','failover 有预算和禁止条件。写入结果不确定时不能安全盲重试，因此不是 RDMA 出错就一定改走 TCP。'],boundary:'源码有实现 ≠ 编译包含 ≠ 配置开启 ≠ 当前内存与硬件能够使用。',refs:['te-build','te-switch','selector','queue','failover','transports']},
{id:'rs',name:'Rust 原生 Store',en:'AUTHORITY MESH · ALTERNATIVE',summary:'另一条对象存储实现路线：正常路由由客户端间的 authority mesh 维护，TE / TENT 继续负责 payload。',input:'对象请求、成员租约快照、存储 owner 信息与副本策略。',output:'ObjectRoute、CAS 版本与可读取的数据副本。',points:['默认 EmbeddedWrh 通过 weighted rendezvous hashing 选 key 的路由 authority；主 authority 做 CAS，其他 authority 镜像路由。路由镜像不等于数据副本。','PUT 预留位置 → 写副本 → CAS 发布 ObjectRoute。GET 解析路由后直接读取；存储 owner 管本地 replica 与回收。','MetadataBackend 保留租约、segment、策略等职责；普通路由查询离开后端，但严格配额、成员刷新等条件路径仍可访问后端。','builder 默认写模式是 LocalOnly；分布式放置与 cold tier 需要相应配置。不能因“无专用 Master”就推断默认自动构成远端存储池。'],boundary:'WITH_STORE_RS 是原生实现；WITH_STORE_RUST 是 C++ Store 的 Rust 绑定。Python 默认后端仍为 cpp。',refs:['rs-readme','rs-default','rs-route','rs-put','rs-quota']},
{id:'conductor',name:'Conductor',en:'CACHE INDEX · ROUTER INPUT',summary:'汇总哪些实例、DP rank、存储层持有哪些可复用前缀，为上层缓存感知路由提供查询服务。',input:'vLLM / SGLang / Store 的 Stored、Removed、Cleared 等 KV 事件；查询 token_ids。',output:'每个候选实例在 device / cpu_local / cpu_share / disk 上的前缀命中长度。',points:['事件订阅与来源适配 → KVEventHandler → PrefixCacheTable → Query。HTTP / RPC 支持注册服务、查询命中和全局视图。','索引 context 区分 tenant、model、LoRA 与 block size；不能把不同模型的同一 token 序列直接视为可共享 KV。','Query 返回索引结果，当前公开实现不在这里排队推理请求、分配 prefill/decode 计算或综合 SLO 选节点。'],boundary:'缓存命中信息是路由决策的输入。事件传播、重放与失联会影响索引新鲜度，最终可读性仍应由实际对象/框架路径确认。',refs:['conductor','events','prefix','build']},
{id:'pg',name:'PG · Process Group',en:'MEMBERSHIP & COLLECTIVES',summary:'PyTorch 分布式通信后端，连接集合通信、成员状态和恢复流程。它服务于分布式执行，不是 Store 的复制协议。',input:'torch.distributed 集合通信 / send-recv；通信组、rank 与健康状态。',output:'collective 结果、active ranks、epoch 与故障同步状态。',points:['提供 allreduce、allgather、alltoall、broadcast 等接口；具体算子仍有类型、布局和设备约束。','控制面跟踪稳定 rank slot 与活跃成员，支持 activate/deactivate、join 与 syncAfterFailure。','数据路径利用 TE 及设备通信路径；成员变化后，EP 与推理引擎还需协调重连和继续执行。'],boundary:'通信层能报告和恢复 rank，不代表模型语义、专家冗余或请求恢复策略已经自动解决。',refs:['pg','pg-membership','overview','build']},
{id:'ep',name:'EP · Expert Parallel',en:'MOE TOKEN DISPATCH / COMBINE',summary:'为 MoE 推理提供 expert-parallel token 分发与结果合并，并感知活跃 rank。',input:'token tensors、top-k expert indices / weights、active ranks。',output:'发往专家的 token 分片，以及 combine 后的结果。',points:['dispatch 将 token 发往目标专家，combine 按路由信息合并结果；支持异步和接收 hook 等接口。','有节点内 P2P 与跨节点 RDMA device fast path；可以复用传入 TE 的 transport，也可自己持有 transport。','通常由 PG 供给成员状态与启动信息；恢复 rank 加入前需刷新 peer metadata 与通信 buffer 状态。'],boundary:'EP 并非每一步都走普通 CPU submitTransfer。对失效专家如何替补或重排，仍需要上层引擎配合。',refs:['ep','pg-membership','build']},
{id:'reshard',name:'Reshard',en:'LAYOUT → PLAN → BIND → TRANSFER',summary:'把资源从一种并行布局迁移到另一种布局所需的拷贝关系，转成可检查、可绑定的传输计划。',input:'框架提供的规范化 manifest、源/目标 placement、资源版本与运行时 buffer 快照。',output:'逻辑 N 维传输区域 → 绑定后的地址区间 → Store 或 TE 执行。',points:['权重 planner 检查 resource / revision / generation 一致性，描述 TP / PP / EP / DP 的逻辑放置，不直接嵌入运行时地址。','binding 将逻辑计划与不可变的运行时 buffer 快照核对，执行器再提交地址区间并确认 completion fence。','代码也包含 KV cache 的规划与 Store upload/load 适配路径，语义由上层构造的 manifest 表达。'],boundary:'copy-only 规划不自动转换 dtype、量化、packing 或任意 checkpoint 格式。数据完成前必须保持 pin / reservation。',refs:['reshard','binding','executor','reshard-scope']},
{id:'p2p',name:'P2P Store',en:'PEER-ASSISTED SHARD DISTRIBUTION',summary:'独立的 Go 分片传播服务，把大 payload 注册为多个 shard，允许接收者从已有副本继续拉取。',input:'本地已注册大块数据、分片参数，以及 etcd 中的 shard 位置。',output:'新增 peer replica 与更新后的 ReplicaList。',points:['Register 划分 shard，记录原始 Gold 位置；GetReplica 查找可用来源，通过 TE 把分片拉到本地。','已产生副本也可以成为来源；源失败后可尝试其他 Replica 或 Gold 位置。成功后更新副本元数据。','源码顶层默认不构建；README 将另一高性能实现指向 MoonshotAI/checkpoint-engine。'],boundary:'它不是经典 Store 的“无 Master 部署模式”，也不是 Rust Store authority mesh。三条路径需要分别理解。',refs:['p2p','p2p-get','build']},
{id:'metadata',name:'元数据与存储介质',en:'DISCOVERY IS NOT THE PAYLOAD',summary:'同样出现 etcd、Redis 或 HTTP，不代表它们在每个组件中承担相同职责。先问它保存的是对象目录、连接描述还是成员租约。',input:'TE segment / endpoint 描述；Store HA 选主状态；Store-RS 成员租约与策略。',output:'可发现的连接与控制状态；数据本身仍留在注册内存或配置的存储层。',points:['TE 可以用元数据服务发现 peer，也可采用 P2PHANDSHAKE。它不把 tensor payload 存进 etcd。','经典 Store 的 Master 对象目录与 TE 的 segment 元数据是不同层；HA 的选主后端也不能替代对象数据副本。','DRAM 是热存储主路径。SSD / NVMe、分布式文件或对象存储等冷层改变了恢复与 staging 路径，需单独配置。'],boundary:'“支持 SSD”不等于 CPU/GPU 可以直接 RDMA 读取普通文件；“缓存池”也不自动等于持久数据库。',refs:['metadata','ha','ssd','rs-cold-design']}
];
const scenarios = [
{id:'all',name:'完整图谱',nodes:components.map(x=>x.id),title:'可组合的层次',text:'从上往下看：框架提供 KV / 权重语义，Store 提供对象服务，TE 承担传输。Conductor 提供缓存索引，PG / EP 与 Reshard 服务于特定执行和布局需求。并非每次请求都经过全部组件。'},
{id:'pd',name:'Prefill → Decode',nodes:['framework','te','tent','metadata'],title:'直接搬 KV',text:'Prefill 生成 KV 后，上层交换位置与就绪信息，TE 将数据送到 Decode 的已注册缓冲区。此路径可独立于 Store；调度、KV 格式兼容和开始 decode 的时机由框架负责。'},
{id:'cache',name:'共享 KV 缓存',nodes:['framework','master','client','te','tent','metadata','conductor'],title:'复用已有对象',text:'连接器把 KV blocks 映射为对象。经典 Store Master 给出副本位置，Client 通过 TE 读取；Conductor 可以帮助路由器识别前缀命中较多的实例。'},
{id:'rust',name:'Rust Store',nodes:['framework','rs','te','tent','metadata'],title:'路由权威节点网格',text:'Rust Store 默认使用 EmbeddedWrh 管理路由 authority。它与经典 Master 不是上下级；对象路由走 authority mesh，payload 仍由 TE / TENT 传输。默认 LocalOnly 写入也需与远端放置配置区分。'},
{id:'weight',name:'权重 / 重分片',nodes:['framework','reshard','master','client','te','tent','metadata'],title:'先描述布局，再搬字节',text:'Reshard 从源、目标 placement 生成 copy plan，绑定运行时地址后通过 TE 或 Store 执行。另有独立 P2P Store 面向大 payload 分片传播，不是此流程的必经节点。'},
{id:'moe',name:'弹性 MoE',nodes:['framework','pg','ep','te','metadata'],title:'通信与成员一起变化',text:'PG 管通信组与成员状态，EP 执行 token dispatch/combine。EP 可以使用设备侧快速通路，不能把它画成每步都经普通 TE 提交接口。专家替补与请求语义仍由引擎处理。'}
];
const flows = [
{id:'put',name:'C++ PUT',subtitle:'经典 Store · 正常内存副本写入路径',caveat:'普通 Put 是缓存插入：对象已存在时可直接成功。覆写另有 Upsert；磁盘/NoF/DFS 路径和批处理省略于此。',steps:[
['准备对象','应用 → Client','应用侧','data','框架把 KV 或权重映射成 key 与 slices，准备稳定且可传输的源缓冲区。对象语义和模型版本隔离来自集成层。','Client::Put(key, slices, config)',['put','integration']],
['申请位置','Client → Master','控制面','control','PutStart 请求副本空间。Master 检查对象状态、可选配额并分配，返回目标 segment、地址和副本描述符。此时新对象还不能当作写入完成。','PutStart → replica descriptors',['put','tenant']],
['直写副本','Client → Storage Client','数据面','data','Client 依据描述符写入副本。payload 在客户端与存储节点之间移动，绕过 Master；本地与远端位置走相应传输路径。','TransferWrite / batch transfer',['put','te-api']],
['确认完成','Client ← Transport','数据面','data','等待数据传输结束。submit 成功只说明请求已受理；完成前源缓冲区和副本预留必须保持有效。','transfer completion',['put','submit']],
['发布可读','Client → Master','控制面','control','成功调用 PutEnd，把已完成副本发布为可读状态；失败进入 PutRevoke 等清理分支。Master 管理可见性，TE 管理搬运完成。','PutEnd / PutRevoke',['put']]
]},
{id:'get',name:'C++ GET',subtitle:'经典 Store · 查询位置 → 保护读取 → 直读',caveat:'GET 可能命中 local hot cache，或走 DFS / SSD 路径；图中突出普通可读内存副本。缓存 miss 需要上层决定重新计算或回源。',steps:[
['定位对象','Client → Master','控制面','control','Query / GetReplicaList 获取 complete、可读的副本列表与 read lease。仅知道 key 存在还不等于任意副本都能读取。','Query → readable replicas + lease',['query','get']],
['选择副本','Client','客户端策略','control','客户端在可读副本中选择一个来源；可选 local hot cache 可缩短路径。选择副本与按 transport 选网卡是不同层的决策。','select readable replica',['get']],
['读取字节','Storage Client → Client','数据面','data','通过 TransferRead 将 payload 从副本所在 segment 读入目标缓冲区。Master 不经过这些字节。','TransferRead → completion',['get','te-api']],
['验证返回','Client → 应用','完成检查','control','传输成功后检查配置的 checksum 与 read lease 是否过期，再将成功结果返回。失败、缺失或租约问题不能当作有效 KV 命中。','checksum / lease validation',['get','evict']]
]},
{id:'te',name:'TE / PD 搬运',subtitle:'独立使用 Transfer Engine · 不要求 Store',caveat:'TE 只负责地址区间传输。Prefill/Decode 的 KV 布局、就绪通知、失败后的请求重试仍由推理框架负责。',steps:[
['注册内存','Source / Destination','资源准备','control','双方准备并注册可访问缓冲区。运行时记录内存位置与成功注册的 transport，发布相应 segment 描述。','registerLocalMemory',['register']],
['打开远端','发送 / 接收进程','位置发现','control','上层获得目标位置后 openSegment，拿到远端地址空间 handle。这个调用不分配 Store 对象，也不验证 token 语义。','openSegment(segment_name)',['open','metadata']],
['提交请求','应用 → TE','传输 API','data','分配 batch，给出 READ/WRITE、source、target_id、target_offset、length。多个区域可以聚合提交。','allocateBatchID → submitTransfer',['te-api','classic']],
['匹配执行','TE / TENT → transport','数据面','data','经典 TE 分派 transport；可选 TENT 根据策略、内存能力与拓扑选路。需要时走 host staging，不承诺任意硬件端到端直达。','transport selection → execution',['selector','transports']],
['完成再消费','TE → 框架','完成边界','control','poll / wait 确认完成后，框架才能让 Decode 安全消费数据，并释放传输 batch。跨引擎通知不是 TE 自动执行推理。','poll / wait → freeBatchID',['te-api','submit']]
]},
{id:'rust',name:'Rust PUT',subtitle:'Store-RS · reserve → transfer → route publication',caveat:'此流程描述有配置的目标副本写入。builder 默认 LocalOnly；正常 EmbeddedWrh 路由不走 metadata backend，但严格配额等条件路径仍可能访问后端。',steps:[
['解析路由','Client → Authorities','控制面','control','依据成员快照与 WRH 顺序找到当前 key 的 authority，获取现有路由及版本，确定此次写入的预期版本。','RouteOperations / expected_version',['rs-route','rs-arch']],
['预留资源','Client → Storage Owners','控制面','control','按放置策略预留本地或远端 segment 空间；启用严格配额时先完成对应 quota reservation。','reserve_replica_targets',['rs-put','rs-quota']],
['写入副本','Client → Storage Owners','数据面','data','将字节写到预留的 replica 位置，并等待传输完成。此时 payload 已存在，但新路由尚未正式发布。','write_reserved_replicas',['rs-put']],
['CAS 发布','Client → Authorities','控制面','control','构造 ObjectRoute，以 expected_version 执行发布。路由可见性与数据完成被明确拆开；失败时清理预留并中止配额记账。','publish_route(expected_version)',['rs-put','rs-arch']]
]},
{id:'index',name:'缓存感知路由',subtitle:'Conductor · 事件索引不是请求调度',caveat:'该流程最后的负载/SLO 决策属于外部路由器，不在 Conductor::Query 内完成。事件索引的命中视图也不是读取成功承诺。',steps:[
['产生事件','引擎 / Store','控制信息','control','缓存新增、删除或清空后，发布 KV 事件。不同引擎来源需要适配，事件带有 context 与 block 信息。','Stored / Removed / Cleared',['events']],
['更新索引','Subscriber → PrefixCacheTable','控制信息','control','Conductor 接收事件，按 tenant/model/LoRA/block size 等上下文更新实例、层级与 DP rank 的 prefix index。','KVEventHandler → PrefixCacheTable',['events','prefix']],
['查询命中','Router → Conductor','控制信息','control','路由器提交 token_ids 等信息，得到候选实例在 device、host、shared pool、disk 各层的前缀命中长度。','ConductorService::Query',['conductor','prefix']],
['决定路由','外部 Router → Engine','上层决策','control','外部路由器结合缓存命中、队列负载与调度目标选择执行实例。Mooncake 在此提供数据依据，而非承包整套调度。','external scheduling decision',['conductor']]
]},
{id:'reshard',name:'权重重分片',subtitle:'Reshard · 把布局转换成字节拷贝计划',caveat:'planner 为 copy-only，不自动猜张量名称语义或转换量化格式；框架必须提供正确 manifest 并保证设备流与缓冲区生命周期。',steps:[
['声明布局','Framework','逻辑层','control','框架给出规范化资源 manifest、源 placement 与目标 placement。明确资源、revision、generation 和并行切分。','source / target placement',['reshard','reshard-scope']],
['规划区域','Reshard Planner','逻辑层','control','根据逻辑布局相交关系产生 N 维传输区域。此阶段不把进程虚拟地址混入布局契约。','plan_placement_transfer',['reshard']],
['绑定地址','Runtime Binding','绑定层','control','用运行时 buffer 快照核对计划，将逻辑区域绑定成实际可访问地址区间，同时保留生命周期约束。','bind_logical_transfer_plan',['binding']],
['执行完成','Executor → TE / Store','数据面','data','提交物理 batch，等待 completion fence。执行器搬地址区间，不拥有权重或 KV 的模型语义。','MooncakeTransferEngineExecutor',['executor']]
]}
];
const comparisons = [
['控制面中心','独立 Master 管对象目录与分配。','EmbeddedWrh 默认选择路由 authorities；storage owner 管本地存储。'],
['正常路由查询','Client 查询 Master 的可读副本与 lease。','RouteOperations 经 authority mesh 查路由；MetadataOnly 是替代模式。'],
['写入可见性','PutStart → 写副本 → PutEnd；失败 Revoke。','预留 replica → 写数据 → CAS 发布 ObjectRoute。'],
['后端职责','TE 连接元数据、Master HA 选主等各有分工。','MetadataBackend 保存租约、segment、策略；严格配额等路径可能访问后端。'],
['回收与冷层','淘汰先检查 hard pin / read lease / group lease；SSD offload 可选。','Storage owner 管回收；cold tier 默认关闭，远端冷读可先 stage 到 DRAM。'],
['使用入口','Python 默认 cpp；源码 WITH_STORE 默认 ON。','显式选择 rs；WITH_STORE_RS 默认 OFF，builder 写模式默认 LocalOnly。'],
['阅读依据','Client::Put / Get、MasterService、evictor。','builder、RouteOperations、runtime_io、cold_tier。']
];
const boundaries = [
['01','“元数据服务”不是一个东西','Store Master 维护对象目录；TE metastore 维护 segment / endpoint；HA backend 管领导权；Store-RS backend 管租约和策略。不能把它们画成一台万能 etcd。','metadata'],
['02','缓存复用与 PD 分离是两条轴','共享 Store 减少跨请求重复计算；直接 TE 传输连接 Prefill 与 Decode。前者关心对象生命周期，后者关心本次推理的 KV 就绪。','overview'],
['03','HA、多副本、重试各管一层','Master HA 保护元数据服务；对象多副本提高数据可用性；transport failover 绕开可恢复的传输失败。任何一个都不能代替另外两个。','ha'],
['04','TENT 是可选运行时，QoS 也需配置','源码支持 intent / priority、QP pool 和 deadline 并不意味着默认全开。写结果不确定等场景会禁止重放，不承诺“所有失败透明恢复”。','queue'],
['05','缓存不是无条件持久存储','经典 SSD offload 与 Rust cold tier 都需开启。Rust foreground write 默认先满足 DRAM 与路由发布，不等待后台 cold 持久化；不要据此承诺掉电不丢数据。','rs-cold-design'],
['06','“支持某硬件”是一组条件','后端需被编译、启用并成功注册相应内存。直接路径不可用时可能 staging；本页没有运行硬件集群，不用 README 峰值代替当前环境测量。','transports']
];
const choices = [
['只想拆开 Prefill / Decode','先读 TE + 框架 connector','KV 布局一致性、就绪通知、调度和失败请求恢复。'],
['多个实例共享 prefix cache','C++ Store 或 Store-RS + TE；Conductor 可选','hash / 模型版本隔离、缓存收益、miss 回源与路由策略。'],
['MoE 局部故障后继续服务','PG + EP + 推理引擎适配','专家冗余、active-rank 变化、语义与恢复流程。'],
['频繁分发 / 切换模型权重','Store / TE；涉及布局变化时看 Reshard','规范化 manifest、权重版本、设备流和内存生命周期。'],
['大 checkpoint 的 peer 传播','单独研究 P2P Store / checkpoint-engine','分片源选择、完整性、版本一致性与集群部署。']
];
const failures = [
['找不到缓存 / 命中率不对','先核对连接器的 key、模型版本与 context，再看对象是否被回收；若只是路由器视图异常，检查 Conductor 事件与注册状态。'],
['能定位对象但搬不动','检查 segment 描述、注册内存范围、transport 可用性和任务状态。TENT selector 拒绝与底层网络失败是不同原因。'],
['故障后服务无法恢复','先分清控制面 Master、数据 storage owner 还是 PG rank 出错。检查恢复上下文、存活副本或 membership，不能只看“选主成功”。']
];
const glossary = [
['Segment','TE 中是可访问地址空间；Store 把节点贡献的存储范围纳入资源池。它不是 KV token block，也不是对象 key。'],
['Slice / Batch','Slice 是对象或传输的子区间；Batch 聚合多个异步任务。Batch 提交与所有任务完成是两个时刻。'],
['Replica','对象的一份可读数据副本。Rust route authority 的元数据镜像与 payload replica 要分别理解。'],
['Read lease / Pin','读租约在一段时间内保护正在使用的数据；hard pin / soft pin 是另一类生命周期策略。它们不等于永久持久化。'],
['Authority / CAS','Authority 保存对象路由；CAS 只在预期版本匹配时更新，避免并发更新把路由直接覆盖。'],
['WRH','Weighted Rendezvous Hashing，加权 rendezvous 哈希。Store-RS 依活跃成员与 key 排序选择路由权威节点。'],
['Staging','经中间缓冲区分段搬运，例如设备 → host → 网络。可扩展兼容性，但不等于端到端零拷贝。'],
['Dispatch / Combine','MoE 把 token 派发到专家并合并结果的通信操作，与 Store Put/Get 是不同接口。'],
['Manifest / Binding','Manifest 描述资源的逻辑布局；binding 把逻辑拷贝区域落实到运行时缓冲区与生命周期约束。']
];
