---
title: MRC 与 SRv6：OpenAI 如何把百 K GPU 训练网络做成端侧自愈系统
date: 2026-05-07 12:00:00 +0800
author: Ethan
kind: reading
category: Reading
intro: OpenAI 的 MRC 论文把百 K GPU 训练网络的主线从平均带宽推进到尾延迟治理：让端侧协议主动绕开拥塞和故障，交换机控制面反而变得更静态。
tags: [Networking, Training]
---

OpenAI 这篇 MRC 论文最重要的信号，是大模型训练网络的竞争焦点已经从"能不能把平均带宽堆上去"，转向"能不能在故障常态化时稳住同步训练的尾部"。在十万级 GPU 的 pretraining job 里，一轮 collective 的完成时间由最慢那条 transfer 决定；一个 link flap、一次 ECMP 碰撞、一次交换机灰故障，都可能被同步屏障放大成整批 GPU 的等待。

论文给出的答案很有工程侵略性：把路径选择从交换机动态路由里拿出来，交给 NIC 侧的 MRC 传输协议；交换机只保留 SRv6 静态转发；MRC 在端侧把一个 Queue Pair 的包喷洒到所有 plane 和大量 path 上，持续根据 ECN、SACK/NACK、probe 结果撤换坏路径<a href="https://cdn.openai.com/pdf/resilient-ai-supercomputer-networking-using-mrc-and-srv6.pdf">[1]</a>。它已经超出 RoCE 调参的范畴，实质上把 AI 后端网络的失败恢复职责整体前移到端侧协议。

## 一、问题不在带宽总量，而在同步训练的失败放大

同步预训练的网络形态和普通数据中心流量很不同。数据并行、张量并行、流水线并行、专家并行会在每个 step 中交织出现，训练系统希望通信尽量被计算覆盖；一旦某一轮通信出现尾部 outlier，GPU 就会等在 barrier 上。OpenAI 官方文章把这类 workload 形容成"failure amplifier"：集群越大，单个链路或交换机异常越容易被作业级同步机制放大<a href="https://openai.com/index/mrc-supercomputer-networking/">[2]</a>。

传统 RoCE 体系通常依赖 lossless Ethernet、PFC、DCQCN 和 ECMP。这个组合在中等规模集群里可用，但在百 K GPU 后端网络里会暴露两个问题。第一，单条 transfer 往往绑定到单一路径，很多 flow 在 hash 后撞到同一条 uplink，会制造 core congestion。第二，动态路由在故障时需要重新收敛；如果端侧传输协议也在根据拥塞信号做调整，两套自适应机制可能互相干扰，造成难以解释的路径变化。

MRC 的设计目标因此变成三件事：让流量在网络内部足够均匀，让 incast 不把尾延迟扩散成全局抖动，让 link 或 fabric failure 不再直接打断训练作业。

## 二、多平面拓扑：先把故障切成更小的份额

论文的第一个关键选择是拓扑 co-design。以 100,000 GPU、每 GPU 一张 800 Gb/s NIC 为目标，如果按传统 800 Gb/s single-plane Clos 来做，当前 51.2 Tb/s switch 只有 64 个 800G 端口，三层 Clos 只能自然扩到约 64K NIC；继续扩展就要四层、超卖，或者拆成多套独立 rail。

![三层 800G single-plane 与两层 8x100G multi-plane 拓扑对比](/assets/openai-mrc-srv6-supercomputer-networking/fig-1-multiplane-topology.png)

*图 1：把一张 800G NIC 拆成 8 个 100G plane 后，同样的 51.2 Tb/s switch 可以提供 512 个 100G 端口，使 131,072 NIC 的网络保持两层 Clos。来源：论文 Figure 1。*

MRC 论文选择把 800G NIC 按 lane 拆成 8 个 100G 端口，构造 8 个并行 plane。这样 T0 switch 可以连接 256 个 NIC port，上行到 256 个 T1；T1 再连回 512 个 T0，形成 131,072 GPU 的两层网络。论文还给出几个直接收益：最长路径只过三台交换机；更多通信可以保持 T0-local；满二分带宽下，所需 optics 约为三层 800G 方案的 2/3，交换机数量约为 3/5。

更关键的是故障颗粒度。论文指出，T0-T1 链路故障在 800G plane 中会让一个 node 失去约 3% 容量，而在 100G multi-plane 中只损失约 0.4%。如果 NIC-T0 链路故障，single-plane 通常意味着训练作业受影响；multi-plane 下则是损失一个 plane 的带宽。OpenAI 在生产里观察到，许多这样的事件可以让作业继续跑，不必立刻 eviction。

拓扑本身并不能解决负载均衡。拆成 8 个 plane 之后，如果传输协议仍把一个 flow 固定到单一路径，path diversity 只存在于图纸上。MRC 的作用，就是把这些 plane 和 path 真正变成单个 transfer 可用的资源池。

## 三、MRC：让一个 QP 同时使用 128 到 256 条路径

MRC 是 RoCEv2 Reliable Connection 语义层的扩展，仍保留 verbs API 和 Queue Pair 抽象，但论文明确说，当前 AI workload 只需要 RDMA write 与 write-with-immediate 两类操作。这个收窄很重要：MRC 没有试图覆盖完整通用 RDMA 语义，而是围绕训练通信里最重要的写路径做硬件协议。

它的核心机制是 Entropy Value（EV）。每个 MRC packet 带一个 32-bit EV，EV 会影响包在网络里的路径选择。QP 启动时，sender 为这个 QP 生成一组 EV，通常是 128 到 256 个；发送包时轮流使用这些 EV，于是同一个 QP 的包会被喷洒到所有 plane 和大量 path 上。因为每个数据包都携带 RDMA virtual address 和 remote key，接收端 NIC 可以把乱序到达的包直接写入最终内存位置，不必等待网络按序交付。

MRC 因此主动放弃 PFC，运行在 best-effort Ethernet 上。代价是丢包恢复必须足够快。论文里 MRC 用 Selective ACK 精确反馈已经到达的 packet；如果拥塞时交换机支持 packet trimming，包的 payload 可以被裁掉，只把 header 优先转发到接收端，由接收端触发 NACK。这让 sender 能区分"目的端 incast 拥塞"和"路径可能坏了"两类丢包。

路径选择也会持续更新。每个 EV 对应的路径都有少量健康状态：ECN 表示某条路径比其他路径更拥塞，sender 会暂时绕开它；如果非 trimmed packet 直接丢失，MRC 会先假设这条路径可能坏了，立刻停止使用对应 EV，再通过 background probe 判断路径是否恢复。论文称，这类路径绕行可以在几十微秒量级完成。

## 四、SRv6：静态源路由让 MRC 的路径状态可解释

多路径传输还可以基于 ECMP hash 做，但 OpenAI 最终选择 SRv6 source routing，原因很现实：在高 radix 两层 Clos 中，动态路由需要维护大量 destination-specific ECMP set；一旦许多 T1 都有少量 downlink 故障，控制面和转发表都会变得复杂。更麻烦的是，MRC 已经会根据拥塞和丢包绕开路径，BGP 等动态路由再收敛一次，会改变 ECMP 映射，破坏端侧正在建立的路径健康模型。

![SRv6 uSID 转发与 EV 到 SRv6 地址的映射](/assets/openai-mrc-srv6-supercomputer-networking/fig-2-srv6-usid-ev.png)

*图 2：MRC 把 EV 中的 path choice 压缩进 SRv6 uSID 序列；交换机只执行 uSID shift 和静态查表，路径健康由发送端依据 EV 状态维护。来源：论文 Figure 2 与 Figure 3。*

SRv6 的价值在这里是让交换机变得更笨、更稳定。MRC NIC 在 QP startup 时生成 EV set，其中的 bit 直接编码每一跳的路径选择；SRv6 地址使用 uSID 格式，由 locator prefix 加一串 16-bit uSID 组成，每个 uSID 对应路径上的一台交换机。包到达交换机时，交换机检查当前 uSID，左移地址，把下一跳 uSID 暴露出来，再按静态路由表转发。

这个过程把 EV、物理路径和遥测联系起来。某个 EV 变差，运维系统能把它映射回具体 plane、T0 uplink 或交换机路径；MRC 可以先绕行，Clustermapper 再决定是否写入 denylist。论文里有一句很能代表这个设计哲学：保持所有 plane 负载均匀是一条很有用的 invariant；如果某个 plane 看起来比其他 plane 差，通常就指向一个网络问题。

## 五、生产结果：故障从作业中断变成短暂降速

论文最有分量的部分来自生产训练结果。作者说 MRC 已用于 OpenAI 最近的 frontier model 训练，并部署在 OpenAI 和 Microsoft 的最大训练集群中；OpenAI 官方文章进一步说明，MRC 已用于 OpenAI 最大的 NVIDIA GB200 supercomputer，包括 OCI Abilene 的 Stargate 站点和 Microsoft Fairwater supercomputer<a href="https://openai.com/index/mrc-supercomputer-networking/">[2]</a>。

![50K GPU 生产预训练作业中的 NIC-T0 光模块 flap 与丢包恢复](/assets/openai-mrc-srv6-supercomputer-networking/fig-3-production-link-flap.png)

*图 3：一次 50K GPU 生产预训练作业中，T0 交换机上的光模块连续 flap 了 4 条链路；吞吐在约一分钟内下降约 25%，随后恢复，作业没有崩溃，也没有 QP 失败。来源：论文 Figure 6 与 Figure 7。*

在 Cluster A 的 50K GPU 生产作业中，一个 T0 switch 光模块 glitch 导致 4 条 NIC-T0 链路连续 flap，其中 3 个相关 node 正参与训练。由于同步训练由最慢节点决定，整体吞吐在 flap 期间约下降 25%，但随后立即恢复；论文强调，作业没有 crash，没有 QP failed，也不需要把受影响节点移出 job。

另一个 75K GPU 作业中，T1 switch 出现需要 reboot 的故障。论文描述，约四分之一 QP 受影响，约 580K packet 被 drop；吞吐在初始失败时有一次 dip，但 QP 很快 map out bad path，switch 真正 reboot 时反而没有进一步影响。这说明 MRC 的工程收益不只是"更快恢复"，还改变了运维动作的风险模型：很多 link repair 或 switch reboot 不再需要和训练团队精确协调窗口。

微基准也支持这个判断。Cluster B 上，MRC 在 32 KB GPU-to-GPU write bandwidth 测试中，T0-local 与 cross-T1 都达到约 770 Gb/s，约为理论峰值的 96%；2-byte latency 则分别是 5.09 us 和 6.54 us，跨 T1 的额外 switch hop 主要影响短消息延迟。

![64-way collective 中 MRC 与 RoCE 在丢包下的吞吐对比](/assets/openai-mrc-srv6-supercomputer-networking/fig-4-collective-loss-benchmark.png)

*图 4：在 64-way all-reduce 与 all-to-all 中，一个 MRC QP 喷洒到 256 条路径，通常优于 RoCE 通过 16 QP 缓解 ECMP 碰撞；0.1% 丢包下 MRC 对大消息仍较稳。来源：论文 Figure 16 与 Figure 17。*

论文在 Cluster C 做了 collective 测试。64-way ring all-reduce 中，RoCE 单 QP 会受 ECMP hash collision 影响，通常只能达到约一半可用吞吐；增加到 16 QP 有帮助，但收益在 8 QP 后趋缓。MRC 单 QP 跨 256 条路径喷洒，表现超过 16 QP RoCE。加入 0.1% 和 1% 随机丢包后，RoCE 退化明显；MRC 在大消息下可以足够快地 retransmit，0.1% loss 影响较小。1% loss 下 MRC 也只能拿到约三分之一目标吞吐，这足够撑过短暂 burst，不适合长期训练。

这里的边界要讲清楚。MRC 面向真实集群里的短暂 link flap、局部拥塞和灰故障，目标是把这些事件变成端侧可绕开的短时扰动；持续高丢包网络已经超出它的舒适区。论文也承认，如果 NIC transceiver 自己 flap，800G NIC 的所有 port 都会一起丢失，QP 无法继续；这是当前光模块设计下仍然存在的 single point of failure。

## 六、这篇论文真正改变的是控制面的职责划分

MRC 与 SRv6 的组合，可以理解为一次职责重排。过去，交换机动态路由负责 reachability 和 failure recovery，RoCE 负责可靠传输，PFC/DCQCN 负责拥塞控制。MRC 之后，交换机控制面大幅静态化，SRv6 让每个 packet 携带路径意图；端侧 NIC 持有 path-level 状态，根据 ECN、loss、probe 结果选择工作路径。网络的复杂性没有消失，但从分布式交换机控制面转移到更贴近 workload 的端侧协议里。

这对 AI Infra 有两个启发。

第一，百 K GPU 训练网络已经脱离"更大的数据中心网络"这个简单类比。它有非常特殊的目标函数：同步训练关心 tail，flow mean 排在后面；一个节点慢下来，会拖住所有并行维度上的伙伴。因此协议的优化目标要落到 collective 完成时间分布上，而不能只看单流公平性。

第二，硬件与拓扑必须一起设计。MRC 需要 400/800 Gb/s RDMA NIC 支持乱序 placement、SACK/NACK、EV、probe 等机制，也需要 switch 侧支持 SRv6 uSID line-rate forwarding 和 packet trimming。论文提到的实现覆盖 NVIDIA ConnectX-8、AMD Pollara/Vulcano、Broadcom Thor Ultra，以及 NVIDIA Spectrum-4/5、Broadcom Tomahawk 5 等交换芯片或系统。这类方案很难靠软件 overlay 补出来。

## 七、结论

MRC 论文的价值不在某一个单点技巧，而在它把百 K GPU 训练网络的工程目标讲透了：当失败成为常态，网络协议必须让训练作业继续向前走。多平面拓扑降低单点故障的容量份额，MRC packet spraying 把一个 QP 变成多路径传输，SRv6 静态源路由让端侧路径状态和物理故障能够对应起来。

它的适用边界也很清晰。MRC 面向大规模 AI 后端 RDMA 网络，当前语义收窄到训练通信最需要的 write 路径；它依赖新 NIC、新交换机、新拓扑和跨厂商协作。通用云网络或传统应用不能把它当作即插即用的替代品。但对 frontier model pretraining 来说，这篇论文给出了一条明确路线：当规模大到动态路由和 lossless fabric 都难以解释时，让端侧协议承担更多自愈职责，反而能让整张网络更稳定。

---

## 参考资料

[1] Resilient AI Supercomputer Networking using MRC and SRv6. https://cdn.openai.com/pdf/resilient-ai-supercomputer-networking-using-mrc-and-srv6.pdf

[2] Supercomputer networking to accelerate large scale AI training. https://openai.com/index/mrc-supercomputer-networking/

[3] Multipath Reliable Connection (MRC) Specification, Revision 1.0. https://www.opencompute.org/documents/ocp-mrc-1-0-pdf
