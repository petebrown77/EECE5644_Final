?	???@g@???@g@!???@g@	??U?
?"@??U?
?"@!??U?
?"@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0???@g@wj.7???1?s?Lh??Iʩ?ajk@Y2??n??r0*	????Ku@2J
Iterator::Root::Map?*??p???!Y?~??O@)DkE????1??')k?E@:Preprocessing2Y
"Iterator::Root::Map::ParallelMapV2yͫ:???!?
1???4@)yͫ:???1?
1???4@:Preprocessing2p
9Iterator::Root::Map::ParallelMapV2::Zip[1]::ForeverRepeat?	??$>??!????*@)7??????1?A?n?~$@:Preprocessing2z
CIterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate???`?>??!?KN>??,@)?lw?N??1}z??(?#@:Preprocessing2?
SIterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceP??????!????03@)P??????1????03@:Preprocessing2^
'Iterator::Root::Map::ParallelMapV2::Zip6??Ľ?!oM?eA@)?6?^???1?,?@:Preprocessing2|
EIterator::Root::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??$?pt??!???e?@)??$?pt??1???e?@:Preprocessing2j
3Iterator::Root::Map::ParallelMapV2::Zip[0]::FlatMapĖM?d??!Hޛ??k1@)?XİØ??1?¥Ð?@:Preprocessing2E
Iterator::Root?^?D???!IY&??wP@)_{fI??z?1?0?qc??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 9.4% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"?68.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*moderate2s6.5 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9??U?
?"@IXf?m6?R@Qf	??A?/@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	wj.7???wj.7???!wj.7???      ??!       "	?s?Lh???s?Lh??!?s?Lh??*      ??!       2      ??!       :	ʩ?ajk@ʩ?ajk@!ʩ?ajk@B      ??!       J	2??n??2??n??!2??n??R      ??!       Z	2??n??2??n??!2??n??b      ??!       JGPUY??U?
?"@b qXf?m6?R@yf	??A?/@?"[
+SGD/SGD/update_2/ResourceApplyKerasMomentumResourceApplyKerasMomentum?w?:⛆?!?w?:⛆?"7
sequential/dense_1/MatMulMatMulv?j?cG??!-?2??q??0"[
+SGD/SGD/update_1/ResourceApplyKerasMomentumResourceApplyKerasMomentum???????!T	??????"[
+SGD/SGD/update_4/ResourceApplyKerasMomentumResourceApplyKerasMomentum???F???!d?>???"[
+SGD/SGD/update_3/ResourceApplyKerasMomentumResourceApplyKerasMomentum?(?'t??!???ae??"L
0gradient_tape/sequential/dense_1/MatMul/MatMul_1MatMulc??jI???!{?(^?N??"Y
)SGD/SGD/update/ResourceApplyKerasMomentumResourceApplyKerasMomentum???
???!аh?Q???"[
+SGD/SGD/update_5/ResourceApplyKerasMomentumResourceApplyKerasMomentum??? ?v??!?(?'t??"L
.gradient_tape/sequential/dense_1/MatMul/MatMulMatMul??,{?`??!???TC ??0"J
,gradient_tape/sequential/dense/MatMul/MatMulMatMul??????!????䁺?0Q      Y@YK??">?@@a???`?P@q	,:H?V@y??:?'??"?
both?Your program is MODERATELY input-bound because 9.4% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?68.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.moderate"s6.5 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?90.2% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 