?	??ԱJ	@??ԱJ	@!??ԱJ	@	?|?f?????|?f????!?|?f????"h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'??ԱJ	@1ƿϸp ??I?6?h?M@Y??r????r0*	??? ?4~@2J
Iterator::Root::Map????????!???:?L@)?k?,	P??1??`?0?G@:Preprocessing2z
CIterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateQ??֥??!
??r?$.@)B???D??1?σ?%@:Preprocessing2Y
"Iterator::Root::Map::ParallelMapV2??!????!ڀ?(?$@)??!????1ڀ?(?$@:Preprocessing2p
9Iterator::Root::Map::ParallelMapV2::Zip[1]::ForeverRepeat,?z??m??!!3nN6?'@)?xͫ:???1Th _??!@:Preprocessing2E
Iterator::Root
i?A'???!P?S?P@)	?=b???1<f??? @:Preprocessing2?
SIterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??4c?t??!??݅&@)??4c?t??1??݅&@:Preprocessing2^
'Iterator::Root::Map::ParallelMapV2::Zipv??=???!`?X???@@)2;?ީ??1?????
@:Preprocessing2j
3Iterator::Root::Map::ParallelMapV2::Zip[0]::FlatMapT? ?!Ƕ?!ZǛi2@)??M+???1?R>??
@:Preprocessing2|
EIterator::Root::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor????
??!1+7?	@)????
??11+7?	@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 1.4% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?84.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?|?f????I??BU@Q???]?6,@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
      ??!             ??!       "	ƿϸp ??ƿϸp ??!ƿϸp ??*      ??!       2      ??!       :	?6?h?M@?6?h?M@!?6?h?M@B      ??!       J	??r??????r????!??r????R      ??!       Z	??r??????r????!??r????b      ??!       JGPUY?|?f????b q??BU@y???]?6,@?"Y
)SGD/SGD/update/ResourceApplyKerasMomentumResourceApplyKerasMomentums?X?|???!s?X?|???"[
+SGD/SGD/update_4/ResourceApplyKerasMomentumResourceApplyKerasMomentums?X?|???!s?X?|???"[
+SGD/SGD/update_2/ResourceApplyKerasMomentumResourceApplyKerasMomentum?=?e????!a?0o?j??"[
+SGD/SGD/update_3/ResourceApplyKerasMomentumResourceApplyKerasMomentum??Q????!???K???"[
+SGD/SGD/update_5/ResourceApplyKerasMomentumResourceApplyKerasMomentum??Q????!.??`-???"[
+SGD/SGD/update_1/ResourceApplyKerasMomentumResourceApplyKerasMomentumb?(?????!F#K_}??"[
+SGD/SGD/update_6/ResourceApplyKerasMomentumResourceApplyKerasMomentum?*??蓃?!??5?
???"[
+SGD/SGD/update_7/ResourceApplyKerasMomentumResourceApplyKerasMomentum?*??蓃?!U??؇#??"L
0gradient_tape/sequential/dense_2/MatMul/MatMul_1MatMul?7/?Il??!H?????"L
0gradient_tape/sequential/dense_1/MatMul/MatMul_1MatMul??5??0??!?<?,???Q      Y@Y??}?	@@a"5?x+?P@q?_?v~?U@yo{E??"?
device?Your program is NOT input-bound because only 1.4% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?84.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?87.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 