?	?t???;@?t???;@!?t???;@	Ha??"@Ha??"@!Ha??"@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?t???;@?,??;??1??ْU???I??????@Y*?~?s??r0*	????x?y@2J
Iterator::Root::Map?8ӄ?'??!\Ԑ?RQ@)?Ѭl???1??G?#?K@:Preprocessing2Y
"Iterator::Root::Map::ParallelMapV2???s?v??!??f'?,@)???s?v??1??f'?,@:Preprocessing2z
CIterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?1??%???!?6??*@)b?7?W???1?????"@:Preprocessing2p
9Iterator::Root::Map::ParallelMapV2::Zip[1]::ForeverRepeatOt	???!??ZS?}%@)????W??1F?`? @:Preprocessing2?
SIterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice-?\o????!??P???@)-?\o????1??P???@:Preprocessing2^
'Iterator::Root::Map::ParallelMapV2::Zip?K?^I??!?/kז?<@).??e?O??19?uBY=@:Preprocessing2|
EIterator::Root::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??s?v???!jG
%?@)??s?v???1jG
%?@:Preprocessing2j
3Iterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap?Q???!??
.?.@)X?5?;N??1?4Q_? @:Preprocessing2E
Iterator::Root??y?]???!4%J??Q@)???)?~?1c?em3??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 9.2% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"?75.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9Ha??"@I?L3??R@QQц?.@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?,??;???,??;??!?,??;??      ??!       "	??ْU?????ْU???!??ْU???*      ??!       2      ??!       :	??????@??????@!??????@B      ??!       J	*?~?s??*?~?s??!*?~?s??R      ??!       Z	*?~?s??*?~?s??!*?~?s??b      ??!       JGPUYHa??"@b q?L3??R@yQц?.@?"9
sequential_2/dense_9/MatMulMatMul??3?\)??!??3?\)??0":
sequential_2/dense_10/MatMulMatMulٿ(Tb??!L,.????0"[
+SGD/SGD/update_2/ResourceApplyKerasMomentumResourceApplyKerasMomentums9?e	???!?=dp??"Y
)SGD/SGD/update/ResourceApplyKerasMomentumResourceApplyKerasMomentum??|?????!lg????"[
+SGD/SGD/update_4/ResourceApplyKerasMomentumResourceApplyKerasMomentum??|?????!V@F????"[
+SGD/SGD/update_5/ResourceApplyKerasMomentumResourceApplyKerasMomentum3Kwq0???!#?}??"[
+SGD/SGD/update_1/ResourceApplyKerasMomentumResourceApplyKerasMomentumOrl6}??!ۗ?G1???"[
+SGD/SGD/update_3/ResourceApplyKerasMomentumResourceApplyKerasMomentumOrl6}??!%&????"[
+SGD/SGD/update_6/ResourceApplyKerasMomentumResourceApplyKerasMomentum?gϸi??!??"???"[
+SGD/SGD/update_7/ResourceApplyKerasMomentumResourceApplyKerasMomentum?gϸi??!???<F???Q      Y@Y??}?	@@a"5?x+?P@qO???1?V@yv?P??"?
both?Your program is MODERATELY input-bound because 9.2% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?75.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?91.3% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 