?	?L?*??@?L?*??@!?L?*??@	?K??J@?K??J@!?K??J@"h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'?L?*??@1l?,	P???I?????@Yǂ L???r0*	?G?z6w@2J
Iterator::Root::Map???C?X??!X}>L?1Q@)|E?^???1 ?1?)K@:Preprocessing2Y
"Iterator::Root::Map::ParallelMapV2???A_z??!???e?,@)???A_z??1???e?,@:Preprocessing2z
CIterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate((E+???!????h,@)?>V????1t???p$@:Preprocessing2p
9Iterator::Root::Map::ParallelMapV2::Zip[1]::ForeverRepeatr75?|??!?n???'@)????KU??1^??+H#@:Preprocessing2?
SIterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice]???lȏ?!??????@)]???lȏ?1??????@:Preprocessing2|
EIterator::Root::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor2????!Hya?7z@)2????1Hya?7z@:Preprocessing2^
'Iterator::Root::Map::ParallelMapV2::Zip?{h+??!?T?RK?=@)?D?<?|?1?g????:Preprocessing2j
3Iterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap/?.?H??!??=???/@)6?Ko.z?1=$>)Y???:Preprocessing2E
Iterator::Roote??Q???!??D+??Q@)?I?Ux?1?[??????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 3.7% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?80.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?K??J@I?x??RT@Q?z?ͳ?/@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
      ??!             ??!       "	l?,	P???l?,	P???!l?,	P???*      ??!       2      ??!       :	?????@?????@!?????@B      ??!       J	ǂ L???ǂ L???!ǂ L???R      ??!       Z	ǂ L???ǂ L???!ǂ L???b      ??!       JGPUY?K??J@b q?x??RT@y?z?ͳ?/@?"[
+SGD/SGD/update_2/ResourceApplyKerasMomentumResourceApplyKerasMomentum?ч?r??!?ч?r??"[
+SGD/SGD/update_4/ResourceApplyKerasMomentumResourceApplyKerasMomentum?ч?r??!?ч?r??"[
+SGD/SGD/update_6/ResourceApplyKerasMomentumResourceApplyKerasMomentum?L?v????!???,\??"7
sequential/dense_2/MatMulMatMul?h?᫄?!k}	?????0"7
sequential/dense_1/MatMulMatMulT??????!?t?E???0"[
+SGD/SGD/update_1/ResourceApplyKerasMomentumResourceApplyKerasMomentumU??Ƞ^??!ճ?wL(??"[
+SGD/SGD/update_5/ResourceApplyKerasMomentumResourceApplyKerasMomentumU??Ƞ^??!u?U???"[
+SGD/SGD/update_3/ResourceApplyKerasMomentumResourceApplyKerasMomentumB??F^??!=,?????"[
+SGD/SGD/update_7/ResourceApplyKerasMomentumResourceApplyKerasMomentum???_??!?jM?-??"Y
)SGD/SGD/update/ResourceApplyKerasMomentumResourceApplyKerasMomentum'??ΰ??!~?I? ???Q      Y@Y??}?	@@a"5?x+?P@q??B??@y?K??X??"?

device?Your program is NOT input-bound because only 3.7% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?80.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Turing)(: B 