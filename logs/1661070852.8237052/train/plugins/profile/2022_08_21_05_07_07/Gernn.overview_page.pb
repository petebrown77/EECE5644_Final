?	^?/?;@^?/?;@!^?/?;@      ??!       "_
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails^?/?;@1??4?(??IԛQ??@r0*	l?????@2J
Iterator::Root::Map!??????!??̏t?P@)?y?Տ??1??v??J@:Preprocessing2Y
"Iterator::Root::Map::ParallelMapV2?Q????!NUO?ʯ+@)?Q????1NUO?ʯ+@:Preprocessing2z
CIterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?Z??α?!`v???M+@)???3????1?:???"@:Preprocessing2p
9Iterator::Root::Map::ParallelMapV2::Zip[1]::ForeverRepeat?/o???!??KC?%@)!<?8b??1X)Fցd @:Preprocessing2?
SIterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?"?dT??!?v?[??@)?"?dT??1?v?[??@:Preprocessing2^
'Iterator::Root::Map::ParallelMapV2::Zip&䃞ͪ??!?іhM'>@)??M???1??"?E@:Preprocessing2|
EIterator::Root::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?:TS?u??!????@)?:TS?u??1????@:Preprocessing2E
Iterator::Root7?֊6???!?Kڥ,vQ@)??4c?t??1?R??7@:Preprocessing2j
3Iterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap{?%T??!?jP??*/@)?????*??1ġ?DV???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?82.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI?5?h;?T@Q?(G]+1@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
      ??!             ??!       "	??4?(????4?(??!??4?(??*      ??!       2      ??!       :	ԛQ??@ԛQ??@!ԛQ??@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?5?h;?T@y?(G]+1@?"[
+SGD/SGD/update_2/ResourceApplyKerasMomentumResourceApplyKerasMomentum;G??\w??!;G??\w??"[
+SGD/SGD/update_4/ResourceApplyKerasMomentumResourceApplyKerasMomentum/??Sу?!???]X$??"[
+SGD/SGD/update_6/ResourceApplyKerasMomentumResourceApplyKerasMomentum?! n???!=??m????"7
sequential/dense_2/MatMulMatMula???u??!w?-$?ڣ?0"7
sequential/dense_3/MatMulMatMula???u??!O?e?ҷ??0"7
sequential/dense_1/MatMulMatMul??u??b??!?CK{???0"[
+SGD/SGD/update_3/ResourceApplyKerasMomentumResourceApplyKerasMomentumB?
0P??!????C2??"[
+SGD/SGD/update_8/ResourceApplyKerasMomentumResourceApplyKerasMomentumB?
0P??!RY??I???"[
+SGD/SGD/update_1/ResourceApplyKerasMomentumResourceApplyKerasMomentum?ןI?=??!HT?T??"[
+SGD/SGD/update_5/ResourceApplyKerasMomentumResourceApplyKerasMomentum?ןI?=??!>O???k??Q      Y@Ys?3R1?@a;#s?3Q@q?ܥ?V@yw?I?E??"?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?82.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?88.3% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 