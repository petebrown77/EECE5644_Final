?	?Z(??z@?Z(??z@!?Z(??z@	?
ֆ??@?
ֆ??@!?
ֆ??@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?Z(??z@?n?|?b??1p??;??I?3??@Y?.?????r0*	??Q?J@2J
Iterator::Root::Map>U?W??!?0m7?,Q@)?]~p??134?H??I@:Preprocessing2Y
"Iterator::Root::Map::ParallelMapV2???/fK??!z[*L?d1@)???/fK??1z[*L?d1@:Preprocessing2z
CIterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateF?T?=ϳ?!=H?5?.@)Q?O?Iҭ?1݋zND'@:Preprocessing2p
9Iterator::Root::Map::ParallelMapV2::Zip[1]::ForeverRepeat???F???!??v1?j#@)???ZӼ??1?I??y?@:Preprocessing2?
SIterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicet|?8c???!??ٝ?@)t|?8c???1??ٝ?@:Preprocessing2^
'Iterator::Root::Map::ParallelMapV2::Zipt?5=((??!C?S?}?=@)?'??Q??1??)?^@:Preprocessing2|
EIterator::Root::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?XİØ??!?+a?? @)?XİØ??1?+a?? @:Preprocessing2j
3Iterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap?K??T??!???l1@)?????*??1|??w??:Preprocessing2E
Iterator::RootO?j?v??!/+???Q@)DkE??|?1??v????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 4.7% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?77.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?
ֆ??@IQ?hS@Q?H???1@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?n?|?b???n?|?b??!?n?|?b??      ??!       "	p??;??p??;??!p??;??*      ??!       2      ??!       :	?3??@?3??@!?3??@B      ??!       J	?.??????.?????!?.?????R      ??!       Z	?.??????.?????!?.?????b      ??!       JGPUY?
ֆ??@b qQ?hS@y?H???1@?"[
+SGD/SGD/update_4/ResourceApplyKerasMomentumResourceApplyKerasMomentum??{?n???!??{?n???"[
+SGD/SGD/update_9/ResourceApplyKerasMomentumResourceApplyKerasMomentum??ۘAu??!???GX???"L
.gradient_tape/sequential/dense_3/MatMul/MatMulMatMul?g?0P??!8?ߵp???0"L
.gradient_tape/sequential/dense_2/MatMul/MatMulMatMul???<>??!??7?Gl??0"L
0gradient_tape/sequential/dense_3/MatMul/MatMul_1MatMul???<>??!`??????"L
.gradient_tape/sequential/dense_1/MatMul/MatMulMatMulJ?-~?=??!?G?@???0"[
+SGD/SGD/update_6/ResourceApplyKerasMomentumResourceApplyKerasMomentum???b?+??!?Y????"[
+SGD/SGD/update_2/ResourceApplyKerasMomentumResourceApplyKerasMomentumߥ?,??!L?wz<N??"7
sequential/dense_1/MatMulMatMul?չ????! 	?rO???0"L
0gradient_tape/sequential/dense_2/MatMul/MatMul_1MatMul
?q????!?<}?#Ҷ?Q      Y@Ys?3R1?@a;#s?3Q@q?,???V@y6y5M??"?
device?Your program is NOT input-bound because only 4.7% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?77.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?90.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 