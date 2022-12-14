?	??|???@??|???@!??|???@	Lu???@Lu???@!Lu???@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??|???@???4??Z?1??r????I}(?@Y^f?(?7??r0*	?p=
?cz@2J
Iterator::Root::Map5ӽN????!?Q/uPR@)<l"3???1"\EkL@:Preprocessing2Y
"Iterator::Root::Map::ParallelMapV2\t??z???!eJk0@)\t??z???1eJk0@:Preprocessing2z
CIterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?4*p???!߅????)@)f?y??̣?1?أiQ"@:Preprocessing2p
9Iterator::Root::Map::ParallelMapV2::Zip[1]::ForeverRepeat??%??:??!~?d? @)???????1,?4?r?@:Preprocessing2?
SIterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceB|`????!U?1?;?@)B|`????1U?1?;?@:Preprocessing2^
'Iterator::Root::Map::ParallelMapV2::Zip/???uR??!?3?
?F9@)4??k???1??&??~@:Preprocessing2|
EIterator::Root::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorj???}?!? ?Y???)j???}?1? ?Y???:Preprocessing2E
Iterator::RootM?p]1??!sB?O?R@)?W?\y?1GK???v??:Preprocessing2j
3Iterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap@?5_%??!^;??^?,@)?E|'f?x?1???g0???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 6.9% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"?77.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9Ku???@I??(;LS@Q?@.?g?/@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???4??Z????4??Z?!???4??Z?      ??!       "	??r??????r????!??r????*      ??!       2      ??!       :	}(?@}(?@!}(?@B      ??!       J	^f?(?7??^f?(?7??!^f?(?7??R      ??!       Z	^f?(?7??^f?(?7??!^f?(?7??b      ??!       JGPUYKu???@b q??(;LS@y?@.?g?/@?"[
+SGD/SGD/update_4/ResourceApplyKerasMomentumResourceApplyKerasMomentum??h????!??h????"[
+SGD/SGD/update_2/ResourceApplyKerasMomentumResourceApplyKerasMomentum?G?????!?}?*I??"7
sequential/dense_2/MatMulMatMul?b{????!h༟?0"7
sequential/dense_1/MatMulMatMul}1
?\???!??t???0"[
+SGD/SGD/update_5/ResourceApplyKerasMomentumResourceApplyKerasMomentumZ?0???![)Ex?0??"[
+SGD/SGD/update_1/ResourceApplyKerasMomentumResourceApplyKerasMomentum??c??s??!??h?M??"[
+SGD/SGD/update_6/ResourceApplyKerasMomentumResourceApplyKerasMomentum??c??s??!??{?<5??"[
+SGD/SGD/update_3/ResourceApplyKerasMomentumResourceApplyKerasMomentum??qg`??!?Ǻ?I???"Y
)SGD/SGD/update/ResourceApplyKerasMomentumResourceApplyKerasMomentum??q???!?	X?7A??"[
+SGD/SGD/update_7/ResourceApplyKerasMomentumResourceApplyKerasMomentum/_p???!?fؚ???Q      Y@Y??}?	@@a"5?x+?P@q?Z??#@y??9%T??"?

both?Your program is MODERATELY input-bound because 6.9% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?77.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Turing)(: B 