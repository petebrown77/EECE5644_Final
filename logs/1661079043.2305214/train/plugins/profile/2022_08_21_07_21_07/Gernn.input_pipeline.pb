	?h?'??@?h?'??@!?h?'??@	?z?e?2#@?z?e?2#@!?z?e?2#@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?h?'??@??$?pt??1ŬC9???IM????@YDkE?????r0*	'1?N?@2J
Iterator::Root::Map? -??!??
?5N@)b?aL?{??1q?bn3?G@:Preprocessing2Y
"Iterator::Root::Map::ParallelMapV2?RD?U???!??upZ?*@)?RD?U???1??upZ?*@:Preprocessing2z
CIterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate/?????!?????91@)??j??1v??p?B*@:Preprocessing2p
9Iterator::Root::Map::ParallelMapV2::Zip[1]::ForeverRepeat_F???j??!??,>@?.@)vQ???`??1?,?}D*@:Preprocessing2?
SIterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??T?G???!???
7b@)??T?G???1???
7b@:Preprocessing2^
'Iterator::Root::Map::ParallelMapV2::Zip?N@a???!yy??IC@)r?CQ?O??1[???i@:Preprocessing2|
EIterator::Root::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorD???XP??!???3@)D???XP??1???3@:Preprocessing2j
3Iterator::Root::Map::ParallelMapV2::Zip[0]::FlatMapg~5??!m /}3@)???+,??1??4??@:Preprocessing2E
Iterator::Root\?nK????!???I?N@)??$?ptu?1F??????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 9.6% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"?75.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?z?e?2#@I????R@Qܝ.=?T-@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??$?pt????$?pt??!??$?pt??      ??!       "	ŬC9???ŬC9???!ŬC9???*      ??!       2      ??!       :	M????@M????@!M????@B      ??!       J	DkE?????DkE?????!DkE?????R      ??!       Z	DkE?????DkE?????!DkE?????b      ??!       JGPUY?z?e?2#@b q????R@yܝ.=?T-@