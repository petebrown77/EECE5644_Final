	H2?w?M@H2?w?M@!H2?w?M@	?­{]r!@?­{]r!@!?­{]r!@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0H2?w?M@ލ?A???1IIC+??I! _B?@Y+1?JZ???r0*	?&1???@2J
Iterator::Root::MapR??8ӄ??!fu?5?P@)o??????1)u?Y??H@:Preprocessing2Y
"Iterator::Root::Map::ParallelMapV2??h>???!H?v???0@)??h>???1H?v???0@:Preprocessing2p
9Iterator::Root::Map::ParallelMapV2::Zip[1]::ForeverRepeat???qn??!???"?*@)???+ҫ?1^?????$@:Preprocessing2z
CIterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?0
?Ƿ??!;ƽ?m'@)??????1J31>s @:Preprocessing2^
'Iterator::Root::Map::ParallelMapV2::Zipu?? ???!??$?@.?@)?m??ʆ??1?ݬ? ?@:Preprocessing2?
SIterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice????9??!?7S??@)????9??1?7S??@:Preprocessing2|
EIterator::Root::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor-?\o????!?U$&??@)-?\o????1?U$&??@:Preprocessing2E
Iterator::Root	3m??J??!?ö?o4Q@)?V??????1'?i YG@:Preprocessing2j
3Iterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap??-]???!J?GI?+@)?B?_?+??1;T.?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 8.7% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"?71.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?­{]r!@I???)?R@Q???h?2@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	ލ?A???ލ?A???!ލ?A???      ??!       "	IIC+??IIC+??!IIC+??*      ??!       2      ??!       :	! _B?@! _B?@!! _B?@B      ??!       J	+1?JZ???+1?JZ???!+1?JZ???R      ??!       Z	+1?JZ???+1?JZ???!+1?JZ???b      ??!       JGPUY?­{]r!@b q???)?R@y???h?2@