	??????@??????@!??????@	!6??*?@!6??*?@!!6??*?@"h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'??????@1M1AG+??I???#?@Y?V]?jJ??r0*	V-?z@2J
Iterator::Root::Map?	L?u??!??G?rP@)?k?˸??1??,A?.I@:Preprocessing2Y
"Iterator::Root::Map::ParallelMapV2QO?????!mRv9??.@)QO?????1mRv9??.@:Preprocessing2z
CIterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?W˝?`??!8?b?,?-@)W{?l??1ȷ???F%@:Preprocessing2p
9Iterator::Root::Map::ParallelMapV2::Zip[1]::ForeverRepeatގpZ????!?^?x%@)?H? O??1ȳ ??? @:Preprocessing2?
SIterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceZg|_\???!?xRG??@)Zg|_\???1?xRG??@:Preprocessing2E
Iterator::Root?1 {????!_/78?>Q@)\?z???1$TC?w	@:Preprocessing2^
'Iterator::Root::Map::ParallelMapV2::ZipUg????!?B#??@)uWv?????1}l???	@:Preprocessing2|
EIterator::Root::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?u??O??!?r??\@)?u??O??1?r??\@:Preprocessing2j
3Iterator::Root::Map::ParallelMapV2::Zip[0]::FlatMapM?<i???!??U?&1@)[&??|??1??k?0@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 7.7% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"?74.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9"6??*?@I?+????R@Q????V?1@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
      ??!             ??!       "	M1AG+??M1AG+??!M1AG+??*      ??!       2      ??!       :	???#?@???#?@!???#?@B      ??!       J	?V]?jJ???V]?jJ??!?V]?jJ??R      ??!       Z	?V]?jJ???V]?jJ??!?V]?jJ??b      ??!       JGPUY"6??*?@b q?+????R@y????V?1@