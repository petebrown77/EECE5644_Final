	2?????@2?????@!2?????@	wG??H0 @wG??H0 @!wG??H0 @"h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'2?????@1??`?.`??IR???T?@Y/?>:u???r0*	??C?l?{@2J
Iterator::Root::MapY2???z??!j??zQ@)?Z??	??1?????{I@:Preprocessing2Y
"Iterator::Root::Map::ParallelMapV2/??س?!q.7??j1@)/??س?1q.7??j1@:Preprocessing2p
9Iterator::Root::Map::ParallelMapV2::Zip[1]::ForeverRepeat?l??爬?!x??_)@)?K?A????1?称$@:Preprocessing2z
CIterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatec??Ց??!c?V?02(@)?A?L????1?????? @:Preprocessing2?
SIterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?fh<??!?FX?@@)?fh<??1?FX?@@:Preprocessing2^
'Iterator::Root::Map::ParallelMapV2::Zip??L?????!?o?1?=@)F%u???1@?e??@:Preprocessing2|
EIterator::Root::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??4c?t??!???Z?@)??4c?t??1???Z?@:Preprocessing2j
3Iterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap?????N??!?O$E??,@)?????*??1?\57??@:Preprocessing2E
Iterator::Rootmq??d???!?9???Q@)??ǘ????1??vM&??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 2.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?84.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9wG??H0 @IS??BU@QGFV??+@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
      ??!             ??!       "	??`?.`????`?.`??!??`?.`??*      ??!       2      ??!       :	R???T?@R???T?@!R???T?@B      ??!       J	/?>:u???/?>:u???!/?>:u???R      ??!       Z	/?>:u???/?>:u???!/?>:u???b      ??!       JGPUYwG??H0 @b qS??BU@yGFV??+@