	??#?g@??#?g@!??#?g@	?&/au@?&/au@!?&/au@"h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'??#?g@1$~?.r??I??{*?@YZ???f???r0*	?????{y@2J
Iterator::Root::Map???ο]??!??Qǎ?Q@)?f?v???1??+N?K@:Preprocessing2Y
"Iterator::Root::Map::ParallelMapV2L???????!?5 ?=S-@)L???????1?5 ?=S-@:Preprocessing2p
9Iterator::Root::Map::ParallelMapV2::Zip[1]::ForeverRepeatT????#??!>???? *@)????7???1H?C??%@:Preprocessing2z
CIterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??l??<??!b?*9n"@)??e??t??14"??y@:Preprocessing2?
SIterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceٳ?25	??!D????@)ٳ?25	??1D????@:Preprocessing2E
Iterator::Root?,?Yf??!uz?DR@)??4c?t??1??Y??@:Preprocessing2^
'Iterator::Root::Map::ParallelMapV2::Zip?E~???!*??N?:@)????O??1?>??`@:Preprocessing2|
EIterator::Root::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??????!?/???? @)??????1?/???? @:Preprocessing2j
3Iterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap? ?=~??!h?gX ?&@)ȗP????13ݤ?O @:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 4.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?80.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?&/au@I@x5?$T@Q??????.@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
      ??!             ??!       "	$~?.r??$~?.r??!$~?.r??*      ??!       2      ??!       :	??{*?@??{*?@!??{*?@B      ??!       J	Z???f???Z???f???!Z???f???R      ??!       Z	Z???f???Z???f???!Z???f???b      ??!       JGPUY?&/au@b q@x5?$T@y??????.@