	m?_u?H@m?_u?H@!m?_u?H@	??[??????[????!??[????"h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'm?_u?H@1?3??????IO"¿j@Yĵ??^(??r0*?O??n?y@)      0=2J
Iterator::Root::Map?t????!?????;Q@)[Υ?????17x??G?H@:Preprocessing2Y
"Iterator::Root::Map::ParallelMapV2?6??nf??!vo???v3@)?6??nf??1vo???v3@:Preprocessing2z
CIterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenateܡa1?Z??!?U4ݺ*@)?]K?=??1????%["@:Preprocessing2p
9Iterator::Root::Map::ParallelMapV2::Zip[1]::ForeverRepeat%̴?++??!?bݟ2$@) F?6???1?٨?@:Preprocessing2?
SIterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?,??;??!3?`U?@)?,??;??13?`U?@:Preprocessing2^
'Iterator::Root::Map::ParallelMapV2::Zip??`?.???!9?g?<@)p_?Q??1??pz	@:Preprocessing2E
Iterator::RootA??ǘ???!1`?<??Q@)X??"?t??1??y@:Preprocessing2j
3Iterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap?d?pu ??!??n_'?.@)^??????1?l???@:Preprocessing2|
EIterator::Root::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor???s??!?Wl?-?@)???s??1?Wl?-?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 2.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?84.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9??[????I?_o?L,U@Q3?'?*@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
      ??!             ??!       "	?3???????3??????!?3??????*      ??!       2      ??!       :	O"¿j@O"¿j@!O"¿j@B      ??!       J	ĵ??^(??ĵ??^(??!ĵ??^(??R      ??!       Z	ĵ??^(??ĵ??^(??!ĵ??^(??b      ??!       JGPUY??[????b q?_o?L,U@y3?'?*@