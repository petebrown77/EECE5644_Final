	???d?@???d?@!???d?@	6@???[%@6@???[%@!6@???[%@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0???d?@lw?}9??1z?W???I^M???n@Y????A???r0*	gffff#?@2J
Iterator::Root::Map?o*Ral??!g,Q???S@)=~oӟ??1v?&?Q@:Preprocessing2Y
"Iterator::Root::Map::ParallelMapV2?aod??!?/D"#?&@)?aod??1?/D"#?&@:Preprocessing2z
CIterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?,??2??!?)u??#@)(?bd???1?@?@??@:Preprocessing2p
9Iterator::Root::Map::ParallelMapV2::Zip[1]::ForeverRepeatǠB]??!{??{??@)???[??1?S???r@:Preprocessing2?
SIterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceı.n???! ?S!/@)ı.n???1 ?S!/@:Preprocessing2^
'Iterator::Root::Map::ParallelMapV2::Zipt]???Թ?!?Yʸ?3@)/?o??e??1+V??Q??:Preprocessing2|
EIterator::Root::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??(????!Ȣ<?S??)??(????1Ȣ<?S??:Preprocessing2j
3Iterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap0?[w???!?ꯝ?%@)???(?v?13?2D?*??:Preprocessing2E
Iterator::Root?v? ݗ??!?i͑{T@)?f׽e?1???r??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 10.7% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"?74.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no96@???[%@I????R@Q__FK-.-@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	lw?}9??lw?}9??!lw?}9??      ??!       "	z?W???z?W???!z?W???*      ??!       2      ??!       :	^M???n@^M???n@!^M???n@B      ??!       J	????A???????A???!????A???R      ??!       Z	????A???????A???!????A???b      ??!       JGPUY6@???[%@b q????R@y__FK-.-@