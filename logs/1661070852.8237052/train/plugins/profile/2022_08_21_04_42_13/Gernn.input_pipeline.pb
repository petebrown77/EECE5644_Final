	CV?zN?@CV?zN?@!CV?zN?@	\L??????\L??????!\L??????"h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'CV?zN?@1@j'w??Io?ꐛ?@Y???ׁs??r0*	B`??"wy@2J
Iterator::Root::MapOw?x???!y??٠WQ@)DOʤ?6??1??e?YK@:Preprocessing2Y
"Iterator::Root::Map::ParallelMapV2g}?1Yܯ?!??o*??.@)g}?1Yܯ?1??o*??.@:Preprocessing2p
9Iterator::Root::Map::ParallelMapV2::Zip[1]::ForeverRepeat~;??"??!ɴ??#'@)???	F??1?C?D`z"@:Preprocessing2z
CIterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate~;??"??!ɴ??#'@)W	?3???1w??|??@:Preprocessing2?
SIterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?m??ʆ??!???H?@)?m??ʆ??1???H?@:Preprocessing2^
'Iterator::Root::Map::ParallelMapV2::Zip?h?^`??!Z?G?T=@)?A?p?-??1?M̛??@:Preprocessing2j
3Iterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap>"?D??!??Յ?+@)l
dv???1?F<?y?@:Preprocessing2|
EIterator::Root::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor|???s??!]??#?@)|???s??1]??#?@:Preprocessing2E
Iterator::Root;?ީ?{??!)??*?Q@)?P?,y?1*,E?}"??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 1.9% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?84.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9[L??????Ir???.U@Q??<???*@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
      ??!             ??!       "	@j'w??@j'w??!@j'w??*      ??!       2      ??!       :	o?ꐛ?@o?ꐛ?@!o?ꐛ?@B      ??!       J	???ׁs?????ׁs??!???ׁs??R      ??!       Z	???ׁs?????ׁs??!???ׁs??b      ??!       JGPUY[L??????b qr???.U@y??<???*@