	)%??@)%??@!)%??@      ??!       "_
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails)%??@1??s?//??I3???p@r0*	? ?rh?{@2J
Iterator::Root::MapZ?A??v??!G6?N?tN@)6?>W[???1c8?CygF@:Preprocessing2Y
"Iterator::Root::Map::ParallelMapV2???)x??!????0@)???)x??1????0@:Preprocessing2z
CIterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate뫫???!?O!?2?1@)\?z???1???nXr(@:Preprocessing2p
9Iterator::Root::Map::ParallelMapV2::Zip[1]::ForeverRepeatX?\T??!?o??X?(@)????+??11??p5U#@:Preprocessing2?
SIterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice????P??!?х?4@)????P??1?х?4@:Preprocessing2^
'Iterator::Root::Map::ParallelMapV2::Zip_Ӄ?R???!"?`?B@)Q?+?ϒ?1$?j,?f@:Preprocessing2E
Iterator::Root7qr?CQ??!??m??O@)?;FzQ??1?y?U?@:Preprocessing2|
EIterator::Root::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?z?V????!&?qS??@)?z?V????1&?qS??@:Preprocessing2j
3Iterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap=E7???!.?;???3@)??st??1?z????@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?84.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIP|Z??6U@Q?,??J.@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
      ??!             ??!       "	??s?//????s?//??!??s?//??*      ??!       2      ??!       :	3???p@3???p@!3???p@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qP|Z??6U@y?,??J.@