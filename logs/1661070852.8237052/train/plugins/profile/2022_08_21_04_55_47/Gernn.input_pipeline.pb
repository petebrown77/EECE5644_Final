	?$?z?@?$?z?@!?$?z?@      ??!       "_
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails?$?z?@1f?"????IZ???@r0*	٣p=
?v@2J
Iterator::Root::Map	]????!??w?O@)?U?p??1?騭K?F@:Preprocessing2Y
"Iterator::Root::Map::ParallelMapV2? |??!??ј?1@)? |??1??ј?1@:Preprocessing2p
9Iterator::Root::Map::ParallelMapV2::Zip[1]::ForeverRepeat`?o`r???!n???]*@)U/??dƣ?1?"?E?(%@:Preprocessing2z
CIterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateX9??v???!??h)@)7?DeÚ??1??sM?w@:Preprocessing2?
SIterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceyt#,*???!RZ?yX@)yt#,*???1RZ?yX@:Preprocessing2^
'Iterator::Root::Map::ParallelMapV2::Zip??o
+??!???mn5A@);%?Α?1???/?@:Preprocessing2j
3Iterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap-]?6?ɮ?![{???x0@)T?4??-??1???h?&@:Preprocessing2|
EIterator::Root::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor,??26t??!&????@),??26t??1&????@:Preprocessing2E
Iterator::RootF@?#H???!&%?HeP@)??h?~?1?F(??? @:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?86.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI??C??U@Q?????+@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
      ??!             ??!       "	f?"????f?"????!f?"????*      ??!       2      ??!       :	Z???@Z???@!Z???@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q??C??U@y?????+@