	¤????@¤????@!¤????@	?S?>S@?S?>S@!?S?>S@"h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'¤????@1x&4I,)??I?`7l[?@Y???mnL??r0*	??Q?x}@2J
Iterator::Root::Map?W:?%??!`2:?XR@)???e???1~G?R?8L@:Preprocessing2Y
"Iterator::Root::Map::ParallelMapV2'?%??s??!???Cj?0@)'?%??s??1???Cj?0@:Preprocessing2p
9Iterator::Root::Map::ParallelMapV2::Zip[1]::ForeverRepeat?l??}??!;?? ? @)>?ɋL???1????nM@:Preprocessing2z
CIterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate հ????!?T!@)N???????1^'???F@:Preprocessing2^
'Iterator::Root::Map::ParallelMapV2::Zip???Y.??!p?N?8@)3?FY????1??m?a@:Preprocessing2?
SIterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?%8?????!???fJ?@)?%8?????1???fJ?@:Preprocessing2E
Iterator::Root? ?}????!d?|,??R@)???+,??1x WI>@:Preprocessing2|
EIterator::Root::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor???s??!k6?Wd???)???s??1k6?Wd???:Preprocessing2j
3Iterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap?o*Ral??!Fe??%@)h??5??1C?+6???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 2.5% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?78.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?S?>S@IAj??S@Q???ge?2@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
      ??!             ??!       "	x&4I,)??x&4I,)??!x&4I,)??*      ??!       2      ??!       :	?`7l[?@?`7l[?@!?`7l[?@B      ??!       J	???mnL?????mnL??!???mnL??R      ??!       Z	???mnL?????mnL??!???mnL??b      ??!       JGPUY?S?>S@b qAj??S@y???ge?2@