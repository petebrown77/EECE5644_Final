	??|???@??|???@!??|???@	Lu???@Lu???@!Lu???@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??|???@???4??Z?1??r????I}(?@Y^f?(?7??r0*	?p=
?cz@2J
Iterator::Root::Map5ӽN????!?Q/uPR@)<l"3???1"\EkL@:Preprocessing2Y
"Iterator::Root::Map::ParallelMapV2\t??z???!eJk0@)\t??z???1eJk0@:Preprocessing2z
CIterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?4*p???!߅????)@)f?y??̣?1?أiQ"@:Preprocessing2p
9Iterator::Root::Map::ParallelMapV2::Zip[1]::ForeverRepeat??%??:??!~?d? @)???????1,?4?r?@:Preprocessing2?
SIterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceB|`????!U?1?;?@)B|`????1U?1?;?@:Preprocessing2^
'Iterator::Root::Map::ParallelMapV2::Zip/???uR??!?3?
?F9@)4??k???1??&??~@:Preprocessing2|
EIterator::Root::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorj???}?!? ?Y???)j???}?1? ?Y???:Preprocessing2E
Iterator::RootM?p]1??!sB?O?R@)?W?\y?1GK???v??:Preprocessing2j
3Iterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap@?5_%??!^;??^?,@)?E|'f?x?1???g0???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 6.9% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"?77.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9Ku???@I??(;LS@Q?@.?g?/@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???4??Z????4??Z?!???4??Z?      ??!       "	??r??????r????!??r????*      ??!       2      ??!       :	}(?@}(?@!}(?@B      ??!       J	^f?(?7??^f?(?7??!^f?(?7??R      ??!       Z	^f?(?7??^f?(?7??!^f?(?7??b      ??!       JGPUYKu???@b q??(;LS@y?@.?g?/@