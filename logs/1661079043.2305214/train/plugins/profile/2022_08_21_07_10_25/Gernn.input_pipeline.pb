	ge????@ge????@!ge????@	??Bl?????Bl???!??Bl???"h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'ge????@1V???n/??I)?Ǻ?@Y??f????r0*	9??v?ms@2J
Iterator::Root::Mapׄ?Ơ??!?? ?JbP@)??z?2Q??1????pG@:Preprocessing2Y
"Iterator::Root::Map::ParallelMapV2[??8?	??!????I?3@)[??8?	??1????I?3@:Preprocessing2z
CIterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenatenYk(???!?8??~?-@)?|[?T??1}X?uX8$@:Preprocessing2p
9Iterator::Root::Map::ParallelMapV2::Zip[1]::ForeverRepeat?&??鳣?!%Y??,?(@)??6???1?ݓ???#@:Preprocessing2?
SIterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?]??Nw??!????L$@)?]??Nw??1????L$@:Preprocessing2^
'Iterator::Root::Map::ParallelMapV2::Zip/j?? ߹?!??c:A@@)|???s??17ޅ?wq@:Preprocessing2|
EIterator::Root::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensoru?~?!T?5?"j@)u?~?1T?5?"j@:Preprocessing2j
3Iterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap?q5?+-??!??Æ/1@)??6?{?1??ﳀo@:Preprocessing2E
Iterator::RootгY?????!<?>?b?P@)"ߥ?%?x?1??r??E??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 1.6% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?83.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9??Bl???I???e7?T@Q<?-IW.@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
      ??!             ??!       "	V???n/??V???n/??!V???n/??*      ??!       2      ??!       :	)?Ǻ?@)?Ǻ?@!)?Ǻ?@B      ??!       J	??f??????f????!??f????R      ??!       Z	??f??????f????!??f????b      ??!       JGPUY??Bl???b q???e7?T@y<?-IW.@