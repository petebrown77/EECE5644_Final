	?z?8@?z?8@!?z?8@	??I?????I???!??I???"h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'?z?8@1L?$zE??I?ai?Gu@YfI??Z???r0*	%??C_}@2J
Iterator::Root::MapC??g???!?q?27O@)xG?j????1?????H@:Preprocessing2Y
"Iterator::Root::Map::ParallelMapV2?)?:??!h١??q,@)?)?:??1h١??q,@:Preprocessing2z
CIterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateD??0??!ݍ??Cq2@)?!?
???1n?X??+@:Preprocessing2p
9Iterator::Root::Map::ParallelMapV2::Zip[1]::ForeverRepeat?j?ѫ?!M?'@)?}??ϥ?1?% ]Z!"@:Preprocessing2?
SIterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice???%P??!G[*??@)???%P??1G[*??@:Preprocessing2^
'Iterator::Root::Map::ParallelMapV2::Zip??r?????!ƞ??B@)?
?<??1?????
@:Preprocessing2j
3Iterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap?_[??g??!?E??5@)?2?FY???10?M??f@:Preprocessing2|
EIterator::Root::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?E?x??!?qӿ?@)?E?x??1?qӿ?@:Preprocessing2E
Iterator::Root??n??;??!:a&?:?O@)jkD0.}?1???A??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 1.6% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?83.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9??I???I]+??T@Q|e???.@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
      ??!             ??!       "	L?$zE??L?$zE??!L?$zE??*      ??!       2      ??!       :	?ai?Gu@?ai?Gu@!?ai?Gu@B      ??!       J	fI??Z???fI??Z???!fI??Z???R      ??!       Z	fI??Z???fI??Z???!fI??Z???b      ??!       JGPUY??I???b q]+??T@y|e???.@