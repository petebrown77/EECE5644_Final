	?/.Ui@?/.Ui@!?/.Ui@	
0Q8???
0Q8???!
0Q8???"h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'?/.Ui@1??????IO???*`@Y??[;Q??r0*	???Mb?y@2J
Iterator::Root::Mapj???v???!?U ?P@))????h??1l[?)J@:Preprocessing2Y
"Iterator::Root::Map::ParallelMapV2??QI????!?re??-@)??QI????1?re??-@:Preprocessing2p
9Iterator::Root::Map::ParallelMapV2::Zip[1]::ForeverRepeat	????Q??!xG?a?)(@)?'c|????13 ???#@:Preprocessing2z
CIterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate|?q7??!bnN?+@)jM??St??1??yI??#@:Preprocessing2?
SIterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice$??ŋ???!j+I(?@)$??ŋ???1j+I(?@:Preprocessing2^
'Iterator::Root::Map::ParallelMapV2::Zip??c?g^??!?? ??>?@)?V??????1ɠ??O?@:Preprocessing2j
3Iterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap?HP???!?nJ?-60@)EJ?y??1??3(@:Preprocessing2|
EIterator::Root::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensoro?EE?N??!??by@)o?EE?N??1??by@:Preprocessing2E
Iterator::Rooth??W??!U???D0Q@)??Û5x?1??|?	??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 1.4% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?85.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9
0Q8???Iܾ?P?HU@Qcxo?*@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
      ??!             ??!       "	????????????!??????*      ??!       2      ??!       :	O???*`@O???*`@!O???*`@B      ??!       J	??[;Q????[;Q??!??[;Q??R      ??!       Z	??[;Q????[;Q??!??[;Q??b      ??!       JGPUY
0Q8???b qܾ?P?HU@ycxo?*@