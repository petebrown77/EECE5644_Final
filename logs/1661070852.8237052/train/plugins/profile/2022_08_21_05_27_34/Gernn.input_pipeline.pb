	H?V@H?V@!H?V@	?:\?8C@?:\?8C@!?:\?8C@"h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'H?V@1??_vO???ISX????@Yv?~k'??r0*	?~j?tz@2J
Iterator::Root::Map??V*???!{??&hrQ@)	??????1??.???I@:Preprocessing2Y
"Iterator::Root::Map::ParallelMapV2?{b?*??!l;?\e?1@)?{b?*??1l;?\e?1@:Preprocessing2p
9Iterator::Root::Map::ParallelMapV2::Zip[1]::ForeverRepeatg_y??"??!j??끢%@)?\??X3??1!??G*!@:Preprocessing2z
CIterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate???M+??!?:?i??#@)???Qј?1?@?5@:Preprocessing2?
SIterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice$??ŋ???!?3??b@)$??ŋ???1?3??b@:Preprocessing2E
Iterator::Root???????!%1p?&rR@)Yk(???1b5?v??@:Preprocessing2^
'Iterator::Root::Map::ParallelMapV2::Zip???????!j;?vd7:@)?@?v??1??7???@:Preprocessing2j
3Iterator::Root::Map::ParallelMapV2::Zip[0]::FlatMapS??%?Ѫ?!??I)@)'i????1?a7??!@:Preprocessing2|
EIterator::Root::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorl
dv???!"ES?^u@)l
dv???1"ES?^u@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 4.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?78.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?:\?8C@I??)r??S@Q2??T,1@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
      ??!             ??!       "	??_vO?????_vO???!??_vO???*      ??!       2      ??!       :	SX????@SX????@!SX????@B      ??!       J	v?~k'??v?~k'??!v?~k'??R      ??!       Z	v?~k'??v?~k'??!v?~k'??b      ??!       JGPUY?:\?8C@b q??)r??S@y2??T,1@