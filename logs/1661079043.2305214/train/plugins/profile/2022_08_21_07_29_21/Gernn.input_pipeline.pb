	??$???@??$???@!??$???@	h??n??@h??n??@!h??n??@"h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'??$???@1[rPB??I????@Y? {???r0*	??ʡE?{@2J
Iterator::Root::Map??]?p??!#qEU?N@)??(]???1?x|???F@:Preprocessing2Y
"Iterator::Root::Map::ParallelMapV2;%?α?!???;?m/@);%?α?1???;?m/@:Preprocessing2z
CIterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate ???-???!?|M???1@)?,_?????1????)@:Preprocessing2p
9Iterator::Root::Map::ParallelMapV2::Zip[1]::ForeverRepeat6Φ#????!???? *@)t]?@???1 ???$%@:Preprocessing2?
SIterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?F?????!m???B?@)?F?????1m???B?@:Preprocessing2^
'Iterator::Root::Map::ParallelMapV2::Zip?͋_???!??? ?wB@)ꗈ?ο??1???O?@
@:Preprocessing2j
3Iterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap?!S>U??!?	?9?4@)-σ??v??1e??}<@:Preprocessing2|
EIterator::Root::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor'i????!??2?+?@)'i????1??2?+?@:Preprocessing2E
Iterator::Root????????!x`?2?O@)>?$@M-{?1﫧2????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 3.5% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?78.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9g??n??@I?$:??S@Q??DZ2@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
      ??!             ??!       "	[rPB??[rPB??![rPB??*      ??!       2      ??!       :	????@????@!????@B      ??!       J	? {???? {???!? {???R      ??!       Z	? {???? {???!? {???b      ??!       JGPUYg??n??@b q?$:??S@y??DZ2@