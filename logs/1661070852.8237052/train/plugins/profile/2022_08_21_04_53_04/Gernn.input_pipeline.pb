	??? @??? @!??? @      ??!       "_
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails??? @1Ҩ??6??IB?p?-?@r0*	??K7?Mq@2J
Iterator::Root::Map^????!fT$?*OP@)t_?lW??1,CٽG@:Preprocessing2Y
"Iterator::Root::Map::ParallelMapV2jin????!p?
?/3@)jin????1p?
?/3@:Preprocessing2p
9Iterator::Root::Map::ParallelMapV2::Zip[1]::ForeverRepeat??X?_"??!???((;/@):\?=셢?1q"???"*@:Preprocessing2?
SIterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice>?$@M-??!???7Q,@)>?$@M-??1???7Q,@:Preprocessing2z
CIterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate<?ݭ,љ?!?i?i?6"@);5?u??1?k?A@:Preprocessing2j
3Iterator::Root::Map::ParallelMapV2::Zip[0]::FlatMapH??Q???!??S?lI*@)?,C????1??R?j%@:Preprocessing2^
'Iterator::Root::Map::ParallelMapV2::Zipc&Q/?4??!? ?M_@@)'i????1???$??@:Preprocessing2|
EIterator::Root::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorDkE??|?!????a@)DkE??|?1????a@:Preprocessing2E
Iterator::Root?[z4???!???'Y?P@)U2 Tq?v?1fe?y?% @:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?85.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI?uJ??wU@Q?R?}Q@,@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
      ??!             ??!       "	Ҩ??6??Ҩ??6??!Ҩ??6??*      ??!       2      ??!       :	B?p?-?@B?p?-?@!B?p?-?@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?uJ??wU@y?R?}Q@,@