	???xR@???xR@!???xR@	??)?n?@??)?n?@!??)?n?@"h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'???xR@1n3??x??I?Vд?:@Y?X????r0*?n??z@)      0=2J
Iterator::Root::Map9
p??!??/G?;Q@)a???U??1(u򭟝H@:Preprocessing2Y
"Iterator::Root::Map::ParallelMapV2 ??L??!????ճ3@) ??L??1????ճ3@:Preprocessing2p
9Iterator::Root::Map::ParallelMapV2::Zip[1]::ForeverRepeat)#. ?ҭ?!?p?p??+@)??Cl??1??ct??&@:Preprocessing2z
CIterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate???c[??!u?B??"@)?䠄???1?c??@:Preprocessing2?
SIterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(??&2s??!G?w?jO@)(??&2s??1G?w?jO@:Preprocessing2^
'Iterator::Root::Map::ParallelMapV2::Zip??0? ??!?Օr?<@)p_?Q??1i@aj?@:Preprocessing2j
3Iterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap3ı.n???!t?????'@)}w+Kt??1?/????@:Preprocessing2|
EIterator::Root::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?4?($???!?#???/@)?4?($???1?#???/@:Preprocessing2E
Iterator::RootˡE?????!???c?Q@)?RD?U???1?RǓ @:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 3.7% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?83.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9??)?n?@II<A?{?T@Q??~?.*@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
      ??!             ??!       "	n3??x??n3??x??!n3??x??*      ??!       2      ??!       :	?Vд?:@?Vд?:@!?Vд?:@B      ??!       J	?X?????X????!?X????R      ??!       Z	?X?????X????!?X????b      ??!       JGPUY??)?n?@b qI<A?{?T@y??~?.*@