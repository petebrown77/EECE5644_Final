	N???=@N???=@!N???=@	??ݿ?@??ݿ?@!??ݿ?@"h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'N???=@1?W??I??I?<0??@Y֍wG???r0*	??ʡS~@2J
Iterator::Root::MapM.??:???!??V0H?P@)-]?6???1S+ñH@:Preprocessing2Y
"Iterator::Root::Map::ParallelMapV21_^?}t??!??;?2@)1_^?}t??1??;?2@:Preprocessing2p
9Iterator::Root::Map::ParallelMapV2::Zip[1]::ForeverRepeatP??????!;?x?Ǐ)@)?p??|#??19????
%@:Preprocessing2z
CIterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenates?<G仰?!????^?*@)	????=??1??i?ݵ"@:Preprocessing2?
SIterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?G6W?s??!???w@)?G6W?s??1???w@:Preprocessing2^
'Iterator::Root::Map::ParallelMapV2::Zip?jIG9???!Rw)z??@)?W?B???1??NmS?	@:Preprocessing2|
EIterator::Root::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor}w+Kt??!
?7??@)}w+Kt??1
?7??@:Preprocessing2E
Iterator::Rootd??A??!,9?u?Q@)`w???s??1h?(@:Preprocessing2j
3Iterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap??^?S??!?"??/@)?බ????1?????? @:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 7.9% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"?72.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9??ݿ?@I@???R@Qw?1|(?3@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
      ??!             ??!       "	?W??I???W??I??!?W??I??*      ??!       2      ??!       :	?<0??@?<0??@!?<0??@B      ??!       J	֍wG???֍wG???!֍wG???R      ??!       Z	֍wG???֍wG???!֍wG???b      ??!       JGPUY??ݿ?@b q@???R@yw?1|(?3@