	???*??@???*??@!???*??@	??;?a?@??;?a?@!??;?a?@"h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'???*??@1ȶ8KI??IT?4???@Ynj??????r0*	I+??u@2J
Iterator::Root::Map?]=?1??!?????NO@)?J???11??_?D@:Preprocessing2Y
"Iterator::Root::Map::ParallelMapV2???E&??!??[m?_6@)???E&??1??[m?_6@:Preprocessing2z
CIterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??r-Z???!??? A?0@)?q75??1N??z?*@:Preprocessing2p
9Iterator::Root::Map::ParallelMapV2::Zip[1]::ForeverRepeatӠh?"??!??|??)@)?????1?SO??$@:Preprocessing2^
'Iterator::Root::Map::ParallelMapV2::Zip????'??!  d+??A@)I?V????1[
??m@:Preprocessing2?
SIterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?P?,??!͂???@)?P?,??1͂???@:Preprocessing2|
EIterator::Root::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensoro?EE?N??!?T~?T@)o?EE?N??1?T~?T@:Preprocessing2j
3Iterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap????W??!+?"?A3@))v4????1SU???@:Preprocessing2E
Iterator::Root?\5????! ?M?P@)??D2?x?1? ǽǣ??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 4.9% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?78.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9??;?a?@I???.?S@Q?z?E??0@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
      ??!             ??!       "	ȶ8KI??ȶ8KI??!ȶ8KI??*      ??!       2      ??!       :	T?4???@T?4???@!T?4???@B      ??!       J	nj??????nj??????!nj??????R      ??!       Z	nj??????nj??????!nj??????b      ??!       JGPUY??;?a?@b q???.?S@y?z?E??0@