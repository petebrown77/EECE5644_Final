?	???*??@???*??@!???*??@	??;?a?@??;?a?@!??;?a?@"h
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
      ??!             ??!       "	ȶ8KI??ȶ8KI??!ȶ8KI??*      ??!       2      ??!       :	T?4???@T?4???@!T?4???@B      ??!       J	nj??????nj??????!nj??????R      ??!       Z	nj??????nj??????!nj??????b      ??!       JGPUY??;?a?@b q???.?S@y?z?E??0@?"[
+SGD/SGD/update_2/ResourceApplyKerasMomentumResourceApplyKerasMomentumy?[?硆?!y?[?硆?"7
sequential/dense_1/MatMulMatMul?Cq4<b??!?f????0"[
+SGD/SGD/update_3/ResourceApplyKerasMomentumResourceApplyKerasMomentum7B??????!?R?Q¹??"[
+SGD/SGD/update_1/ResourceApplyKerasMomentumResourceApplyKerasMomentum?AUOs???!??%?'??"[
+SGD/SGD/update_4/ResourceApplyKerasMomentumResourceApplyKerasMomentum??j??x??!FS
$???"Y
)SGD/SGD/update/ResourceApplyKerasMomentumResourceApplyKerasMomentum??9j???!?QL??d??"[
+SGD/SGD/update_5/ResourceApplyKerasMomentumResourceApplyKerasMomentum?>??Ź??!p?A????"L
0gradient_tape/sequential/dense_1/MatMul/MatMul_1MatMult?ی???!>??9????"L
.gradient_tape/sequential/dense_1/MatMul/MatMulMatMul????z??!??ڍ???0"J
,gradient_tape/sequential/dense/MatMul/MatMulMatMul??2/6%??!? ?S????0Q      Y@YK??">?@@a???`?P@q????V@y???.???"?
device?Your program is NOT input-bound because only 4.9% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?78.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?88.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 