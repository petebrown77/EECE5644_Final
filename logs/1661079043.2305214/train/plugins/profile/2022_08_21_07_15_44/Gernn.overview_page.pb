?	¤????@¤????@!¤????@	?S?>S@?S?>S@!?S?>S@"h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'¤????@1x&4I,)??I?`7l[?@Y???mnL??r0*	??Q?x}@2J
Iterator::Root::Map?W:?%??!`2:?XR@)???e???1~G?R?8L@:Preprocessing2Y
"Iterator::Root::Map::ParallelMapV2'?%??s??!???Cj?0@)'?%??s??1???Cj?0@:Preprocessing2p
9Iterator::Root::Map::ParallelMapV2::Zip[1]::ForeverRepeat?l??}??!;?? ? @)>?ɋL???1????nM@:Preprocessing2z
CIterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate հ????!?T!@)N???????1^'???F@:Preprocessing2^
'Iterator::Root::Map::ParallelMapV2::Zip???Y.??!p?N?8@)3?FY????1??m?a@:Preprocessing2?
SIterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?%8?????!???fJ?@)?%8?????1???fJ?@:Preprocessing2E
Iterator::Root? ?}????!d?|,??R@)???+,??1x WI>@:Preprocessing2|
EIterator::Root::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor???s??!k6?Wd???)???s??1k6?Wd???:Preprocessing2j
3Iterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap?o*Ral??!Fe??%@)h??5??1C?+6???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 2.5% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?78.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?S?>S@IAj??S@Q???ge?2@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
      ??!             ??!       "	x&4I,)??x&4I,)??!x&4I,)??*      ??!       2      ??!       :	?`7l[?@?`7l[?@!?`7l[?@B      ??!       J	???mnL?????mnL??!???mnL??R      ??!       Z	???mnL?????mnL??!???mnL??b      ??!       JGPUY?S?>S@b qAj??S@y???ge?2@?"Y
)SGD/SGD/update/ResourceApplyKerasMomentumResourceApplyKerasMomentum!???????!!???????"L
0gradient_tape/sequential/dense_3/MatMul/MatMul_1MatMulq??P???!Iְ?nŒ?"[
+SGD/SGD/update_9/ResourceApplyKerasMomentumResourceApplyKerasMomentum??Z?;??!*,^U????"L
.gradient_tape/sequential/dense_3/MatMul/MatMulMatMul?.O[?(??!???A?{??0"L
.gradient_tape/sequential/dense_1/MatMul/MatMulMatMul????(??!i?~????0"L
.gradient_tape/sequential/dense_2/MatMul/MatMulMatMul????(??!Kz?????0"7
sequential/dense_1/MatMulMatMul{?C?/??!???P?
??0"7
sequential/dense_2/MatMulMatMul{?C?/??!??NbM??0"[
+SGD/SGD/update_4/ResourceApplyKerasMomentumResourceApplyKerasMomentum?U?[???!?|
????"[
+SGD/SGD/update_2/ResourceApplyKerasMomentumResourceApplyKerasMomentumW48????!&?q??ж?Q      Y@Ys?3R1?@a;#s?3Q@q ??1?"V@y?????1??"?
device?Your program is NOT input-bound because only 2.5% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?78.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?88.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 