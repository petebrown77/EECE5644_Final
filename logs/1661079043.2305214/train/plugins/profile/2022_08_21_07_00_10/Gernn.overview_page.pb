?	???xR@???xR@!???xR@	??)?n?@??)?n?@!??)?n?@"h
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
      ??!             ??!       "	n3??x??n3??x??!n3??x??*      ??!       2      ??!       :	?Vд?:@?Vд?:@!?Vд?:@B      ??!       J	?X?????X????!?X????R      ??!       Z	?X?????X????!?X????b      ??!       JGPUY??)?n?@b qI<A?{?T@y??~?.*@?"[
+SGD/SGD/update_2/ResourceApplyKerasMomentumResourceApplyKerasMomentum?@6?n??!?@6?n??"7
sequential/dense_1/MatMulMatMul??t??{??!D?Ւ?Ŗ?0"[
+SGD/SGD/update_1/ResourceApplyKerasMomentumResourceApplyKerasMomentum??FR?<??!!p??????"[
+SGD/SGD/update_4/ResourceApplyKerasMomentumResourceApplyKerasMomentumSNX???!??B?Mq??"[
+SGD/SGD/update_3/ResourceApplyKerasMomentumResourceApplyKerasMomentum??N4-Ӆ?!Y5V.???"Y
)SGD/SGD/update/ResourceApplyKerasMomentumResourceApplyKerasMomentum???dl??!??#????"[
+SGD/SGD/update_5/ResourceApplyKerasMomentumResourceApplyKerasMomentum??0????!??)??-??"L
.gradient_tape/sequential/dense_1/MatMul/MatMulMatMul?F???!?3#]õ?0"L
0gradient_tape/sequential/dense_1/MatMul/MatMul_1MatMul?F???!???K?X??"J
,gradient_tape/sequential/dense/MatMul/MatMulMatMul?<?(?B??!9=?0???0Q      Y@YK??">?@@a???`?P@qc??/-?V@yu={??"?
device?Your program is NOT input-bound because only 3.7% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?83.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?90.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 