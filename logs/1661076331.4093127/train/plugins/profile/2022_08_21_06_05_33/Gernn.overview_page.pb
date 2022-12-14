?	?? ?@?? ?@!?? ?@	!O????!O????!!O????"h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'?? ?@1?up?7???I?iQ?@Yt?5=((??r0*	j?t?v@2J
Iterator::Root::MapU??C???!"????R@)?|\*???1??ο?N@:Preprocessing2Y
"Iterator::Root::Map::ParallelMapV2B_z?sѨ?!???͔l+@)B_z?sѨ?1???͔l+@:Preprocessing2p
9Iterator::Root::Map::ParallelMapV2::Zip[1]::ForeverRepeat?hUK:ʡ?!?N??n?#@)x?????1S??A??@:Preprocessing2z
CIterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?h8en???!x?e?#@)????ח?1K?R??X@:Preprocessing2?
SIterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??? 4J??!?????	@)??? 4J??1?????	@:Preprocessing2|
EIterator::Root::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorb?[>??~?!
??0? @)b?[>??~?1
??0? @:Preprocessing2j
3Iterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap?_??D??!????K?'@) ?_>Y1|?1?e0'??:Preprocessing2^
'Iterator::Root::Map::ParallelMapV2::Zip?G?RE??!?~`??7@)?4?ׂ?{?1MO??????:Preprocessing2E
Iterator::Root^??N??!U???S@)JB"m?Ot?1?????q??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.9% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?86.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9!O????IB1'Մ?U@QaJ%??)@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
      ??!             ??!       "	?up?7????up?7???!?up?7???*      ??!       2      ??!       :	?iQ?@?iQ?@!?iQ?@B      ??!       J	t?5=((??t?5=((??!t?5=((??R      ??!       Z	t?5=((??t?5=((??!t?5=((??b      ??!       JGPUY!O????b qB1'Մ?U@yaJ%??)@?"7
sequential/dense_1/MatMulMatMul? Q???!? Q???0"7
sequential/dense_2/MatMulMatMul?乗????!??=	???0"7
sequential/dense_3/MatMulMatMul?乗????!*?]iM??0"[
+SGD/SGD/update_6/ResourceApplyKerasMomentumResourceApplyKerasMomentum	???(.??!י7?>r??"[
+SGD/SGD/update_2/ResourceApplyKerasMomentumResourceApplyKerasMomentum?k?T???!?????"[
+SGD/SGD/update_4/ResourceApplyKerasMomentumResourceApplyKerasMomentum????5???!x??HaŬ?"[
+SGD/SGD/update_3/ResourceApplyKerasMomentumResourceApplyKerasMomentum3??zkg??!?ϰ????"Y
)SGD/SGD/update/ResourceApplyKerasMomentumResourceApplyKerasMomentumn??MU??!????G???"[
+SGD/SGD/update_1/ResourceApplyKerasMomentumResourceApplyKerasMomentum?#&?T??!)??;?D??"[
+SGD/SGD/update_5/ResourceApplyKerasMomentumResourceApplyKerasMomentum?{ŝB??!	a?2???Q      Y@Ys?3R1?@a;#s?3Q@q??????@yo??-$???"?

device?Your program is NOT input-bound because only 0.9% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?86.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Turing)(: B 