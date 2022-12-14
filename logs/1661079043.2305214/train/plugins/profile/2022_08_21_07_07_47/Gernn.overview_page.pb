?	??,zW@??,zW@!??,zW@      ??!       "_
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails??,zW@1?%?"????I??.Q??@r0*	?
ףp??@2J
Iterator::Root::Map؃I??	??!,E?7??P@)???1>???1Q?E?.?H@:Preprocessing2Y
"Iterator::Root::Map::ParallelMapV2U4??????!?A??1@)U4??????1?A??1@:Preprocessing2p
9Iterator::Root::Map::ParallelMapV2::Zip[1]::ForeverRepeat)_?BF??!?a?R??&@)Jy???1??Z-??#@:Preprocessing2^
'Iterator::Root::Map::ParallelMapV2::ZippxADj???!???U??>@)??H?H??1?0B-?@:Preprocessing2z
CIterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?S?<??!H@}?G@)Yk(???1????s-@:Preprocessing2j
3Iterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap?>???ʳ?!l%?A]'@)PT6??,??1&??@:Preprocessing2E
Iterator::Root?V?9?m??!܁j?^Q@)?-???=??1???YfA
@:Preprocessing2?
SIterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceQ?+?ϒ?!LW??4@)Q?+?ϒ?1LW??4@:Preprocessing2|
EIterator::Root::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?.񝘅?!T?,q~??)?.񝘅?1T?,q~??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?80.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI??y?V'T@Q?h>?b3@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
      ??!             ??!       "	?%?"?????%?"????!?%?"????*      ??!       2      ??!       :	??.Q??@??.Q??@!??.Q??@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q??y?V'T@y?h>?b3@?"7
sequential/dense_1/MatMulMatMulb	????!b	????0"7
sequential/dense_2/MatMulMatMul??j>ׄ?!???????0"[
+SGD/SGD/update_2/ResourceApplyKerasMomentumResourceApplyKerasMomentum?Ԑ?,_??!?fա??"[
+SGD/SGD/update_4/ResourceApplyKerasMomentumResourceApplyKerasMomentum?Ԑ?,_??!?h????"Y
)SGD/SGD/update/ResourceApplyKerasMomentumResourceApplyKerasMomentum	?OD???!??f'????"[
+SGD/SGD/update_7/ResourceApplyKerasMomentumResourceApplyKerasMomentum	?OD???!?D(;????"[
+SGD/SGD/update_3/ResourceApplyKerasMomentumResourceApplyKerasMomentumQ??
I???!???>Hα?"[
+SGD/SGD/update_6/ResourceApplyKerasMomentumResourceApplyKerasMomentum???R???!{.?2F??"[
+SGD/SGD/update_1/ResourceApplyKerasMomentumResourceApplyKerasMomentum?.??????!Q?,????"[
+SGD/SGD/update_5/ResourceApplyKerasMomentumResourceApplyKerasMomentum?.??????!'?+9?5??Q      Y@Y??}?	@@a"5?x+?P@qR{????R@y?娲????"?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?80.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?75.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 