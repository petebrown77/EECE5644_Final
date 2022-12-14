?	e?P3?Z@e?P3?Z@!e?P3?Z@	c??I]??c??I]??!c??I]??"h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'e?P3?Z@1Xr????I??U??B@YJ??%?L??r0*	B`??"?z@2J
Iterator::Root::Map>"?D??!??-?^P@)%?z?ۡ??1? n6HI@:Preprocessing2Y
"Iterator::Root::Map::ParallelMapV2???[??!E{˔??.@)???[??1E{˔??.@:Preprocessing2p
9Iterator::Root::Map::ParallelMapV2::Zip[1]::ForeverRepeat?tu?b???!?%U?)@)L???????1??}???$@:Preprocessing2z
CIterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateгY??ڪ?!???S(@)???X???1KT?@:Preprocessing2?
SIterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?XİØ??!,?? ??@)?XİØ??1,?? ??@:Preprocessing2^
'Iterator::Root::Map::ParallelMapV2::Zip_????@??!`?W?B?@)?A?p?-??1???V@:Preprocessing2E
Iterator::RootRԙ{H???!(;?[/Q@)2???z???1?i???
@:Preprocessing2j
3Iterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap	??Ln??!????-@)?E?x??1c?ҷ??@:Preprocessing2|
EIterator::Root::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor????7???!@???h?@)????7???1@???h?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 2.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?82.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9c??I]??IhFͭX?T@Qd %a?!/@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
      ??!             ??!       "	Xr????Xr????!Xr????*      ??!       2      ??!       :	??U??B@??U??B@!??U??B@B      ??!       J	J??%?L??J??%?L??!J??%?L??R      ??!       Z	J??%?L??J??%?L??!J??%?L??b      ??!       JGPUYc??I]??b qhFͭX?T@yd %a?!/@?"[
+SGD/SGD/update_6/ResourceApplyKerasMomentumResourceApplyKerasMomentumx%0*C??!x%0*C??"[
+SGD/SGD/update_4/ResourceApplyKerasMomentumResourceApplyKerasMomentum??j?????!?Z?k????"[
+SGD/SGD/update_8/ResourceApplyKerasMomentumResourceApplyKerasMomentum^Ӄ>΃?!?	??????"Y
)SGD/SGD/update/ResourceApplyKerasMomentumResourceApplyKerasMomentum*,<Z????!??j?????"[
+SGD/SGD/update_2/ResourceApplyKerasMomentumResourceApplyKerasMomentum*,<Z????!????Ϩ?"[
+SGD/SGD/update_1/ResourceApplyKerasMomentumResourceApplyKerasMomentumM??0<???!??"????"[
+SGD/SGD/update_7/ResourceApplyKerasMomentumResourceApplyKerasMomentum?d߳?q??!XY?6K??"7
sequential/dense_2/MatMulMatMul?d߳?q??!?E	5m???0"7
sequential/dense_1/MatMulMatMul?2H?7_??!JLR&T%??0"7
sequential/dense_3/MatMulMatMul? ?`?L??!jlh?ꎸ?0Q      Y@Ys?3R1?@a;#s?3Q@q?%#o?U@yj?h???"?
device?Your program is NOT input-bound because only 2.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?82.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?87.4% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 