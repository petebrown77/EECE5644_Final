?	?/.Ui@?/.Ui@!?/.Ui@	
0Q8???
0Q8???!
0Q8???"h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'?/.Ui@1??????IO???*`@Y??[;Q??r0*	???Mb?y@2J
Iterator::Root::Mapj???v???!?U ?P@))????h??1l[?)J@:Preprocessing2Y
"Iterator::Root::Map::ParallelMapV2??QI????!?re??-@)??QI????1?re??-@:Preprocessing2p
9Iterator::Root::Map::ParallelMapV2::Zip[1]::ForeverRepeat	????Q??!xG?a?)(@)?'c|????13 ???#@:Preprocessing2z
CIterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate|?q7??!bnN?+@)jM??St??1??yI??#@:Preprocessing2?
SIterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice$??ŋ???!j+I(?@)$??ŋ???1j+I(?@:Preprocessing2^
'Iterator::Root::Map::ParallelMapV2::Zip??c?g^??!?? ??>?@)?V??????1ɠ??O?@:Preprocessing2j
3Iterator::Root::Map::ParallelMapV2::Zip[0]::FlatMap?HP???!?nJ?-60@)EJ?y??1??3(@:Preprocessing2|
EIterator::Root::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensoro?EE?N??!??by@)o?EE?N??1??by@:Preprocessing2E
Iterator::Rooth??W??!U???D0Q@)??Û5x?1??|?	??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 1.4% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?85.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9
0Q8???Iܾ?P?HU@Qcxo?*@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
      ??!             ??!       "	????????????!??????*      ??!       2      ??!       :	O???*`@O???*`@!O???*`@B      ??!       J	??[;Q????[;Q??!??[;Q??R      ??!       Z	??[;Q????[;Q??!??[;Q??b      ??!       JGPUY
0Q8???b qܾ?P?HU@ycxo?*@?"Y
)SGD/SGD/update/ResourceApplyKerasMomentumResourceApplyKerasMomentume?n? ??!e?n? ??"[
+SGD/SGD/update_2/ResourceApplyKerasMomentumResourceApplyKerasMomentum/?}???!?j=???"[
+SGD/SGD/update_4/ResourceApplyKerasMomentumResourceApplyKerasMomentum?e-S????!??f?m??"[
+SGD/SGD/update_5/ResourceApplyKerasMomentumResourceApplyKerasMomentumi?0?׃?!??ꊬ??"[
+SGD/SGD/update_6/ResourceApplyKerasMomentumResourceApplyKerasMomentum??U?CÃ?!T 0?[???"[
+SGD/SGD/update_1/ResourceApplyKerasMomentumResourceApplyKerasMomentum?Yz?l???!Ѷ?7???"[
+SGD/SGD/update_3/ResourceApplyKerasMomentumResourceApplyKerasMomentum?Yz?l???!??v????"[
+SGD/SGD/update_7/ResourceApplyKerasMomentumResourceApplyKerasMomentum?Yz?l???!??%?v0??"L
0gradient_tape/sequential/dense_2/MatMul/MatMul_1MatMul|??o????!?pyn???"L
0gradient_tape/sequential/dense_1/MatMul/MatMul_1MatMul??w?R??!?fʘ??Q      Y@Y??}?	@@a"5?x+?P@q'???? T@y7?1Ln??"?
device?Your program is NOT input-bound because only 1.4% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?85.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?80.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 